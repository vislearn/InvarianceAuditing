import torch
import torchvision.models as torchmodels
from collections import defaultdict
from typing import Callable
from warnings import warn
from fff.model.utils import guess_image_shape
from math import prod

WarmupFn = Callable[[int | tuple[int, int]], float]


class LossWeights():
    """Track configured loss weights and select active metrics by mode."""

    def __init__(self, loss_weights: dict[str, float | torch.Tensor],
                 warmup_func: WarmupFn, hparams,
                 current_epoch, training, valid_metrics):
        self.loss_weights = loss_weights
        self.effective_loss_weights = loss_weights.copy()
        self.compute_warmup = warmup_func
        self.current_epoch = current_epoch
        self.hparams = hparams
        self.training = training
        self.val_all_valid_metrics = not self.training and self.hparams.eval_all
        self.valid_metrics = valid_metrics

    def check_train_keys(self, *keys):
        return any(
            (loss_key in self.effective_loss_weights)
            and (
                torch.any(self.effective_loss_weights[loss_key] > 0)
                if torch.is_tensor(self.effective_loss_weights[loss_key])
                else self.effective_loss_weights[loss_key] > 0
            )
            for loss_key in keys
        )

    def apply_warmup(self, key, scale):
        # Keep configured weights unchanged so warmup can be recomputed per batch.
        self.effective_loss_weights[key] = self.loss_weights[key] * scale

    def check_keys(self, *keys):
        # Validation metrics:
        # if key is fiber loss we have to check
        # whether it should get evaluated in the current epoch
        if (not self.training
            and not any((loss_key != "fiber_loss") for loss_key in keys)
            and not (
                        (self.current_epoch + 1) % self.hparams.fiber_loss_every == 0
                        or self.current_epoch == self.hparams.max_epochs - 1
                    )
            ):
            return False
        elif self.val_all_valid_metrics:
            return any(
                (loss_key in self.valid_metrics for loss_key in keys)
            )
        # Compute only training losses
        else:
            return self.check_train_keys(*keys)


class ExteriorMetrics():
    """Compute configured autoencoder, density, and fiber metrics."""

    def __init__(self, FiberModel):
        self.FiberModel = FiberModel
        if "perceptual_loss" in defaultdict(
            float, self.FiberModel.hparams.loss_weights
        ) or "ae_perceptual_loss" in defaultdict(
            float, self.FiberModel.hparams.loss_weights
        ) or self.FiberModel.hparams.eval_all:
            vgg = torchmodels.vgg16(
                weights=torchmodels.VGG16_Weights.IMAGENET1K_V1)
            vgg.eval()
            self.vgg_features = vgg.features
            for param in self.vgg_features.parameters():
                param.requires_grad = False

        self.metric_dict = {
            "ae_reconstruction": self.ae_reconstruction,
            "ae_noisy_reconstruction": self.ae_noisy_reconstruction,
            "ae_lamb_reconstruction": self.ae_lamb_reconstruction,
            "ae_l1_reconstruction": self.ae_l1_reconstruction,
            "ae_cycle_loss": self.ae_cycle,
            "ae_perceptual_loss": self.ae_perceptual,
            "z 1D-Wasserstein-1": self.wasserstein,
            "z std": self.z_std,
            "z_sample_reconstruction": self.z_sample_reconstruction,
            "lambda": self.lambda_metric,
            "latent_reconstruction": self.latent_reconstruction,
            "latent_l1_reconstruction": self.latent_l1_reconstruction,
            "masked_reconstruction": self.masked_reconstruction,
            "cycle_loss": self.cycle,
            "perceptual_loss": self.perceptual,
            "nll": self.nll,
            "ae_elbo": self.ae_elbo,
            "ae_rec_fiber_loss": self.ae_rec_fiber,
            "fiber_loss": self.fiber,
            "jac_fiber_loss": self.jac_fiber,
            "diff_mse": self.diff_mse,
            "fm_loss": self.fm,
        }
        self.LossWeights = None

    def __call__(self, variables: dict, warmup_func: WarmupFn, training: bool,
                 current_epoch, batch_idx) -> tuple:
        # Each call gets a fresh effective-weight view because warmup depends on
        # the current epoch and batch position.
        variables["batch_idx"] = batch_idx
        self.LossWeights = LossWeights(
            variables["loss_weights"],
            warmup_func,
            self.FiberModel.hparams,
            current_epoch,
            training,
            self.FiberModel.valid_metrics
        )

        loss_values = {}

        ####################### AE Losses ####################################
        ae_loss_names = [
            name for name in self.FiberModel.valid_metrics if name.startswith("ae_")]
        for name in ae_loss_names:
            if self.LossWeights.check_keys(name):
                variables, loss_values = self.metric_dict[name](variables,
                                                                loss_values)

        ##################### Losses for density model #######################
        # nll and fff loss funcs should be computed first
        if self.LossWeights.check_keys("nll", "coarse_supervised"):
            variables, loss_values = self.nll(variables, loss_values)
        if self.LossWeights.check_keys("ff_nll"):
            if not training or (
                    self.FiberModel.hparams.exact_train_nll_every is not None and
                    batch_idx % self.FiberModel.hparams.exact_train_nll_every == 0):
                variables, loss_values = self.ff_nll_exact(variables,
                                                           loss_values,
                                                           training)
            else:
                variables, loss_values = self.ff_nll(variables, loss_values)

        # These losses are handled above because later metrics may reuse their
        # cached intermediate values.
        skipped_losses = ("ff_nll", "nll", "coarse_supervised")

        # Valid losses, posssible to compute
        density_loss_names = [
            name for name in self.FiberModel.valid_metrics if (
                not name.startswith("ae_") and name not in skipped_losses
            )
        ]
        for name in density_loss_names:
            if self.LossWeights.check_keys(name):
                variables, loss_values = self.metric_dict[name](
                    variables, loss_values)

        metrics = {k: loss_values.pop(k) for k in loss_values.copy().keys() if k in [
            "z 1D-Wasserstein-1", "z std", "lambda"]}
        # Compute loss as weighted loss
        metrics["loss"] = sum(
            (weight * loss_values[key]).mean(-1)
            for key, weight in self.LossWeights.effective_loss_weights.items()
            if self.LossWeights.check_train_keys(key) and (training or key in loss_values)
        )

        # Per-sample losses become scalar metrics; scalar diagnostics pass
        # through unchanged. Other shapes indicate a metric implementation bug.
        invalid_losses = []
        for key in loss_values:
            # One value per key
            if loss_values[key].shape == (variables["x_noisy"].shape[0],):
                metrics[key] = loss_values[key].mean(-1)
            elif loss_values[key].ndim == 0:
                metrics[key] = loss_values[key]
            else:
                invalid_losses.append(key)
        if len(invalid_losses) > 0:
            raise ValueError(f"Invalid loss shapes for {invalid_losses}")

        return metrics, self.LossWeights.effective_loss_weights

    def compute_z(self, variables):
        # Cache each representation in variables so multiple metrics share the
        # same forward pass within one batch.
        if "z" not in variables:
            z, mu, logvar = self.FiberModel.encode_lossless(
                variables["x_noisy"],
                variables["condition"],
                return_only_x=False
            )
            variables.update({"z": z, "mu": mu, "logvar": logvar})
        return variables

    def compute_x1(self, variables):
        if "x1" not in variables:
            variables = self.compute_z(variables)
            x1 = self.FiberModel.decode_lossless(
                variables["z"], variables["condition"])
            variables["x1"] = x1
        return variables

    def compute_c1(self, variables):
        if "c1" not in variables:
            variables = self.compute_x1(variables)
            variables["c1"] = self.FiberModel.subject_model.encode(
                variables["x1"])
        return variables

    def compute_z_dense(self, variables):
        if "z_dense" not in variables:
            variables = self.compute_z(variables)
            # Density losses should not backpropagate through the lossless model.
            z_dense = self.FiberModel.encode_density(
                variables["z"].detach(), variables["condition"])
            variables["z_dense"] = z_dense
        return variables

    def compute_z_marginal(self, variables):
        if "z_marginal" not in variables:
            variables = self.compute_z_dense(variables)
            variables["z_marginal"] = variables["z_dense"].reshape(-1)
        return variables

    def compute_z1(self, variables):
        if "z1" not in variables:
            variables = self.compute_z_dense(variables)
            variables["z1"] = self.FiberModel.decode_density(
                variables["z_dense"], variables["condition"])
        return variables

    def compute_x_full_recon(self, variables):
        if "x_full_recon" not in variables:
            variables = self.compute_z1(variables)
            x_full_recon = self.FiberModel.decode_lossless(
                variables["z1"], variables["condition"])
            variables["x_full_recon"] = x_full_recon
        return variables

    def compute_x_random(self, variables):
        if "x_random" not in variables:
            x = variables["x_noisy"]
            c = variables["condition"]
            try:
                # Some latent distributions accept conditioning during sampling;
                # unconditional distributions expose the shorter signature.
                z_dense_random = self.FiberModel.get_latent(
                    x.device).sample((x.shape[0],), c)
            except TypeError:
                z_dense_random = self.FiberModel.get_latent(
                    x.device).sample((x.shape[0],))
            if isinstance(z_dense_random, tuple):
                z_dense_random, c_random = z_dense_random
            else:
                c_random = c
            z_random = self.FiberModel.sample_density(z_dense_random, c_random)
            x_random = self.FiberModel.decode_lossless(z_random, c_random)
            variables.update({"x_random": x_random,
                              "z_random": z_random,
                              "z_dense_random": z_dense_random,
                              "c_random": c_random})
        return variables

    def compute_c_samples(self, variables):
        if "c_samples" not in variables:
            variables = self.compute_x_random(variables)
            x_random_sm = variables["x_random"]
            variables["c_samples"] = self.FiberModel.subject_model.encode(
                x_random_sm)
        return variables

    def ae_reconstruction(self, variables, loss_values):
        variables = self.compute_x1(variables)
        loss_values["ae_reconstruction"] = self._L2_distance(
            variables["x0"], variables["x1"])
        return variables, loss_values

    def ae_noisy_reconstruction(self, variables, loss_values):
        variables = self.compute_x1(variables)
        loss_values["ae_noisy_reconstruction"] = self._L2_distance(
            variables["x_noisy"], variables["x1"])
        return variables, loss_values

    def ae_l1_reconstruction(self, variables, loss_values):
        variables = self.compute_x1(variables)
        loss_values["ae_l1_reconstruction"] = self._L1_distance(
            variables["x0"], variables["x1"])
        return variables, loss_values

    def ae_lamb_reconstruction(self, variables, loss_values):
        variables = self.compute_x1(variables)
        loss_values["ae_lamb_reconstruction"] = self._L2_lamb_distance(
            variables["x_noisy"], variables["x1"]
        )
        return variables, loss_values

    def ae_perceptual(self, variables, loss_values):
        variables = self.compute_x1(variables)
        loss_values["ae_perceptual_loss"] = self._perceptual_distance(
            variables["x_noisy"], variables["x1"])
        return variables, loss_values

    def ae_rec_fiber(self, variables, loss_values):
        variables = self.compute_c1(variables)
        loss_values["ae_rec_fiber_loss"] = self._reduced_L2_distance(
            variables["condition"], variables["c1"])
        return variables, loss_values

    def ae_elbo(self, variables, loss_values):
        variables = self.compute_z(variables)
        mu = variables["mu"]
        logvar = variables["logvar"]
        loss_values["ae_elbo"] = -0.5 * torch.sum(
            (1.0 + logvar - torch.pow(mu, 2) - torch.exp(logvar)), -1
        )
        return variables, loss_values

    def ae_cycle(self, variables, loss_values):
        variables = self.compute_x1(variables)
        x1_detached = variables["x1"].detach()
        z_cycle = self.FiberModel.encode_lossless(
            x1_detached, variables["condition"], return_only_x=True)
        loss_values["ae_cycle_loss"] = self._L2_distance(
            variables["z"], z_cycle)
        return variables, loss_values

    def nll(self, variables, loss_values):
        # NLL for architectures whose density encoder returns a log determinant.
        variables = self.compute_z(variables)
        c = variables["condition"]
        z = variables["z"]
        if self.LossWeights.check_keys("coarse_supervised"):
            c_n = c + torch.randn_like(c) * \
                self.FiberModel.hparams.condition_noise
        else:
            c_n = c
        z_combined, log_det = self.FiberModel.encode_density(
            z.detach(), c_n, jac=True)
        if isinstance(z_combined, tuple):
            z_dense, z_coarse = z_combined
            if self.LossWeights.check_keys("coarse_supervised"):
                loss_values["coarse_supervised"] = self._L2_distance(
                    c, z_coarse
                )
        else:
            z_dense = z_combined
        variables["z_dense"] = z_dense
        log_prob = self.FiberModel._latent_log_prob(z_dense, c_n)
        loss_values["nll"] = -(log_prob + log_det)
        return variables, loss_values

    def ff_nll_exact(self, variables, loss_values, training=False):
        # Exact FFF likelihood is expensive, so training may run it periodically.
        key = "ff_nll_exact" if training else "ff_nll"
        if training or (
            self.FiberModel.hparams.skip_val_nll is not True
            and (
                self.FiberModel.hparams.skip_val_nll is False
                or (
                    isinstance(self.FiberModel.hparams.skip_val_nll, int)
                    and variables["batch_idx"] < self.FiberModel.hparams.skip_val_nll
                )
            )
        ):
            variables = self.compute_z(variables)
            z = variables["z"]
            c = variables["condition"]
            with torch.no_grad():
                log_prob_result = self.FiberModel.exact_log_prob(
                    x=z.detach(), c=c, jacobian_target="both"
                )
            variables["z_dense"] = log_prob_result.z
            variables["z1"] = log_prob_result.x1
            deq_vol_change = variables["dequantization_jac"]
            loss_values[key] = -log_prob_result.log_prob - deq_vol_change
            loss_values.update(log_prob_result.regularizations)
        else:
            self.LossWeights.effective_loss_weights["ff_nll"] = 0
        return variables, loss_values

    def ff_nll(self, variables, loss_values):
        # Use the cheaper surrogate likelihood between exact evaluations.
        nll_warmup = self.LossWeights.compute_warmup(
            self.FiberModel.hparams.warm_up_epochs)
        self.LossWeights.apply_warmup("ff_nll", nll_warmup)
        if self.LossWeights.check_keys("ff_nll"):
            variables = self.compute_z(variables)
            z = variables["z"]
            log_prob_result = self.FiberModel.surrogate_log_prob(
                x=z.detach(), c=variables["condition"])
            variables["z_dense"] = log_prob_result.z
            variables["z1"] = log_prob_result.x1
            deq_vol_change = variables["dequantization_jac"]
            loss_values["ff_nll"] = -log_prob_result.log_prob - deq_vol_change
            loss_values.update(log_prob_result.regularizations)
        return variables, loss_values

    def wasserstein(self, variables, loss_values):
        variables = self.compute_z_dense(variables)
        with torch.no_grad():
            # Flatten all latent coordinates to compare their aggregate marginal.
            z_marginal = variables["z_dense"].reshape(-1)
            z_gauss = torch.randn_like(z_marginal)

            z_marginal_sorted = z_marginal.sort().values
            z_gauss_sorted = z_gauss.sort().values

            loss_values["z 1D-Wasserstein-1"] = (
                (z_marginal_sorted - z_gauss_sorted).abs().mean()
            )
        return variables, loss_values

    def z_std(self, variables, loss_values):
        variables = self.compute_z_dense(variables)
        with torch.no_grad():
            z_marginal = variables["z_dense"].reshape(-1)
            loss_values["z std"] = torch.std(z_marginal)
        return variables, loss_values

    def latent_reconstruction(self, variables, loss_values):
        variables = self.compute_z1(variables)
        loss_values["latent_reconstruction"] = self._L2_distance(
            variables["z"], variables["z1"])
        return variables, loss_values

    def latent_l1_reconstruction(self, variables, loss_values):
        variables = self.compute_z1(variables)
        loss_values["latent_l1_reconstruction"] = self._L1_distance(
            variables["z"], variables["z1"])
        return variables, loss_values

    def cycle(self, variables, loss_values):
        variables = self.compute_z1(variables)
        z1_detached = variables["z1"].detach()
        z_dense1 = self.FiberModel.encode_density(
            z1_detached, variables["condition"])
        if isinstance(z_dense1, tuple):
            z_dense1, _ = z_dense1
        loss_values["cycle_loss"] = self._L2_distance(
            variables["z_dense"], z_dense1)
        return variables, loss_values

    def perceptual(self, variables, loss_values):
        variables = self.compute_x_full_recon(variables)
        loss_values["perceptual_loss"] = self._perceptual_distance(
            variables["x_noisy"], variables["x_full_recon"])
        return variables, loss_values

    def diff_mse(self, variables, loss_values):
        variables = self.compute_z(variables)
        z = variables["z"]
        c = variables["condition"]
        t = torch.randint(0, 1000, (z.size(0),), device=z.device).long()
        epsilon = self.FiberModel.get_latent(z.device).sample((z.shape[0],))
        epsilon_pred = self.FiberModel.density_model[0].diffuse_and_reverse(
            t, epsilon, z.detach(), c)
        loss_values["diff_mse"] = self._L2_distance(
            epsilon_pred, epsilon.detach())
        return variables, loss_values

    def fm(self, variables, loss_values):
        variables = self.compute_z(variables)
        z = variables["z"]
        c = variables["condition"]
        t = torch.rand(z.shape[0], device=z.device)
        z_fm = self.FiberModel.get_latent(z.device).sample((z.shape[0],))
        if self.FiberModel.hparams.cfg:
            p_uncond = self.FiberModel.hparams.cfg.get("p_unconditional", 0.1)
            if p_uncond > 0:
                # Classifier-free guidance trains on both conditional and null
                # conditions by randomly masking part of the batch.
                mask = torch.rand(c.shape[0], device=c.device) < p_uncond
                c[mask] = self.FiberModel.get_null_condition(c[mask])
        loss_values["fm_loss"] = self.FiberModel.density_model[0].compute_fm_loss(
            t, z_fm, z.detach(), self.FiberModel.embed_condition(c)
        )
        return variables, loss_values

    def masked_reconstruction(self, variables, loss_values):
        variables = self.compute_z(variables)
        z = variables["z"]
        c = variables["condition"]
        x = variables["x_noisy"]
        z_dense = variables["z_dense"]
        latent_mask = torch.zeros(
            z.shape[0],
            self.FiberModel.latent_dim,
            device=z.device)
        latent_mask[:, : self.FiberModel.hparams.reconstruct_dims] = 1
        z_masked_dense = z_dense * latent_mask
        x_zmask = self.FiberModel.decode(z_masked_dense, c)
        loss_values["masked_reconstruction"] = self._L2_distance(x, x_zmask)
        return variables, loss_values

    def z_sample_reconstruction(self, variables, loss_values):
        # variables = compute_z_dense_random(variables)
        variables = self.compute_x_random(variables)
        z_dense_random = variables["z_dense_random"]
        c_random = variables["c_random"]
        x_random = variables["x_random"]
        try:
            # Sanity checks might fail for random data
            z1_random = self.FiberModel.encode(x_random, c_random)
            if isinstance(z1_random, tuple):
                z1_random, _ = z1_random
            loss_values["z_sample_reconstruction"] = self._L2_distance(
                z_dense_random, z1_random
            )
        except Exception as e:
            warn(
                "Error in computing z_sample_reconstruction, setting to nan. Error: " +
                str(e))
            loss_values["z_sample_reconstruction"] = float("nan") * torch.ones(
                x_random.shape[0]
            )
        return variables, loss_values

    def fiber(self, variables, loss_values):
        fl_warmup = self.LossWeights.compute_warmup(
            self.FiberModel.hparams.warm_up_fiber)
        self.LossWeights.apply_warmup("fiber_loss", fl_warmup)
        if self.LossWeights.check_keys("fiber_loss"):
            # Try whether the model learns fibers and therefore has a subject
            # model
            try:
            # There might be no subject model
                variables = self.compute_c_samples(variables)
                loss_values["fiber_loss"] = self._reduced_L2_distance(
                    variables["c_random"], variables["c_samples"]
                )
            except Exception as e:
                warn(
                    "Error in computing fiber loss, setting to nan. Error: " +
                    str(e))
                loss_values["fiber_loss"] = float("nan") * torch.ones(
                    variables["x_noisy"].shape[0])
        return variables, loss_values

    def jac_fiber(self, variables, loss_values):
        fl_warmup = self.LossWeights.compute_warmup(
            self.FiberModel.hparams.warm_up_fiber)
        self.LossWeights.apply_warmup("jac_fiber_loss", fl_warmup)
        # Only datasets that ship a per-sample subject-model Jacobian as a third
        # batch element can define this metric; without one there is nothing to
        # normalise by, so skip it rather than logging a nan every batch.
        if (self.LossWeights.check_keys("jac_fiber_loss")
                and variables["jac_sm"] is not None):
            try:
                # There might be no subject model
                variables = self.compute_c_samples(variables)
                loss_values["jac_fiber_loss"] = self._L2_jac_distance(
                    variables["c_random"], variables["c_samples"],
                    variables["jac_sm"], epsilon=0.01
                )
            except Exception as e:
                warn(
                    "Error in computing jac fiber loss, setting to nan. Error: " +
                    str(e))
                loss_values["jac_fiber_loss"] = float("nan") * torch.ones(
                    variables["x_noisy"].shape[0])
        return variables, loss_values

    def lambda_metric(self, variables, loss_values):
        loss_values["lambda"] = self.FiberModel.lamb
        return variables, loss_values

    def _perceptual_distance(self, x_orig, x1):
        perceptual_distance = 0
        vgg_input = x1.reshape(-1, *guess_image_shape(x1.shape[1]))
        if vgg_input.shape[1] == 1:
            vgg_input = vgg_input.repeat(1, 3, 1, 1)
        vgg_target = x_orig.reshape(-1, *guess_image_shape(x_orig.shape[1]))
        if vgg_target.shape[1] == 1:
            vgg_target = vgg_target.repeat(1, 3, 1, 1)
        for i, m in self.vgg_features._modules.items():
            vgg_input = m(vgg_input)
            vgg_target = m(vgg_target)
            if i in ["3", "8", "15", "22"]:
                # Sum normalized feature distances from the selected VGG blocks.
                perceptual_distance += self._L1_distance(
                    vgg_input, vgg_target) / prod(vgg_input.shape[1:])
        return perceptual_distance

    def _L2_distance(self, a, b):
        return torch.sqrt(torch.sum((a - b).reshape(a.shape[0], -1) ** 2, -1))

    def _L1_distance(self, a, b):
        return torch.mean(torch.abs(a - b).reshape(a.shape[0], -1), -1)

    def _L2_lamb_distance(self, a, b):
        return (
            torch.sum((a - b).reshape(a.shape[0], -1) ** 2, -1)
        ) / self.FiberModel.lamb + torch.log(self.FiberModel.lamb)

    def _sqr_distance(self, a, b):
        return torch.sum((a - b).reshape(a.shape[0], -1) ** 2, -1)

    def _reduced_L2_distance(self, a, b):
        # Root-mean-square over the condition dimensions, not the plain norm.
        # The fiber losses are normalised so that their scale -- and therefore
        # the meaning of the configured fiber-loss weight -- does not depend on
        # how many outputs the subject model has.
        return torch.sqrt(
            torch.sum((a - b).reshape(a.shape[0], -1) ** 2, -1) / float(a.shape[-1])
        )

    def _L2_jac_distance(self, a, b, jac, epsilon=0.01):
        return (
            torch.sqrt(
                torch.sum((a - b).reshape(a.shape[0], -1) ** 2, -1) / float(a.shape[-1])
            )
            / jac
            / epsilon
        )
