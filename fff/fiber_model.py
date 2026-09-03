from copy import deepcopy
from collections import namedtuple, defaultdict
from math import prod

from warnings import warn

import torch
import lightning_trainable
from lightning_trainable.trainable.trainable import SkipBatch

from fff.base import (
    FreeFormBaseHParams,
    FreeFormBase,
    VolumeChangeResult,
    build_model,
    rand_log_uniform,
    soft_heaviside,
)
from fff.lossless_ae import LosslessAE, LosslessAEHParams

from fff.base import LogProbResult
from fff.subject_model import SubjectModel
from fff.loss import volume_change_surrogate
from fff.metrics import ExteriorMetrics
from fff.utils.checkpoint import default_map_location
from fff.utils.jacobian import compute_jacobian
from fff.data import get_model_path, resolve
from fff.evaluate.plot_fiber_model import *


ConditionedBatch = namedtuple(
    "ConditionedBatch",
    ["x0", "x_noisy", "loss_weights", "condition", "dequantization_jac", "jac_sm"],
)

COUPLING_FLOW_NAMES = {
    "fff.model.DenoisingFlow",
    "fff.model.MultilevelFlow",
    "fff.model.INN",
}


class FiberModelHParams(FreeFormBaseHParams):
    cond_dim: int = 0
    compute_c_on_fly: bool = False
    condition_noise: float = 0.0
    
    density_model: list
    load_density_model_path: str | None = None
    reconstruct_dims: int = 1
    cfg: dict = {}

    load_subject_model: bool = False
    sm_input_transform: str | None = None
    sm_empty_condition: bool = False
    add_noise_for_sm: bool = False
    
    lossless_ae: dict | LosslessAEHParams | None = None
    load_lossless_ae_path: str | None = None
    train_lossless_ae: bool = True
    ae_conditional: bool = False
    ae_deterministic_encode: bool | None = None
    vae: bool = False

    val_every_n_epoch: int = 1
    eval_all: bool = True
    fiber_loss_every: int = 1
    cnew_every: int = 1  # deprecated and not used anymore

    warm_up_fiber: int | list = 0
    warm_up_epochs: int | list = 0

    def __post__init__(self):
        # delete models list
        if "models" in self:
            del self["models"]

    @classmethod
    def _migrate_hparams(cls, hparams):
        """Bring hyperparameters written by older revisions up to the current schema.

        Every released checkpoint predates one or more of these renames, so this
        runs before validation and lets `load_from_checkpoint` work directly.
        """
        # The beta schedule moved from the top level into the density model spec.
        moved = {
            new_key: hparams.pop(old_key)
            for old_key, new_key in (
                ("diffusion_betas_max", "betas_max"),
                ("diffusion_beta_schedule", "beta_schedule"),
            )
            if old_key in hparams
        }
        for spec in hparams.get("density_model") or []:
            if not isinstance(spec, dict):
                continue
            name = spec.get("name", "")
            if name.endswith("DiffusionModel"):
                # Older checkpoints store the schedule as a materialised beta
                # tensor; the spec keys below replace it.
                spec.pop("betas", None)
                for key, value in moved.items():
                    spec.setdefault(key, value)
            elif name.endswith("FlowMatching"):
                # The flow matching model gained a nested network spec and renamed
                # its schedule; `sigma` became part of the schedule itself.
                if "interpolation" in spec:
                    spec["interpolation_schedule"] = spec.pop("interpolation")
                spec.pop("sigma", None)
                if "layers_spec" in spec:
                    spec.setdefault("network_hparams", {})["layers_spec"] = spec.pop("layers_spec")

        # Older checkpoints give `lossless_ae` as the bare model spec list.
        if isinstance(hparams.get("lossless_ae"), list):
            hparams["lossless_ae"] = {"model_spec": hparams["lossless_ae"]}

        # `cond_dim` used None where it now uses 0 for "unconditional".
        if hparams.get("cond_dim", 0) is None:
            hparams["cond_dim"] = 0

        # Recorded checkpoint paths are relative to the repository data directory.
        for key in ("load_lossless_ae_path", "load_density_model_path"):
            if hparams.get(key):
                hparams[key] = resolve(hparams[key])

        return hparams


class FiberModel(FreeFormBase):
    """
    This class abstracts the functionalities of a model which learns
    the fibers of a "subject model".
    """

    hparams: FiberModelHParams

    def __init__(self, hparams: FiberModelHParams | dict):
        if not isinstance(hparams, FiberModelHParams):
            hparams = FiberModelHParams(**hparams)
        assert (
            not (hparams.cond_dim == 0 and hparams.compute_c_on_fly)
        ), "You have to provide cond_dim if compute_c_on_fly is true"
        super().__init__(hparams)

        # Add learnable parameter for standard deviation for vae training
        self.lamb = torch.nn.Parameter(torch.ones(1), requires_grad=True)
        if self.hparams.cfg:
            c_temp = torch.empty(1, self.cond_dim)
            self.get_null_condition(c_temp)

    def init_models(self):
        self._configure_valid_metrics()
        self._load_subject_model()
        self._build_condition_embedder()
        self._build_lossless_ae()
        self._build_density_model()

        if not self.hparams.train_lossless_ae:
            for param in self.lossless_ae.parameters():
                param.requires_grad = False

        self.LossComputer = ExteriorMetrics(self)

    def _configure_valid_metrics(self):
        universal_metrics = [
            "ae_reconstruction",
            "ae_noisy_reconstruction",
            "ae_lamb_reconstruction",
            "ae_l1_reconstruction",
            "ae_cycle_loss",
            "z 1D-Wasserstein-1",
            "z std",
            "z_sample_reconstruction",
            "lambda",
        ]
        reconstruction_metrics = [
            "latent_reconstruction",
            "latent_l1_reconstruction",
            "masked_reconstruction",
            "cycle_loss",
        ]
        if self._data_dim >= 32*32*3:
            reconstruction_metrics += ["perceptual_loss"]
            universal_metrics += ["ae_perceptual_loss"]
        density_names = {
            model_hparams["name"]
            for model_hparams in self.hparams.density_model
        }
        if density_names == {"fff.model.Identity"}:
            self.density_model_type = None
            self.valid_metrics = universal_metrics
        elif density_names & COUPLING_FLOW_NAMES:
            self.density_model_type = "inn"
            if not density_names <= COUPLING_FLOW_NAMES:
                raise ValueError(
                    "Coupling Flows cannot be mixed with other models for now."
                )
            self.valid_metrics = (universal_metrics
                                  + reconstruction_metrics
                                  + ["nll", "coarse_supervised"])
        elif density_names == {"fff.model.DiffusionModel"}:
            assert (
                len(self.hparams.density_model) == 1
            ), "Diffusion model must be the only model in the density model"
            self.density_model_type = "diffusion"
            if self.hparams.cfg:
                raise NotImplementedError("Diffusion model cfg not implemented")
            self.valid_metrics = universal_metrics + ["diff_mse"]
        elif density_names == {"fff.model.FlowMatching"}:
            assert (
                len(self.hparams.density_model) == 1
            ), "Flow matching model must be the only model in the density model"
            self.density_model_type = "flow_matching"
            if (
                not self.hparams.density_model[0].get("conditional", False)
                and self.hparams.cfg
            ):
                raise ValueError("Flow matching model must be conditional to use cfg")
            self.valid_metrics = universal_metrics + ["fm_loss"]
        else:
            self.density_model_type = "fff"
            if "nll" in self.hparams.loss_weights.keys():
                print("WARNING: Changing nll loss to ff_nll, since model is no coupling flow") 
                self.hparams.loss_weights["ff_nll"] = self.hparams.loss_weights.pop("nll")
            self.valid_metrics = universal_metrics + reconstruction_metrics + ["ff_nll"]

        if self.hparams.vae:
            self.vae = True
            self.valid_metrics.append("ae_elbo")

    def _validate_loss_weights(self):
        invalid_losses = set(self.hparams.loss_weights) - set(self.valid_metrics)
        if invalid_losses:
            raise ValueError(
                "Losses not applicable to this model: "
                + ", ".join(sorted(invalid_losses))
            )

    def _load_subject_model(self):
        if self.hparams.load_subject_model:
            self.valid_metrics += ("fiber_loss", "jac_fiber_loss", "ae_rec_fiber_loss")
            print("loading subject_model")
            sm_dir = get_model_path(**self.hparams["data_set"])
            self.subject_model = SubjectModel(
                sm_dir,
                self.hparams.data_set.subject_model_type,
                fixed_transform=self.hparams.sm_input_transform,
                empty_condition=self.hparams.sm_empty_condition,
            )
            self.subject_model.eval()
            for param in self.subject_model.parameters():
                param.requires_grad = False
        else:
            self.subject_model = None
        self._validate_loss_weights()

    def _build_condition_embedder(self):
        cond_emb_out_dim = self._data_cond_dim
        if self.hparams.use_condition_decoder:
            cond_emb_out_dim = prod(self.hparams.cond_embedding_shape)
        elif self.hparams.cond_dim is not None:
            cond_emb_out_dim = self.hparams.cond_dim

        # Build condition embedder
        self.condition_embedder = build_model(
            self.hparams.condition_embedder,
            cond_emb_out_dim,
            0,
        )
        if self.condition_embedder is not None:
            assert not any(
                [
                    loss == "coarse_supervised"
                    for loss, _ in self.hparams.loss_weights.items()
                ]
            ), "coarse_supervised loss is not applicable for a model with condition embedder."
            for model in self.condition_embedder:
                if not self.hparams.use_condition_decoder:
                    del model.model.decoder
                else:
                    del model.model.encoder

    def _build_lossless_ae(self):
        ae_hparams = {}
        if self.hparams.load_lossless_ae_path is None:
            if self.hparams.lossless_ae is None:
                raise ValueError("No lossless_ae specified!")
            ae_hparams = self.hparams.lossless_ae
        elif self.hparams.lossless_ae is not None:
            warn("Overwriting model_spec from config with loaded model!")
        ae_hparams["data_dim"] = self.data_dim
        if self.hparams.ae_conditional:
            ae_hparams["cond_dim"] = self.ae_cond_dim
        ae_hparams["vae"] = self.vae
        if ae_hparams.get("path") is not None:
            raise (
                RuntimeError(
                    "Specificy pretrained models via the load_lossless_ae_path flag, not the path key in lossless_ae hparams"
                )
            )
        ae_hparams["path"] = self.hparams.load_lossless_ae_path
        ae_hparams["train"] = self.hparams.train_lossless_ae
        self.lossless_ae = LosslessAE(ae_hparams)

    def _build_density_model(self):
        self.density_model = build_model(
            self.hparams.density_model,
            self.lossless_ae.latent_dim,
            self.cond_dim,
        )
        if self.hparams.load_density_model_path:
            print("load density_model checkpoint")
            checkpoint = torch.load(self.hparams.load_density_model_path,
                                    map_location=default_map_location())
            density_model_weights = {
                k[14:]: v
                for k, v in checkpoint["state_dict"].items()
                if k.startswith("density_model.")
            }
            self.density_model.load_state_dict(density_model_weights)

    @property
    def latent_dim(self):
        if self.density_model_type:
            return self.density_model[-1].hparams.latent_dim
        else:
            return self.lossless_ae.latent_dim

    @property
    def cond_dim(self):
        if self.condition_embedder is not None:
            if self.hparams.use_condition_decoder:
                return self.hparams.cond_embedding_shape[0]
            else:
                return self.condition_embedder[-1].hparams.latent_dim
        else:
            return self.ae_cond_dim

    @property
    def ae_cond_dim(self):
        if self.hparams.compute_c_on_fly:
            assert self.subject_model != None, "No subject model loaded!"
            return self.hparams.cond_dim
        else:
            return self._data_cond_dim

    def is_conditional(self):
        return self.cond_dim != 0

    def embed_condition(self, c):
        if self.condition_embedder is not None:
            if self.hparams.use_condition_decoder:
                for model in self.condition_embedder[::-1]:
                    c = model.decode(
                        c, torch.empty((c.shape[0], 0), device=c.device, dtype=c.dtype)
                    )
            else:
                for model in self.condition_embedder:
                    c = model.encode(
                        c, torch.empty((c.shape[0], 0), device=c.device, dtype=c.dtype)
                    )
        return c

    def encode_lossless(self, x, c, return_only_x=True, return_codebook_loss=False):
        deterministic = self.hparams.ae_deterministic_encode
        if deterministic is None:
            if not self.training:
                deterministic = True
            else:
                deterministic = False
        kwargs = {"return_only_x": return_only_x, "deterministic": deterministic}
        if return_codebook_loss:
            kwargs["return_codebook_loss"] = return_codebook_loss
        return self.lossless_ae.encode(x, c, **kwargs)

    def encode_density(self, z, c, jac=False, **kwargs):
        c = self.embed_condition(c)
        jacs = []
        for net in self.density_model:
            z = net.encode(z, c, **kwargs)
            if isinstance(z, tuple):
                z, jac_i = z
                jacs.append(jac_i)
        if jac:
            z = z, torch.sum(torch.stack(jacs, dim=1), dim=1)
        return z

    def encode(self, x, c):
        z_dense = self.encode_density(self.encode_lossless(x, c, return_only_x=True), c)
        return z_dense

    def decode_lossless(self, z, c):
        return self.lossless_ae.decode(z, c)

    def decode_density(self, z_dense, c, **kwargs):
        # c = self.unflatten_ce(c).unsqueeze(1)
        c = self.embed_condition(c)
        for net in self.density_model:
            z_dense = net.decode(z_dense, c, **kwargs)
        return z_dense

    def decode(self, z_dense, c):
        x = self.decode_lossless(self.decode_density(z_dense, c), c)
        return x

    def sample_density(self, z_dense, c, **kwargs):
        # c = self.unflatten_ce(c).unsqueeze(1)
        c = self.embed_condition(c)
        if self.hparams.cfg:
            for net in self.density_model:
                z_dense = net.sample_with_guidance(
                    z_dense, c, self.get_null_condition(c), **kwargs
                )
        else:
            z_dense = self.decode_density(z_dense, c, **kwargs)
        return z_dense

    def _encoder_jac(self, x, c, **kwargs):
        return compute_jacobian(
            x,
            self.encode_density,
            c,
            chunk_size=self.hparams.exact_chunk_size,
            **kwargs,
        )

    def _decoder_jac(self, z, c, **kwargs):
        return compute_jacobian(
            z,
            self.decode_density,
            c,
            chunk_size=self.hparams.exact_chunk_size,
            **kwargs,
        )

    def _encoder_volume_change(self, x, c, **kwargs) -> VolumeChangeResult:
        z, jac_enc = self._encoder_jac(x, c, **kwargs)
        jac_enc = jac_enc.reshape(x.shape[0], prod(z.shape[1:]), prod(x.shape[1:]))
        jtj = torch.einsum("bik,bjk->bij", jac_enc, jac_enc)
        log_det = jtj.slogdet()[1] / 2
        return VolumeChangeResult(z, log_det, {})

    def _decoder_volume_change(self, z, c, **kwargs) -> VolumeChangeResult:
        # Forward gradient is faster because latent dimension is smaller than data dimension
        x1, jac_dec = self._decoder_jac(z, c, grad_type="forward", **kwargs)
        jac_dec = jac_dec.reshape(z.shape[0], prod(x1.shape[1:]), prod(z.shape[1:]))
        jjt = torch.einsum("bki,bkj->bij", jac_dec, jac_dec)
        log_det = jjt.slogdet()[1] / 2
        return VolumeChangeResult(x1, log_det, {})

    def _latent_log_prob(self, z, c):
        try:
            return self.get_latent(z.device).log_prob(z, c)
        except TypeError:
            return self.get_latent(z.device).log_prob(z)

    def sample(self, sample_shape, condition=None, **kwargs):
        """
        Sample via the density_model and lossless_ae decoder.
        """
        # sample first via the density_model, if included in the latent distribution
        try:
            z_dense = self.get_latent(self.device).sample(sample_shape, condition)
        except TypeError:
            z_dense = self.get_latent(self.device).sample(sample_shape)
        z_dense = z_dense.reshape(
            prod(sample_shape), *z_dense.shape[len(sample_shape) :]
        )
        if condition is not None:
            c = condition
        else:
            c = torch.empty(z_dense.shape[0], 0).to(z_dense.device)
        z = self.sample_density(z_dense, c, **kwargs)
        x = self.decode_lossless(z, c)
        return x.reshape(sample_shape + x.shape[1:])

    def surrogate_log_prob(self, x, c, **kwargs) -> LogProbResult:
        # Then compute JtJ
        config = deepcopy(self.hparams.log_det_estimator)
        estimator_name = config.pop("name")
        assert estimator_name == "surrogate"

        out = volume_change_surrogate(
            x,
            lambda _x: self.encode_density(_x, c),
            lambda z: self.decode_density(z, c),
            **kwargs,
        )

        volume_change = out.surrogate

        latent_prob = self._latent_log_prob(out.z, c)
        return LogProbResult(
            out.z, out.x1, latent_prob + volume_change, out.regularizations
        )

    def get_null_condition(self, cond_batch):
        """
        Returns a null condition for the given batch.
        """
        if self.hparams.cfg.get("null_condition", "learned") == "learned":
            if not hasattr(self, "null_condition"):
                self.null_condition = torch.nn.Parameter(
                    torch.randn(1, *cond_batch.shape[1:]), requires_grad=True
                )
            return self.null_condition.expand(*cond_batch.shape).to(cond_batch.device)
        elif self.hparams.cfg["null_condition"] == "zero":
            return torch.zeros_like(cond_batch)
        else:
            raise ValueError(
                "Unknown null condition type: "
                + self.hparams.cfg["null_condition"]
                + ". Use 'learned' or 'zero'."
            )

    def compute_metrics(self, batch, batch_idx) -> dict:
        """
        Computes the metrics for the given batch.

        Rationale:
        - In training, we only compute the terms that are actually used in the loss function.
        - During validation, all possible terms and metrics are computed.

        :param batch:
        :param batch_idx:
        :return:
        """
        conditioned = self.apply_conditions(batch)
        variables = conditioned._asdict()
        loss_weights = variables["loss_weights"]
        deq_vol_change = conditioned.dequantization_jac

        def compute_warmup(epochs):
            if isinstance(epochs, int):
                epochs = epochs, epochs + 1
            warm_up = map(lambda x: x * self.hparams.max_epochs // 100, epochs)
            start, warm_up_end = warm_up
            if start == 0:
                warmup = 1
            else:
                warmup = soft_heaviside(
                    self.current_epoch
                    + batch_idx
                    / len(
                        self.trainer.train_dataloader
                        if self.training
                        else self.trainer.val_dataloaders
                    ),
                    start,
                    warm_up_end,
                )
            return warmup
 
        metrics, loss_weights = self.LossComputer(
            variables, 
            compute_warmup,
            self.training,
            self.current_epoch,
            batch_idx
        )
        # Store loss weights
        if self.training:
            for key, weight in loss_weights.items():
                if not torch.is_tensor(weight):
                    weight = torch.tensor(weight)
                self.log(f"weights/{key}", weight.float().mean())

        # Check finite loss
        if not torch.isfinite(metrics["loss"]) and self.training:
            self.trainer.save_checkpoint("erroneous.ckpt")
            print(f"Encountered nan loss from: {metrics}!")
            raise SkipBatch

        return metrics

    def apply_conditions(self, batch) -> ConditionedBatch:
        x0 = batch[0]
        base_cond_shape = (x0.shape[0], 1)
        device = x0.device
        dtype = x0.dtype

        conds = []
        x_sm = x0
        if self.hparams.add_noise_for_sm:
            if self.hparams["data_set"].get("data") == "highdose":
                x_sm = batch[1]
            else:
                raise (
                    ValueError(
                        "Adding noise from condition only works for highdose images as data"
                    )
                )

        # Dataset condition
        if self.is_conditional() and len(batch) < 2:
            if self.hparams.compute_c_on_fly:
                conds.append(self.subject_model.encode(x_sm).detach())
            else:
                raise ValueError(
                    "You must pass a batch including conditions for each dataset condition"
                )
        if len(batch) > 1:
            if self.hparams.compute_c_on_fly:
                dataset_cond = self.subject_model.encode(x_sm).detach()
            else:
                dataset_cond = batch[1]
            conds.append(dataset_cond)
        if len(batch) > 2:
            jac_sm = batch[2]
        else:
            jac_sm = None

        # SoftFlow
        noise_conds, x, dequantization_jac = self.dequantize(batch)
        conds.extend(noise_conds)

        # Loss weight aware
        loss_weights = defaultdict(float, self.hparams.loss_weights)
        for loss_key, loss_weight in self.hparams.loss_weights.items():
            if isinstance(loss_weight, list):
                min_weight, max_weight = loss_weight
                if not self.training:
                    # Per default, select the first value in the list
                    max_weight = min_weight
                weight_scale = rand_log_uniform(
                    min_weight,
                    max_weight,
                    shape=base_cond_shape,
                    device=device,
                    dtype=dtype,
                )
                loss_weights[loss_key] = (10**weight_scale).squeeze(1)
                conds.append(weight_scale)

        if len(conds) == 0:
            c = torch.empty((x.shape[0], 0), device=x.device, dtype=x.dtype)
        elif len(conds) == 1:
            # This is a hack to pass through the info dict from QM9
            (c,) = conds
        else:
            c = torch.cat(conds, -1)
        return ConditionedBatch(x0, x, loss_weights, c, dequantization_jac, jac_sm)

    def configure_optimizers(self):
        params = []
        if self.hparams.train_lossless_ae:
            params.extend(list(self.lossless_ae.parameters()))
            if self.vae:
                params.append(self.lamb)
            if self.density_model_type:
                print("WARNING: lossless_ae gets trained jointly with a density_model")
        else:
            print("WARNING: lossless_ae gets not trained")
        if self.density_model_type:
            params.extend(list(self.density_model.parameters()))
        else:
            print("WARNING: density model gets not trained")
        if self.condition_embedder is not None:
            params.extend(list(self.condition_embedder.parameters()))
        else:
            print("WARNING: no condition embedder for optimizer")

        kwargs = dict()

        match self.hparams.optimizer:
            case str() as name:
                optimizer = lightning_trainable.utils.get_optimizer(name)(
                    params, **kwargs
                )
            case dict() as kwargs:
                name = kwargs.pop("name")
                optimizer = lightning_trainable.utils.get_optimizer(name)(
                    params, **kwargs
                )
                self.hparams.optimizer["name"] = name
            case type(torch.optim.Optimizer) as Optimizer:
                optimizer = Optimizer(params, **kwargs)
            case torch.optim.Optimizer() as optimizer:
                pass
            case None:
                return None
            case other:
                raise NotImplementedError(f"Unrecognized Optimizer: {other}")

        lr_scheduler = lightning_trainable.trainable.lr_schedulers.configure(
            self, optimizer
        )

        if lr_scheduler is None:
            return optimizer

        return dict(
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )

    def validation_step(self, batch, batch_idx):
        if self.current_epoch % self.hparams.val_every_n_epoch == 0:
            metrics = self.compute_metrics(batch, batch_idx)
            for key, value in metrics.items():
                self.log(
                    f"validation/{key}",
                    value,
                    prog_bar=key == self.hparams.loss,
                    #sync_dist=self.hparams.strategy.startswith("ddp"),
                )
