from torch.nn import Module
import torch
from math import prod

from fff.base import ModelHParams, build_model
from fff.model import Identity
import copy


from fff.utils.checkpoint import default_map_location  # noqa: E402


class LosslessAEHParams(ModelHParams):
    model_spec: list = []
    cond_dim: int | list = 0
    path: str | None = None
    vae: bool = True
    data_dim: int
    train: bool = False
    cond_embedding_network: list = []
    cond_embedding_shape: int | list | None = None
    use_condition_decoder: bool = False


class LosslessAE(Module):

    hparams: LosslessAEHParams

    def __init__(self, hparams: LosslessAEHParams | dict):
        if hparams.get("path"):
            checkpoint = torch.load(hparams["path"], weights_only=False,
                                    map_location=default_map_location())
            print("Overwriting lossless ae model spec with pretrained model")
            stored = checkpoint["hyper_parameters"]["lossless_ae"]
            if isinstance(stored, list):
                # The released checkpoints predate the dict form: `lossless_ae`
                # was the bare model spec. FiberModelHParams._migrate_hparams
                # normalises this, but nothing routes a LosslessAE through it.
                stored = {"model_spec": stored}
            hparams["model_spec"] = stored["model_spec"]
            hparams["cond_embedding_network"] = stored.get("cond_embedding_network", [])
            hparams["cond_embedding_shape"] = stored.get("cond_embedding_shape", None)
            hparams["use_condition_decoder"] = stored.get("use_condition_decoder", False)
            hparams["vae"] = stored.get("vae", True)

        if not isinstance(hparams, LosslessAEHParams):
            hparams = LosslessAEHParams(**hparams)
        super().__init__()

        self.hparams = hparams
        self.data_dim = self.hparams.data_dim

        if self.hparams.cond_embedding_shape is None:
            assert (
                self.hparams.cond_embedding_network == []
            ), "cond_embedding_shape must be specified if cond_embedding_network is specified"
            self.hparams.cond_embedding_shape = [self.hparams.cond_dim]
        else:
            if isinstance(self.hparams.cond_embedding_shape, int):
                self.hparams.cond_embedding_shape = [self.hparams.cond_embedding_shape]
            assert not (
                self.hparams.cond_embedding_network == []
            ), "cond_embedding_network must be specified if cond_embedding_shape is specified"

        model_spec = copy.deepcopy(self.hparams.model_spec)
        if self.hparams.vae:
            lat_dim = self.hparams.model_spec[-1]["latent_dim"]
            model_spec[-1]["latent_dim"] = lat_dim * 2

        self.models = build_model(
            model_spec,
            self.data_dim,
            self.hparams.cond_embedding_shape[0],
        )

        if self.hparams.cond_embedding_network:
            assert not (
                self.hparams.cond_dim == 0
            ), "ae_conditional has to be set to True if a cond_embedding_network is built"
            # Build a network to embed the conditioning
            if self.hparams.use_condition_decoder:
                self.condition_embedder = build_model(
                    self.hparams.cond_embedding_network,
                    prod(self.hparams.cond_embedding_shape),
                    0,
                )
                for model in self.condition_embedder:
                    del model.model.encoder
            else:
                self.condition_embedder = build_model(
                    self.hparams.cond_embedding_network,
                    self.hparams.cond_dim,
                    0,
                )
                for model in self.condition_embedder:
                    del model.model.decoder
            if not self.hparams.train:
                self.condition_embedder.eval()
        else:
            self.condition_embedder = Identity(self.hparams)

        if self.hparams.path:
            try:
                print("Loading lossless_ae checkpoint from: ", hparams["path"])
                lossless_ae_weights = {
                    k[len("lossless_ae.") :]: v
                    for k, v in checkpoint["state_dict"].items()
                    if k.startswith("lossless_ae.")
                }
                self.load_state_dict(lossless_ae_weights)
            except (RuntimeError, KeyError):
                # older checkpoints stored the same weights under "models."
                print("Loading lossless_ae checkpoint from: ", hparams["path"])
                lossless_ae_weights = {
                    k[len("models.") :]: v
                    for k, v in checkpoint["state_dict"].items()
                    if k.startswith("models.")
                }
                self.models.load_state_dict(lossless_ae_weights)

        if not self.hparams.train:
            self.models.eval()

    @property
    def latent_dim(self):
        latent_dim = self.models[-1].hparams.latent_dim
        return latent_dim // 2 if self.hparams.vae else latent_dim

    def embed_condition(self, c):
        if self.hparams.cond_embedding_network:
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
        return c.reshape(c.shape[0], *self.hparams.cond_embedding_shape)

    def decode(self, z, c, **kwargs):
        if self.hparams.cond_dim == 0:
            c = torch.empty((z.shape[0], 0), device=z.device, dtype=z.dtype)
        c = self.embed_condition(c)
        if self.hparams.vae:
            z = torch.nn.functional.pad(z, (0, z.shape[1]))
        for model in self.models[::-1]:
            z = model.decode(z, c, **kwargs)
        return z

    def encode(self, x, c, return_only_x=False, deterministic=False, **kwargs):
        if self.hparams.cond_dim == 0:
            c = torch.empty((x.shape[0], 0), device=x.device, dtype=x.dtype)
        c = self.embed_condition(c)
        for model in self.models:
            x = model.encode(x, c, **kwargs)
            other = []
            if isinstance(x, tuple):
                x, other = x[0], x[1:]
        mu, logvar = None, None
        if self.hparams.vae:
            # VAE latent sampling
            mu = x[:, : x.shape[1] // 2].reshape(-1, x.shape[1] // 2)
            logvar = x[:, x.shape[1] // 2 :].reshape(-1, x.shape[1] // 2)
            if deterministic:
                x = mu
            else:
                epsilon = torch.randn_like(logvar).to(mu.device)
                x = mu + torch.exp(0.5 * logvar) * epsilon
        if return_only_x:
            return x
        return x, mu, logvar, *other

