from collections import OrderedDict

import torch
from torch import Tensor, nn

from .auto_encoder import SkipConnection
from fff.base import ModelHParams
from .utils import wrap_batch_norm1d, make_dense, CrossAttention
from fff.utils.diffusion import make_betas


class DiffHParams(ModelHParams):
    layers_spec: list
    activation: str = "gelu"
    id_init: bool = False
    batch_norm: str | bool = False
    dropout: float | None = None
    num_heads: int = 4
    hidden_dim: int = 32
    time_dim: int = 8
    betas_max: float = 0.04
    beta_schedule: str = "linear"
    num_timesteps: int = 1000
    eta: float = 0.1

    def __init__(self, **hparams):
        if "latent_spec" in hparams:
            assert len(hparams["latent_spec"]) == 0
            del hparams["latent_spec"]
        super().__init__(**hparams)

def res_layer(data_dim, widths, activation, id_init: bool,
                 batch_norm: str | bool, dropout: float = None):
    return SkipConnection(
        make_dense([data_dim, *widths, data_dim], activation,
                   batch_norm=batch_norm, dropout=dropout),
        id_init=id_init
    )


class DiffusionModel(nn.Module):
    hparams: DiffHParams

    def __init__(self, hparams: dict | DiffHParams):
        if not isinstance(hparams, DiffHParams):
            hparams = DiffHParams(**hparams)

        super().__init__()
        self.hparams = hparams
        self.build_model()

        # Precompute constants for sampling.
        #
        # These are registered as buffers, not set as plain attributes. A plain
        # tensor attribute does not follow the module across .to(device), so on
        # a GPU the schedule stayed on the CPU while `t` did not, and the first
        # validation batch died in `alpha_cumprod[t]` with "indices should be
        # either on cpu or on the same device as the indexed tensor". On CPU
        # everything is on one device and the bug is invisible, which is why it
        # survived a CPU-only check.
        #
        # persistent=False keeps them out of state_dict: they are derived from
        # hparams, and adding keys would break loading every checkpoint that
        # predates this.
        betas = make_betas(
            self.hparams.num_timesteps,
            self.hparams.betas_max,
            self.hparams.beta_schedule,
        )
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas, persistent=False)
        self.register_buffer(
            "sample_steps",
            torch.linspace(0, 1, self.hparams.num_timesteps).flip(0),
            persistent=False)
        self.register_buffer("alphas", alphas, persistent=False)
        self.register_buffer("alpha_cumprod", alpha_cumprod, persistent=False)
        self.register_buffer(
            "alpha_cumprod_prev",
            torch.cat([torch.ones(1, dtype=alpha_cumprod.dtype),
                       alpha_cumprod[:-1]]),
            persistent=False)
        self.register_buffer("sqrt_1malpha_cumprod",
                             torch.sqrt(1 - alpha_cumprod), persistent=False)
        self.eta = self.hparams.eta

    def diffuse(self, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        noise = x0
        alpha_t = self.alpha_cumprod[t].unsqueeze(1)
        alpha_t = alpha_t.repeat([1, x1.shape[1]])
        noisy_x = alpha_t.sqrt() * x1 + (1 - alpha_t).sqrt() * noise
        # noisy_x = alpha_t.sqrt() * x + noise
        return noisy_x

    def diffuse_and_reverse(self, t: Tensor, x0: Tensor, x1: Tensor, c: Tensor) -> Tensor:
        x_noisy = self.diffuse(t, x0, x1)
        return self.forward(t, x_noisy, c)

    def forward(self, t: Tensor, x: Tensor, c: Tensor, guidance_scale=1.0, conditional=True) -> Tensor:
        # Embed the time step
        time_embedding = self.time_embedding(t)
        #condition = torch.zeros_like(x_noisy)
        
        # Initial input
        x = torch.cat((x, time_embedding), dim=-1)
        x = self.fc_in(x)
        
        if conditional:
            # Conditional forward pass
            conditional_x = x
            for layer in self.layers:
                if isinstance(layer, CrossAttention):
                    conditional_x = conditional_x + layer(conditional_x, c)
                else:
                    conditional_x = layer(conditional_x)
            conditional_output = self.fc_out(conditional_x)
            
            # Unconditional forward pass (condition replaced with zeros)
            unconditional_x = x
            for layer in self.layers:
                if isinstance(layer, CrossAttention):
                    unconditional_x = unconditional_x + layer(unconditional_x, torch.zeros_like(c))
                else:
                    unconditional_x = layer(unconditional_x)
            unconditional_output = self.fc_out(unconditional_x)
            
            # Classifier-free guidance
            output = unconditional_output + guidance_scale * (conditional_output - unconditional_output)
            #output = conditional_output
        else:
            # Unconditional forward pass
            for layer in self.layers:
                if isinstance(layer, CrossAttention):
                    x = x + layer(x, torch.zeros_like(c))
                else:
                    x = layer(x)
            output = self.fc_out(x)
        #print(torch.sum(x-x_noisy))
        return output

    def build_model(self):
        input_dim = self.hparams.data_dim
        hidden_dim = self.hparams.hidden_dim
        condition_dim = self.hparams.cond_dim
        #condition_dim = 2
        time_dim = self.hparams.time_dim
        num_heads = self.hparams.num_heads
        activation = self.hparams.activation
        
        # Time embedding layer
        self.time_embedding = nn.Embedding(self.hparams.num_timesteps, time_dim)  
        
        self.fc_in = nn.Linear(input_dim + time_dim, hidden_dim)
        
        # Hidden layers with cross-attention
        self.layers = nn.ModuleList()
        
        for widths in self.hparams.layers_spec:
            self.layers.append(res_layer(hidden_dim, widths, activation,
                id_init=self.hparams.id_init,
                batch_norm=self.hparams.batch_norm, dropout=self.hparams.dropout)
            )
            self.layers.append(CrossAttention(hidden_dim, condition_dim, num_heads))
        
        # Output layer
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x, *args) -> Tensor:
        return x

    def decode(self, x, condition, guidance_scale=1.0):
        device = x.device
        num_steps = self.hparams.num_timesteps
        shape = x.shape[0]

        for i in reversed(range(num_steps)):
            t = torch.full([shape], i, device=device, dtype=torch.long)
            alpha_t = self.alpha_cumprod[i]
            alpha_t_prev = self.alpha_cumprod_prev[i]
            beta_t = self.betas[i]

            # Predict noise
            eps_pred = self.forward(t, x, condition, guidance_scale)

            # Compute the mean for the reverse process
            pred_x0 = (
                x - self.sqrt_1malpha_cumprod[i] * eps_pred
            ) / alpha_t.sqrt()

            noise = torch.randn_like(x) if self.eta > 0 else torch.zeros_like(x)
            sigma_t = self.eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * beta_t)

            dir_xt = (1. - alpha_t_prev - sigma_t**2).sqrt() * eps_pred

            x = alpha_t_prev.sqrt() * pred_x0 + dir_xt + sigma_t * noise

        return x
