import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from math import prod
import math
import sys

from tqdm.auto import tqdm, trange
from torch.utils.data import DataLoader
from diffusers import UNet2DModel


device = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class DiffusionScheduleConfig:
    beta_schedule: str = 'linear'
    beta_start: float = 1e-4
    beta_end: float = 0.02
    num_diffusion_timesteps: int = 1000
    given_betas: torch.Tensor = None  # Optional, if provided, will override the schedule

class DiffusionSchedule:
    def __init__(self, hparams, device=None):
        # The betas are built with numpy and so land on the CPU. Do not force
        # them onto CUDA -- that makes the library unusable without one. The
        # default still picks CUDA wherever it is available.
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        # Instantiate the diffusion process
        if hparams.given_betas is None:
            if hparams.beta_schedule == "quad":
                betas = (
                    np.linspace(
                        hparams.beta_start**0.5,
                        hparams.beta_end**0.5,
                        hparams.num_diffusion_timesteps,
                        dtype=np.float64,
                    )
                    ** 2
                )
            elif hparams.beta_schedule == "linear":
                betas = np.linspace(
                    hparams.beta_start, hparams.beta_end, hparams.num_diffusion_timesteps, dtype=np.float64
                )
            elif hparams.beta_schedule == "const":
                betas = hparams.beta_end * np.ones(hparams.num_diffusion_timesteps, dtype=np.float64)
            elif hparams.beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
                betas = 1.0 / np.linspace(
                    hparams.num_diffusion_timesteps, 1, hparams.num_diffusion_timesteps, dtype=np.float64
                )
            elif hparams.beta_schedule == "sigmoid":
                betas = np.linspace(-6, 6, hparams.num_diffusion_timesteps)
                betas = 1 / (np.exp(-betas) + 1) * (hparams.beta_end - hparams.beta_start) + hparams.beta_start
            else:
                raise NotImplementedError(hparams.beta_schedule)
            assert betas.shape == (hparams.num_diffusion_timesteps,)
            betas = torch.from_numpy(betas)
        else:
            betas = hparams.given_betas
        self.betas = torch.cat([torch.zeros(1, device=betas.device), betas], dim=0).to(device).float()
        self.alphas = (1 - self.betas).cumprod(dim=0).to(device).float()
        if not (self.alphas > 0).all():
            # The "jsd" schedule ends at beta = 1, so alpha_bar reaches exactly
            # zero and predict_x_from_eps divides by sqrt(0): the whole
            # trajectory comes back inf/NaN, and NDTM reports a loss rather than
            # an error. Refuse it here instead.
            raise ValueError(
                f"the {hparams.beta_schedule!r} schedule drives alpha_bar to "
                f"zero (beta reaches {self.betas.max().item():.4g}), which makes "
                f"the Tweedie estimate a division by zero; use a schedule whose "
                f"betas stay below 1")
        self.hparams = hparams

    def alpha(self, t):
        return self.alphas[t+1]
    
    def beta(self, t):
        return self.betas[t+1]

    def predict_x_from_eps(self, xt, et, t):
        alpha_t = self.alpha(t).view(-1, *([1] * (xt.ndim - 1)))
        return (xt - et * (1 - alpha_t).sqrt()) / alpha_t.sqrt()

    @torch.no_grad()
    def noise_image(self, x0, t):
        """
        x0: clean image, shape (B, C, H, W) or flat latent (B, D)
        t: integer or tensor of shape (B,)
        """
        alpha_t = self.alpha(t).view(-1, *([1] * (x0.ndim - 1)))
    
        eps = torch.randn_like(x0)
    
        xt = (
            alpha_t.sqrt() * x0
            + (1.0 - alpha_t).sqrt() * eps
        )
    
        return xt

class DiffusionModel:
    def __init__(self, model: nn.Module, diffusion_schedule: object, class_cond_diffusion_model=False):
        self.model = model
        self.diffusion_schedule = diffusion_schedule
        self.class_cond_diffusion_model = class_cond_diffusion_model

    def __call__(self, xt, y, t, predict_variance=False):
        y = y if self.class_cond_diffusion_model else None
        out = self.model(xt, t, y)
        if out.ndim < 4:
            # Flat (non-image) latents: the model output is the noise estimate itself.
            if predict_variance:
                # no variance channels; empty logvar like a 3-channel image model
                return out, out[:, :0]
            return out
        # The noise estimate has one channel per input channel; anything beyond
        # that is the learned variance. Read the count off the state rather than
        # hard-coding 3: that is right for the RGB models but silently drops a
        # channel of a 4-channel latent, after which the estimate broadcasts
        # against the state instead of failing.
        channels = xt.shape[1]
        et = out[:, :channels]
        if not predict_variance:
            return et
        else:
            logvar = out[:, channels:]
            return et, logvar

class Combine_fn(ABC):
    def __init__(self, gamma_t=None):
        self.gamma_t = gamma_t

    @abstractmethod
    def forward(self, xt, ut, t=None, **kwargs):
        pass

    def __call__(self, xt, ut, t=None, **kwargs):
        return self.forward(xt, ut, t=t, **kwargs)


class Additive(Combine_fn):
    def forward(self, xt, ut, t=None, **kwargs):
        gamma_t = self.gamma_t(t) if callable(self.gamma_t) else self.gamma_t
        return xt + (gamma_t * ut if gamma_t is not None else ut)

@dataclass
class NDTMConfig:
    eta: float = 1.0
    N: int = 2  # Number of optimization steps
    gamma_t: float = 4.0  # u_t weight
    u_lr: float = 0.01  # learning rate for u_t
    combine_fn: str = "additive"  # Function to combine scores
    w_score_scheme: str = "ddim"  # Weighting scheme for score
    w_control_scheme: str = "ddim"  # Weighting scheme for score
    u_lr_scheduler: str = "linear"  # Learning rate scheduler for u_t
    init_control: str = "zero"  # Initialization scheme for u_t
    init_xT: str = "random" # Initialization scheme for x_T
    w_terminal: float = 50.0
    ancestral_sampling: bool = False  # If True, use ancestral sampling
    clip_images: bool = True  # If True, clip images
    clip_range: list = field(default_factory=lambda: [-1, 1])  # Range to clip images
    variance_type: str = "small"
    compute_target_per_timestep: bool = False
    fiber_loss: str = "l2"

class NDTM:
    def __init__(self, generative_model, subject_model, hparams):
        self.generative_model = generative_model
        self.diffusion = generative_model.diffusion_schedule
        self.subject_model = subject_model
        self.hparams = hparams
        self.F = self._get_combine_fn()

    def _get_combine_fn(self):
        if self.hparams.combine_fn == "additive":
            return Additive(gamma_t=self.hparams.gamma_t)
        raise ValueError(
            f"unknown combine_fn {self.hparams.combine_fn!r}; the only one "
            f"implemented is 'additive'")

    def _get_score_weight(self, scheme, t, s, **kwargs):
        alpha_t = self.diffusion.alpha(t)
        alpha_s = self.diffusion.alpha(s)
        beta_t = self.diffusion.beta(t)
        alpha_t_im = 1 - beta_t

        if scheme == "zero":
            return torch.tensor([0.0], device=alpha_s.device)
        elif scheme == "ones":
            return torch.tensor([1.0], device=alpha_s.device) * 2.e-5
        elif scheme == "ddpm":
            return (beta_t**2) / (alpha_t_im * (1 - alpha_t))
        elif scheme == "ddim":
            c1 = (
                (1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)
            ).sqrt() * self.hparams.eta
            c2 = ((1 - alpha_s) - c1**2).sqrt()
            c2_ = ((alpha_s / alpha_t) * (1 - alpha_t)).sqrt()
            return (c2 - c2_) ** 2
        # bool is an int subclass and is never a weight anyone means
        elif isinstance(scheme, (int, float)) and not isinstance(scheme, bool):
            return float(scheme)
        else:
            raise ValueError(
                f"unknown w_score_scheme {scheme!r}; give a number, or one of "
                f"'zero', 'ones', 'ddpm', 'ddim'")

    def _get_control_weight(self, scheme, t, s):
        alpha_t = self.diffusion.alpha(t)
        alpha_s = self.diffusion.alpha(s)
        beta_t = self.diffusion.beta(t)
        alpha_t_im = 1 - beta_t

        if scheme == "zero":
            return torch.tensor([0.0], device=alpha_s.device)
        elif scheme == "ones":
            return torch.tensor([1.0], device=alpha_s.device) * 1.e-4
        elif scheme == "ddpm":
            return 1 / alpha_t_im
        elif scheme == "ddim":
            return alpha_t / alpha_s
        elif isinstance(scheme, (int, float)) and not isinstance(scheme, bool):
            return float(scheme)
        else:
            raise ValueError(
                f"unknown w_control_scheme {scheme!r}; give a number, or one of "
                f"'zero', 'ones', 'ddpm', 'ddim'")

    def get_learning_rate(self, base_lr, current_step, total_steps):
        assert self.hparams.u_lr_scheduler in ["linear", "const", "cos"], \
            f"Unknown learning rate scheduler: {self.hparams.u_lr_scheduler}"

        if self.hparams.u_lr_scheduler == "linear":
            return base_lr * (1.0 - current_step / total_steps)
        elif self.hparams.u_lr_scheduler == "cos":
            # Verbatim from the ndtm.py the paper's sampling notebook embedded,
            # which carries a scheduler the committed one did not -- so it is kept
            # here, or no configuration using it could be expressed at all.
            #
            # It is the REVERSE of "linear", not a smoothing of it. current_step is
            # the loop index i, which starts at the noisiest timestep, so "linear"
            # spends the learning rate at high noise and decays to zero by t -> 0,
            # while this starts at zero and ramps up to base_lr at the end --
            # concentrating the control optimisation exactly where gamma ramps 2 -> 10
            # and where Appendix C says the texture match is won.
            return base_lr + 0.5 * (0.0 - base_lr) * (
                1 + np.cos(np.pi * current_step / total_steps))
        else:  # const
            return base_lr

    def sample(self, x, y, ts, **kwargs):
        x_orig = x.clone()
        x = self.initialize(x, y, ts, **kwargs)
        y_0 = kwargs["y_0"]
        bs = x.size(0)
        xt = x
        ss = [-1] + list(ts[:-1])
        xt_s = [xt.cpu()]
        x0_s = []
        uts = []
        if self.hparams.variance_type not in ("small", "large", "learned_range"):
            raise ValueError(
                f"unknown variance_type {self.hparams.variance_type!r}; expected "
                f"'small', 'large' or 'learned_range'")

        # The rescaled schedule below is only read by the ancestral update, and
        # the linear-schedule requirement is only the ancestral update's. Both
        # stay inside the branch: outside it, DDIM sampling from a model trained
        # on any other schedule is refused with a message about ancestral
        # sampling.
        given_betas = alphas_cumprod = alphas_cumprod_prev = None
        if self.hparams.ancestral_sampling:
            if self.diffusion.hparams.beta_schedule != "linear":
                raise ValueError(
                    "ancestral sampling rescales a linear beta schedule to the "
                    f"number of sampling steps, and this model was trained with "
                    f"a {self.diffusion.hparams.beta_schedule!r} schedule; sample "
                    f"it with ancestral_sampling=False")
            # This rebuilds a LINEAR beta schedule over `len(ts)` steps with the
            # betas scaled by 1000/len(ts), and the ancestral update then runs on
            # its cumulative products -- while the network is still conditioned on
            # the original timestep t and predicts eps for the ORIGINAL
            # alpha_bar[t]. The two disagree: they match to 1% around the middle
            # of the trajectory but diverge towards the noisy end, where at
            # num_steps=200 alpha_bar_rescaled/alpha_bar[t] reaches 0.69. The
            # Tweedie estimate divides by their square roots, so x0_pred is
            # mis-scaled by up to 20% for the first steps, and by more than 5%
            # for 81 of the 200. The DDIM branch has no such mismatch: it reads
            # the model's own alphas_cumprod at the strided t.
            #
            # So sampling with ancestral_sampling=False, eta=1.0 does NOT match
            # this branch, even though the two updates are otherwise algebraically
            # the same at eta=1 (c1 interpolates geometrically between
            # sqrt(beta~) and sqrt(beta), exactly what learned_range's
            # log-variance does, and c2 puts the mean at the posterior mean).
            #
            # This branch is what the paper's numbers were produced through, so
            # it is kept as is: "correcting" the mismatch reproduces nothing.
            scale = self.diffusion.hparams.num_diffusion_timesteps/len(ts)
            beta_start = scale * self.diffusion.hparams.beta_start
            beta_end = scale * self.diffusion.hparams.beta_end
            given_betas = torch.linspace(
                beta_start, beta_end, len(ts), dtype=torch.float64, device=x.device
            )
            # Constant across the loop, so build them once rather than per step.
            alphas_cumprod = (1 - given_betas).cumprod(dim=0).float()
            alphas_cumprod_prev = torch.cat(
                (torch.ones(1, device=x.device), alphas_cumprod[:-1]), dim=0)

        # Note the terminal cost minimised here is the l2 *norm*, while the fiber
        # loss reported in Tables 5-7 is the *squared* norm summed over dimensions
        # (see experiments/imagenet/evaluate.py). A terminal loss of ~16 printed
        # below therefore corresponds to a reported fiber loss of ~270.
        if self.hparams.fiber_loss == "l2":
            fiber_loss_fct = lambda x, y: torch.norm(x-y, p=2, dim=-1)
        elif self.hparams.fiber_loss == "l1":
            fiber_loss_fct = lambda x, y: torch.norm(x-y, p=1, dim=-1)
        elif self.hparams.fiber_loss == "cross_entropy":
            fiber_loss_fct = lambda x, y: F.cross_entropy(x, y.softmax(dim=-1), reduction="none")
        else:
            raise ValueError(
                f"unknown fiber_loss {self.hparams.fiber_loss!r}; expected 'l2', "
                f"'l1' or 'cross_entropy'")


        u_t = torch.zeros_like(xt)
        # Redirected to a file, tqdm cannot rewrite a line, so it appends a new
        # one per update -- a 200-step sampling run wrote megabytes of bar into
        # the job log and buried every real message. Throttle it off a terminal.
        pbar = tqdm(enumerate(zip(reversed(ts), reversed(ss))), total=len(ts),
                    leave=False,
                    mininterval=0.1 if sys.stderr.isatty() else 60.0)
        for i, (ti, si) in pbar:
            t = torch.ones(bs).to(x.device).long() * ti
            s = torch.ones(bs).to(x.device).long() * si
            alpha_t = self.diffusion.alpha(t).view(-1, *([1] * (x.ndim - 1)))
            alpha_s = self.diffusion.alpha(s).view(-1, *([1] * (x.ndim - 1)))
            if self.hparams.compute_target_per_timestep:
                with torch.no_grad():
                    noised_x_gt = self.diffusion.noise_image(y_0, t)
                    et_gt = self.generative_model(noised_x_gt, y, t).detach()
                    approx_x_gt = self.diffusion.predict_x_from_eps(noised_x_gt, et_gt, t)
                    approx_x_gt = torch.clamp(approx_x_gt, self.hparams.clip_range[0], self.hparams.clip_range[1])
                    y_t = self.subject_model(approx_x_gt)
            else:
                # With a static target, `y_0` is already the target representation
                # phi(x): callers that pass an image must embed it themselves.
                y_t = y_0
            if self.hparams.variance_type == "small":
                c1 = (
                    (1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)
                ).sqrt() * self.hparams.eta
                c2 = ((1 - alpha_s) - c1**2).sqrt()
            elif self.hparams.variance_type == "large":
                c1 = (1 - alpha_t / alpha_s).sqrt() * self.hparams.eta
                c2 = ((1 - alpha_s) - ((1 - alpha_s) / (1 - alpha_t))*c1**2).sqrt()
            elif self.hparams.variance_type == "learned_range":
                c1_small = (
                    (1 - alpha_t / alpha_s) * (1 - alpha_s) / (1 - alpha_t)
                ).sqrt() * self.hparams.eta
                c1_large = (1 - alpha_t / alpha_s).sqrt() * self.hparams.eta

                c2 = ((1 - alpha_s) - c1_small**2).sqrt()

            # Initialize control and the optimizer
            u_t = self.initialize_ut(u_t, i)
            ut_clone = u_t.clone().detach()
            ut_clone.requires_grad = True
            current_lr = self.get_learning_rate(self.hparams.u_lr, i, len(ts))
            optimizer = torch.optim.Adam([ut_clone], lr=current_lr)

            # Loss weightings
            w_terminal = self.hparams.w_terminal
            w_score = self._get_score_weight(self.hparams.w_score_scheme, t, s, **kwargs)
            w_control = self._get_control_weight(self.hparams.w_control_scheme, t, s)
            time_rev_ind = len(ts) - i - 1
            # Only the ancestral update reads these; they are built once, before
            # the loop, and are None for DDIM sampling.
            beta_eff = given_betas[time_rev_ind] if given_betas is not None else None


            ####################################################
            ############## Control Optimization ################
            ####################################################
            et = self.generative_model(xt, y, t).detach()
            for _ in range(self.hparams.N):
                if callable(self.hparams.gamma_t):
                    gamma_t = self.hparams.gamma_t(t)
                else:
                    gamma_t = self.hparams.gamma_t
                if gamma_t == 0:
                    break
                # Guided state vector
                cxt = self.F(xt, ut_clone, t=t, **kwargs)

                # Unguided and guided noise estimates
                et_control = self.generative_model(cxt, y, t)

                # Tweedie's estimate from the guided state vector
                if self.hparams.ancestral_sampling:
                    x0_pred = cxt/alphas_cumprod[time_rev_ind].sqrt() - (1/alphas_cumprod[time_rev_ind] - 1).sqrt()*et_control
                else:
                    x0_pred = self.diffusion.predict_x_from_eps(cxt, et_control, t)
                if self.hparams.clip_images:
                    x0_pred = torch.clamp(x0_pred, self.hparams.clip_range[0], self.hparams.clip_range[1])
                score_diff = ((et - et_control) ** 2).reshape(bs, -1).sum(dim=1)
                c_score = w_score * score_diff

                # Control loss
                control_loss = (
                    ((self.F(xt, ut_clone, t=t, **kwargs) - xt) ** 2).reshape(bs, -1).sum(dim=1)
                )

                c_control = w_control * control_loss * (gamma_t**2)

                # Terminal Cost
                c_terminal = fiber_loss_fct(self.subject_model(x0_pred), y_t).reshape(bs, -1).sum(dim=1)
                c_terminal = w_terminal * c_terminal

                # Aggregate Cost and optimize
                c_t = c_score + c_control + c_terminal

                # print(
                #     f"Diffusion step: {ti} Terminal Loss: {c_terminal.mean().item()} "
                #     f"Control loss: {c_control.mean().item()} Score loss: {c_score.mean().item()}"
                # )
                

                optimizer.zero_grad()
                c_t.sum().backward()
                g = ut_clone.grad
                norms = g.flatten(1).norm(dim=1, keepdim=True)
                g.mul_(torch.clamp(1.0 / (norms + 1e-6), max=1.0).view(-1, *[1]*(g.dim()-1)))
                optimizer.step()
            if self.hparams.N > 0 and gamma_t != 0:
                # refresh=False: set_description redraws by default, which
                # writes a line per step off a terminal no matter what
                # mininterval says. The bar's own throttled update carries the
                # latest description along with it.
                line = (f"Diffusion step: {ti} Terminal Loss: {c_terminal.mean().item()} "
                        f"Control loss: {c_control.mean().item()} "
                        f"Score loss: {c_score.mean().item()}")
                pbar.set_description(line, refresh=False)
                if i == len(ts) - 1:
                    # The last step, printed unconditionally rather than left to
                    # the throttled bar. The paper's surviving job logs carry this
                    # exact line, and its value at step 0 squared tracks the
                    # published fiber loss closely: the cue conflict runs end at
                    # 21.8 and 5.9 for DINOv2 and ResNet-50, against Table 5's 487
                    # and 38.1. So a run can be compared against those logs from
                    # its own output, with no sampling or evaluation at all --
                    # which is the only calibration left for the settings whose
                    # configuration was lost.
                    print(line, flush=True)
            ###########################################
            ############## DDIM update ################
            ###########################################
            with torch.no_grad():

                u_t = ut_clone.detach()
                cxt = self.F(xt, u_t, t=t, **kwargs)
                
                if self.hparams.ancestral_sampling:
                    et_control, log_var = self.generative_model(cxt, y, t, predict_variance=True)
                    x0_pred = cxt/alphas_cumprod[time_rev_ind].sqrt() - (1/alphas_cumprod[time_rev_ind] - 1).sqrt()*et_control
                    # x0_pred = self.diffusion.predict_x_from_eps(xt, et_control, t)
                    if self.hparams.clip_images:
                        x0_pred = torch.clamp(x0_pred, self.hparams.clip_range[0], self.hparams.clip_range[1])
                    posterior_mean_coef1 = (
                        beta_eff * alphas_cumprod_prev[time_rev_ind].sqrt() / (1.0 - alphas_cumprod[time_rev_ind])
                    )
                    posterior_mean_coef2 = (
                        (1.0 - alphas_cumprod_prev[time_rev_ind])
                        * (1 - beta_eff).sqrt()
                        / (1.0 - alphas_cumprod[time_rev_ind])
                    )
                    xt = (
                        posterior_mean_coef1 * x0_pred
                        + posterior_mean_coef2 * cxt
                    )
                    
                    min_log = torch.log(
                        beta_eff * (1.0 - alphas_cumprod_prev[time_rev_ind]) / (1.0 - alphas_cumprod[time_rev_ind])
                    )
                    max_log = torch.log(beta_eff)
                    
                    if self.hparams.variance_type == "learned_range":
                        # The log_var is [-1, 1] for [min_var, max_var].
                        frac = (log_var + 1) / 2
                        model_log_variance = frac * max_log + (1 - frac) * min_log
                    elif self.hparams.variance_type == "large":
                        model_log_variance = max_log 
                    elif self.hparams.variance_type == "small":
                        model_log_variance = min_log

                    if ti > 0:
                        noise = torch.randn_like(xt) * (0.5*model_log_variance).exp()
                        xt = xt + noise
                else:
                    if self.hparams.variance_type == "learned_range":
                        et_control, log_var = self.generative_model(cxt, y, t, predict_variance=True)
                        frac = (log_var + 1) / 2
                        c1 = (c1_large.log() * frac + c1_small.log() * (1 - frac)).exp()
                    else:
                        et_control = self.generative_model(cxt, y, t)
                    x0_pred = self.diffusion.predict_x_from_eps(cxt, et_control, t)
                    if self.hparams.clip_images:
                        x0_pred = torch.clamp(x0_pred, self.hparams.clip_range[0], self.hparams.clip_range[1])
                    xt = (
                        alpha_s.sqrt() * x0_pred
                        + c1 * torch.randn_like(xt)
                        + c2 * et_control
                    )
                uts.append(u_t.cpu())

            xt_s.append(xt.cpu())
            x0_s.append(x0_pred.cpu())

        return list(reversed(xt_s)), list(reversed(x0_s))

    # `causal_*` carry the control over from the previous denoising step and so
    # only decide what the *first* step starts from.
    INIT_CONTROL = ("zero", "random", "causal_zero", "causal_random")
    INIT_XT = ("random", "sdedit", "guided")

    def initialize_ut(self, ut, i):
        """The control the optimiser starts this denoising step from."""
        init_control = self.hparams.init_control
        if init_control not in self.INIT_CONTROL:
            # Raise rather than falling off the end and returning None, which
            # surfaces as "unsupported operand type(s) for +: NoneType" several
            # frames away.
            raise ValueError(
                f"unknown init_control {init_control!r}; expected one of "
                f"{', '.join(self.INIT_CONTROL)}")

        if init_control == "zero":  # constant zero
            return torch.zeros_like(ut)
        elif init_control == "random":  # constant random
            return torch.randn_like(ut)
        elif i > 0:  # causal_*: carry the previous step's control over
            return ut
        elif init_control == "causal_zero":
            return torch.zeros_like(ut)
        else:  # causal_random
            return torch.randn_like(ut)

    def initialize(self, x, y, ts, **kwargs):
        """x_T, the state the reverse process starts from.

        random: x_T ~ N(0, 1)
        sdedit: x_T ~ q(x_t | x), the query noised to the first timestep
        guided: x_T ~ DDPM(H^(y_0)) - Only for Linear IP
        """
        init_scheme = self.hparams.init_xT
        if init_scheme not in self.INIT_XT:
            # Raise rather than falling through to "random", where a typo would
            # silently sample unconditionally from the prior.
            raise ValueError(
                f"unknown init_xT {init_scheme!r}; expected one of "
                f"{', '.join(self.INIT_XT)}")

        if init_scheme == "sdedit":
            n = x.size(0)
            ti = ts[-1]
            t = torch.ones(n).to(x.device).long() * ti
            alpha_t = self.diffusion.alpha(t).view(-1, *([1] * (x.ndim - 1)))
            return x * alpha_t.sqrt() + torch.randn_like(x) * (1 - alpha_t).sqrt()
        elif init_scheme == "guided":
            raise NotImplementedError(
                "guided initialization is not implemented (it could be useful "
                "where the subject model has a decoder)")
        else:
            return torch.randn_like(x)

class SampleRefinement:
    """
    Class for refining samples using gradient descent or autoencoder latent space optimization.
    """
    def __init__(self, subject_model, autoencoder=None):
        self.subject_model = subject_model
        self.autoencoder = autoencoder
        subject_model.eval()
        if self.autoencoder is not None:
            self.autoencoder.eval()

    @torch.enable_grad()
    def refine_with_gradient_descent(self, samples, originals, steps=25, lr=0.01):
        samples = samples.clone().requires_grad_(True).to(device)
        optimizer = torch.optim.Adam([samples], lr=lr)
        original_embeddings = self.subject_model(originals.to(device)).detach()
        criterion = torch.nn.MSELoss(reduce="mean")
        loss_start = 0.0
        loss_end = 0.0
        
        for i in trange(steps):
            optimizer.zero_grad()
            loss = criterion(self.subject_model(samples), original_embeddings)
            if i == 0:
                loss_start = torch.sqrt(loss).item()
            if i == steps - 1:
                loss_end = torch.sqrt(loss).item()
            loss.backward()
            optimizer.step()
    
        print(f"Loss before refinement: {loss_start}, after refinement: {loss_end}")
        return samples.detach()
    
    @torch.enable_grad()
    def refine_in_ae_latent_space(self, samples, originals, steps=25, lr=0.01):
        assert self.autoencoder is not None, "Autoencoder must be provided for latent space refinement."
        with torch.no_grad():
            z_samples = self.autoencoder.encode(samples).detach()
        z_samples = z_samples.clone().requires_grad_(True).to(device)
        optimizer = torch.optim.Adam([z_samples], lr=lr)
        original_embeddings = self.subject_model(originals.to(device)).detach()
        criterion = torch.nn.MSELoss(reduce="mean")
        loss_start = 0.0
        loss_end = 0.0
        
        for i in trange(steps):
            optimizer.zero_grad()
            decoded = self.autoencoder.decode(z_samples)
            loss = criterion(self.subject_model(decoded), original_embeddings)
            if i == 0:
                loss_start = torch.sqrt(loss).item()
            if i == steps - 1:
                loss_end = torch.sqrt(loss).item()
            loss.backward()
            optimizer.step()
    
        print(f"Loss before refinement: {loss_start}, after refinement: {loss_end}")
        return self.autoencoder.decode(z_samples).detach().reshape(samples.shape)

@dataclass
class TimestepConfig:
    t_start: int = 0
    t_end: int = 1000
    num_steps: int = 100
    seed: int = 0
    stride: str = "ddpm_uniform"

def get_timesteps(cfg: TimestepConfig):
    """Denoising timestep grid: `num_steps` increasing ints in [t_start, t_end).

    Vendored from oc-guidance (https://github.com/czi-ai/oc-guidance,
    MIT License, Copyright (c) 2024 czi-ai), which introduced NDTM. The original
    reads its bounds off a nested Hydra config; this takes a TimestepConfig
    directly and keeps the three strides that are actually used here.

    Every stride returns the same thing: a list of `num_steps` strictly
    increasing integers, none of them equal to t_end. Both parts matter.

    `ddpm_uniform` cannot be `range(t0, t1, (t1 - t0) // n_steps)`: that has
    length `num_steps` only when `num_steps` divides the interval, and 400 and
    800 do not divide 1000 -- they would run 500 and 1000 steps instead, both
    points on the steps axis of the colorMNIST ablation.

    `uniform` and `quadratic` return ints over the half-open interval: a float
    cannot index the alpha table, and t_end is one past its last entry.
    """
    t0, t1, n_steps = int(cfg.t_start), int(cfg.t_end), int(cfg.num_steps)
    span = t1 - t0
    if not 0 < n_steps <= span:
        raise ValueError(
            f"num_steps must be between 1 and t_end - t_start = {span}, got "
            f"{n_steps}; there are only {span} distinct timesteps to visit")

    if cfg.stride == "ddpm_uniform":
        grid = [t0 + (i * span) // n_steps for i in range(n_steps)]
    elif cfg.stride == "uniform":
        grid = [t0 + round(i * span / n_steps) for i in range(n_steps)]
    elif cfg.stride == "quadratic":
        rho = 7
        lo, hi = max(t0, 1) ** (1 / rho), t1 ** (1 / rho)
        grid = [int(t0 + (lo + i / n_steps * (hi - lo)) ** rho) - t0
                for i in range(n_steps)]
    else:
        raise NotImplementedError(f"unknown stride {cfg.stride!r}")

    # A stride that concentrates its steps can round two of them onto the same
    # integer; spread those apart rather than hand back a shorter grid than the
    # caller asked for.
    for i in range(1, n_steps):
        if grid[i] <= grid[i - 1]:
            grid[i] = grid[i - 1] + 1
    if grid[-1] >= t1:
        raise ValueError(
            f"the {cfg.stride!r} stride cannot fit {n_steps} distinct steps "
            f"below t_end = {t1}")
    return grid

class StableDiffusionInterface(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = UNet2DModel.from_pretrained(model_path).to(device)
        self.model.eval()

    def forward(self, x, t, y=None):
        # Assuming y is not used in this case
        noise_pred = self.model(x, t).sample
        return noise_pred

def get_gamma_t_fct(anchorpoints, max_timesteps=1000):
    """A guidance strength schedule: a cosine ramp per (start, end, t_start, t_end).

    The first anchor whose interval contains t wins, so overlapping anchors are
    resolved in the order they are given. Returns a plain float -- it used to
    call torch.cos, which works on a tensor timestep and raises on a bare int,
    so the schedule could only be evaluated from inside the sampling loop.
    """
    def gamma_t(t):
        if isinstance(t, torch.Tensor):
            if t.ndim > 0:
                assert torch.all(torch.abs(t - t[0]) < 1e-6), \
                    "the guidance schedule is a function of one timestep, but " \
                    "this batch holds several"
                t = t[0]
            t = t.item()
        for start, end, t_start, t_end in anchorpoints:
            if t > t_start or t < t_end:
                continue
            t_max = t_start - t_end
            t_cur = t_start - t
            return end + 0.5*(start - end)*(1 + math.cos(math.pi*t_cur/t_max))
        raise ValueError(
            f"timestep {t} is not covered by any anchor point in {anchorpoints}")
    # kept so a sampling run can record its schedule; the closure itself does not pickle
    gamma_t.anchorpoints = anchorpoints
    gamma_t.max_timesteps = max_timesteps
    return gamma_t

class NearestNeighborSearch:
    def __init__(self, train_ds, val_ds, test_ds, subject_model, batch_size=32):
        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        self.val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)
        self.subject_model = subject_model

    @torch.no_grad()
    def find_nearest_neighbor(self, original, use_datasets="all", identity_thresh=1e-5):
        if isinstance(use_datasets, str):
            use_train = use_datasets in ["train", "all"]
            use_val = use_datasets in ["val", "all"]
            use_test = use_datasets in ["test", "all"]
        elif isinstance(use_datasets, (list, tuple)):
            use_train = "train" in use_datasets
            use_val = "val" in use_datasets
            use_test = "test" in use_datasets

        best_dist = torch.inf
        best_sample = None
        if use_train:
            print("Searching train dataset...")
            best_dist, best_sample = self.search_in_dataset(original, 
                                                       self.train_loader, 
                                                       best_dist, 
                                                       best_sample,
                                                       identity_thresh=identity_thresh)
        if use_val:
            print("Searching val dataset...")
            best_dist, best_sample = self.search_in_dataset(original, 
                                                       self.val_loader, 
                                                       best_dist, 
                                                       best_sample,
                                                       identity_thresh=identity_thresh)
        if use_test:
            print("Searching test dataset...")
            best_dist, best_sample = self.search_in_dataset(original, 
                                                       self.test_loader, 
                                                       best_dist, 
                                                       best_sample,
                                                       identity_thresh=identity_thresh)
        return best_dist, best_sample

    @torch.no_grad()
    def search_in_dataset(self, original, loader, best_dist, best_sample, identity_thresh=1e-5):
        original_embedding = self.subject_model(original).detach()
        for batch in tqdm(loader):
            sample = batch[0].to(original.device).reshape(-1, *original.shape[1:])
            sample_embedding = self.subject_model(sample).detach()
            dist = torch.norm(original_embedding - sample_embedding, p=2, dim=1)
            keep = dist > identity_thresh
            if not keep.any():
                # Every row of this batch is the query itself -- the last batch
                # of a shard, or any batch of a dataset with duplicates. Skip it:
                # argmin over the empty result raises and kills the search.
                continue
            dist, sample = dist[keep], sample[keep]
            min_in_batch = torch.argmin(dist)
            dist, sample = dist[min_in_batch], sample[min_in_batch]
            if dist < best_dist:
                best_dist = dist
                best_sample = sample
        return best_dist, best_sample
