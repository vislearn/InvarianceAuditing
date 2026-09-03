import torch
import torch.nn as nn
from fff.utils.truncate import Truncate
from fff.fif import FreeFormInjectiveFlow
from fff.fff import FreeFormFlow
import os
from warnings import warn
import torch.nn.functional as F
from fff.model.utils import guess_image_shape
from math import prod
from fff.data.utils import decolorize, normalize_ct_image
from fff.utils.checkpoint import default_map_location, load_checkpoint


class SubjectModel(torch.nn.Module):
    def __init__(
        self,
        subject_model_path,
        model_type=None,
        truncate=False,
        fixed_transform=None,
        empty_condition=False,
    ):
        super(SubjectModel, self).__init__()

        # Set before the early return below: forward() and encode() read both
        # and must find them even when no model was loaded, so they can raise
        # their own RuntimeError rather than an AttributeError.
        self.fixed_transform = None
        self.empty_condition = empty_condition

        if subject_model_path is None:
            self.model = None
            warn("No subject model path given, continuing without subject model")
            return
        if not os.path.exists(subject_model_path):
            # Raise here: a wrong path that falls through to the loader comes
            # back as "cannot load <path>", which reads like an unsupported
            # format rather than a missing file.
            raise FileNotFoundError(
                f"subject model path {subject_model_path} given, but does not exist")

        if model_type == "FreeFormFlow":
            self.model = load_checkpoint(FreeFormFlow, subject_model_path)
            self.model.eval()
        elif model_type in ("FiberModel", "SomeModel", "AutoEncoder"):
            # "SomeModel" and "AutoEncoder" are legacy names recorded in released
            # checkpoints; all three are loaded by FiberModel.
            from fff.fiber_model import FiberModel  # local: avoids a circular import

            self.model = load_checkpoint(FiberModel, subject_model_path)
            self.model.eval()
        elif model_type == "FreeFormInjectiveFlow":
            self.model = load_checkpoint(FreeFormInjectiveFlow, subject_model_path)
            self.model.eval()
        elif model_type == "PrecompiledModel":
            self.model = torch.load(subject_model_path, weights_only=False,
                                    map_location=default_map_location())
            self.model.eval()
        elif model_type == None:
            model_type, self.model = infer_and_load_model_type(subject_model_path)
            self.model.eval()
        else:
            raise NotImplementedError(f"Model type {model_type} not implemented")

        if truncate:
            self.model = Truncate(self.model)

        for param in self.model.parameters():
            param.requires_grad = False

        if fixed_transform is not None:
            if fixed_transform == "decolorize":
                self.fixed_transform = decolorize
            elif fixed_transform == "normalize_ct_image":
                self.fixed_transform = normalize_ct_image
            elif callable(fixed_transform):
                self.fixed_transform = fixed_transform
            else:
                raise NotImplementedError(
                    f"You have to implement {fixed_transform} in subject_model.py"
                )

    def forward(self, x, *c, **kwargs):
        if self.fixed_transform is not None:
            x = self.fixed_transform(x)
        if self.model is None:
            raise RuntimeError("No subject model loaded")
        if self.empty_condition:
            c = [torch.empty(x.shape[0], 0, device=x.device)]
        return self.model(x, *c, **kwargs)

    def encode(self, x, *c, **kwargs):
        if self.empty_condition:
            c = [torch.empty(x.shape[0], 0, device=x.device)]
        if self.fixed_transform is not None:
            x = self.fixed_transform(x)
        if self.model is None:
            raise RuntimeError("No subject model loaded")
        # Encoders that expose encode() (our flows and fiber models) use it;
        # plain nn.Module subject models are just called. Dispatch on hasattr
        # rather than try/except: a bare except swallows the real error from
        # encode() and re-raises whatever forward() then says, so a conditional
        # model called without its condition reports a missing argument to
        # forward() and points away from the call that actually failed.
        encode = getattr(self.model, "encode", None)
        if callable(encode):
            return encode(x, *c, **kwargs)
        return self.model(x, *c, **kwargs)

    def decode(self, z, *c, **kwargs):
        if self.model is None:
            raise RuntimeError("No subject model loaded")
        if self.empty_condition:
            c = [torch.empty(z.shape[0], 0, device=z.device)]
        return self.model.decode(z, *c, **kwargs)


def infer_and_load_model_type(subject_model_path):
    if subject_model_path.endswith(".ckpt"):
        try:
            model = FreeFormFlow.load_from_checkpoint(subject_model_path)
            return "FreeFormFlow", model
        except Exception:  # not an FFF checkpoint; try the injective flow
            model = FreeFormInjectiveFlow.load_from_checkpoint(subject_model_path)
            return "FreeFormInjectiveFlow", model
    else:
        try:
            model = torch.load(subject_model_path, weights_only=False,
                               map_location=default_map_location())
        except Exception as exc:
            # Carry the underlying cause: a bare "Model type not implemented"
            # hides a missing file or an unpicklable object.
            raise NotImplementedError(
                f"cannot load {subject_model_path}: {exc}") from exc
        if hasattr(model, "encode") and hasattr(model, "decode"):
            return "PrecompiledModel", model
        raise NotImplementedError(
            f"{subject_model_path} unpickled to {type(model).__name__}, which has "
            f"no encode/decode; pass model_type explicitly")


class BiomedClipModel(torch.nn.Module):
    def __init__(self, model, tokenizer, image_only=True):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.image_only = image_only

    def encode(self, x, c=None):
        if x.ndim == 2:
            x = x.reshape(x.shape[0], *guess_image_shape(prod(x.shape[1:])))
        if self.image_only:
            text = None
        else:
            text = self.tokenizer(c, context_length=256).to(x.device)

        images = self.preprocess(x)
        image_features, text_features, logit_scale = self.model(images, text)
        if self.image_only:
            return image_features
        return image_features, text_features

    def decode(self, x, c=None):
        raise (RuntimeError("BiomedClip has no decoder"))

    def preprocess(self, img):
        _, ch, h, w = img.shape

        if h < w:
            new_h, new_w = 224, int(w * 224 / h)  # Scale width
        else:
            new_w, new_h = 224, int(h * 224 / w)  # Scale height

        img = F.interpolate(
            img,
            size=(new_h, new_w),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

        # Center crop manually
        top = (new_h - 224) // 2
        left = (new_w - 224) // 2

        img = img[..., top : top + 224, left : left + 224]

        # Convert to RGB (if needed)
        if ch == 1:  # Grayscale input
            img = img.repeat(1, 3, 1, 1)  # Expand to RGB channels

        # Normalize
        mean = torch.tensor(
            [0.48145466, 0.4578275, 0.40821073], device=img.device
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.26862954, 0.26130258, 0.27577711], device=img.device
        ).view(1, 3, 1, 1)
        img = (img - mean) / std

        return img
