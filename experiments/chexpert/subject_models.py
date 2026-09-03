"""Subject models audited on CheXpert (paper Section 4.5, Appendix B.4).

Two classifiers over five findings: a frozen BiomedCLIP feature extractor with a
residual classification head, and an end-to-end ConvNeXt-V2-tiny. Both take
normalised grayscale 384x384 images and return the five class logits, which is
where the fiber loss is measured.

The Qwen-2B vision encoder (Section 4.5, Figure 10) is also audited here. It is
not a classifier: phi(x) is the mean-pooled patch embedding, so its fiber loss is
measured in embedding space rather than on logits.
"""

import os
from dataclasses import dataclass

import torch
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoModelForImageClassification
from transformers.modeling_outputs import ModelOutput

@dataclass
class ClassifierOutput(ModelOutput):
    """What the HuggingFace Trainer expects back from a classifier."""
    loss: torch.Tensor = None
    logits: torch.Tensor = None


IMAGE_SIZE = 384
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


class ResidualMLPBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return x + self.net(x)
    

class ResNetClassifier(nn.Module):
    def __init__(self, embed_dim, num_labels, blocks=2, dropout=0.0, hidden_dim_factor=4):
        super().__init__()
        hidden_dim = int(embed_dim * hidden_dim_factor)

        # One or more residual blocks; add more if desired
        self.blocks = nn.ModuleList()
        for _in in range(blocks):
            self.blocks.append(ResidualMLPBlock(embed_dim, hidden_dim, dropout))

        # Final classification head
        self.out = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_labels),
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.out(x)


BIOMEDCLIP_HUB_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


class BiomedClipVisionEncoder(nn.Module):
    """The image tower of BiomedCLIP, resizing and normalising the way it expects.

    The checkpoints were trained against a pickled copy of this wrapper, so the
    submodule has to stay named `model` for the stored keys (`biomedclip.model.*`)
    to line up. The text tower is kept for the same reason and never called.
    """

    def __init__(self, hub_id=BIOMEDCLIP_HUB_ID):
        super().__init__()
        import open_clip
        self.model, _ = open_clip.create_model_from_pretrained(hub_id)

    @staticmethod
    def preprocess(img):
        """Shortest side to 224, centre crop, grayscale to RGB, CLIP normalisation."""
        _, channels, h, w = img.shape
        if h < w:
            new_h, new_w = 224, int(w * 224 / h)
        else:
            new_w, new_h = 224, int(h * 224 / w)
        img = nn.functional.interpolate(img, size=(new_h, new_w), mode="bicubic",
                                        align_corners=False, antialias=True)
        top, left = (new_h - 224) // 2, (new_w - 224) // 2
        img = img[..., top:top + 224, left:left + 224]
        if channels == 1:
            img = img.repeat(1, 3, 1, 1)
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                            device=img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                           device=img.device).view(1, 3, 1, 1)
        return (img - mean) / std

    def encode(self, x):
        image_features, _, _ = self.model(self.preprocess(x), None)
        return image_features


class BiomedClipClassifier(nn.Module):
    """Frozen BiomedCLIP features with a residual classification head."""

    def __init__(self, num_labels=5, dropout=0.05, hub_id=BIOMEDCLIP_HUB_ID):
        super().__init__()
        self.biomedclip = BiomedClipVisionEncoder(hub_id)
        for p in self.biomedclip.parameters():
            p.requires_grad = False

        embed_dim = self.biomedclip.model.visual.head.proj.out_features
        self.classifier = ResNetClassifier(embed_dim, num_labels, blocks=2, dropout=dropout)

    def forward(self, pixel_values=None, labels=None, **kwargs):
        logits = self.classifier(self.biomedclip.encode(pixel_values))
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return ClassifierOutput(loss=loss, logits=logits)


def to_rgb(x: torch.Tensor, n_channels: int = 1) -> torch.Tensor:
    """Grayscale to 3 channels: what phi's input goes through before the model.

    The pipeline normalises with the grayscale-collapsed ImageNet stats (mean
    0.4490, std 0.2260) and hands the same channel to all three inputs. This is
    not the per-channel convention the classifiers were fine-tuned under --
    `renormalize_grayscale` below converts to that -- but it is what Table 5
    measured, and phi drives the guidance, so changing it changes the samples
    and not only their scores. Do not swap in the renormalised form to "fix"
    the preprocessing without redrawing every CheXpert run.
    """
    return x.repeat(1, 3 // n_channels, 1, 1)


def renormalize_grayscale(x: torch.Tensor) -> torch.Tensor:
    """The per-channel convention the classifiers were fine-tuned under.

    Not applied in the sampling pipeline; see `to_rgb` for why. Provided because
    it is the correct preprocessing for anyone using these weights on their own.

    x is a (B, 1, H, W) tensor normalized with:
        mean = mean(0.485, 0.456, 0.406) = 0.4490
        std  = mean(0.229, 0.224, 0.225) = 0.2260

    This undoes that normalization and re-normalizes each of the
    3 repeated channels with its proper per-channel ImageNet stats.
    """
    gray_mean = IMAGENET_MEAN.mean()   # scalar: 0.4490
    gray_std  = IMAGENET_STD.mean()    # scalar: 0.2260

    # Undo grayscale normalization → back to [0, 1] pixel space
    x = x * gray_std + gray_mean                          # (B, 1, H, W)

    # Repeat to 3 channels
    x = x.repeat(1, 3, 1, 1)                              # (B, 3, H, W)

    # Apply proper per-channel normalization
    mean = IMAGENET_MEAN.view(1, 3, 1, 1).to(x.device)
    std  = IMAGENET_STD.view(1, 3, 1, 1).to(x.device)
    x = (x - mean) / std

    return x


class BiomedClipSubjectModel(nn.Module):
    def __init__(self, model_path, n_channels=1):
        super().__init__()
        self.model = BiomedClipClassifier()
        self.n_channels = n_channels
        weights = load_file(os.path.join(model_path, "model.safetensors"))
        self.model.load_state_dict(weights)
        self.model.eval()

    def forward(self, x):
        return self.model(to_rgb(x, self.n_channels)).logits



class ConvNextClassfierSubjectModel(nn.Module):
    def __init__(self, model_path, n_channels=1):
        super().__init__()
        self.model = AutoModelForImageClassification.from_pretrained(model_path)
        self.model.eval()
        self.n_channels = n_channels

    def forward(self, x):
        return self.model(to_rgb(x, self.n_channels)).logits


class QwenSubjectModel(nn.Module):
    """The Qwen2-VL-2B vision encoder; phi(x) is the mean-pooled patch embedding.

    Audited in Section 4.5 to test whether a VLM's failure on rare anatomy
    (situs inversus) originates in the vision encoder or the language model. The
    encoder is used in isolation: `get_description` is kept because the paper
    checks what the full VLM answers for the same images.

    The processor expects channels-first uint8-range images, so inputs arrive in
    the pipeline's normalised space and are mapped back before being handed over.
    """

    MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

    def __init__(self, model_id=MODEL_ID, n_channels=1, global_mean=None, global_std=None):
        super().__init__()
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        # No device_map="auto": NDTM backpropagates through the vision tower, so
        # it has to sit on a single device, and accelerate's dispatch hooks leave
        # buffers behind when the module is moved afterwards.
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16)
        self.model.eval()
        # Only the vision tower goes to fp32: NDTM differentiates through it and
        # fp16 gradients through the mean-pooled embedding underflow. The language
        # side is never touched while sampling, and casting the whole 2.2B model
        # costs 8.2 GiB against 5.4 GiB this way -- the difference between fitting
        # alongside the diffusion model on an 11 GB card and not.
        _vision_tower(self.model).float()
        self.processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        self.n_channels = n_channels
        gray = n_channels == 1
        self.register_buffer("global_mean", _grayscale_stat(IMAGENET_MEAN, gray)
                             if global_mean is None else global_mean)
        self.register_buffer("global_std", _grayscale_stat(IMAGENET_STD, gray)
                             if global_std is None else global_std)

    def to_vision_device(self, device):
        """Put only the vision tower on the accelerator.

        The language side is idle while sampling -- phi(x) is the mean-pooled
        patch embedding -- but in fp16 it still occupies 3.1 GiB, which is the
        difference between fitting alongside the diffusion model on an 11 GB card
        and not. `get_description` brings it over when it is actually needed.
        """
        _vision_tower(self.model).to(device)
        self.global_mean = self.global_mean.to(device)
        self.global_std = self.global_std.to(device)
        return self

    @property
    def vision_device(self):
        return next(_vision_tower(self.model).parameters()).device

    def _to_processor_space(self, x):
        x = x.repeat(1, 3 // self.n_channels, 1, 1)
        return denormalize(x, self.global_mean, self.global_std)

    def forward(self, x):
        inputs = self.processor(
            images=self._to_processor_space(x),
            text=[""] * len(x),          # vision features only
            padding=True,
            return_tensors="pt",
            input_data_format="channels_first",
        )
        # The processor always returns CPU tensors. Moving them keeps the graph
        # intact -- the fast (torchvision) backend preprocesses with torch ops, so
        # pixel_values still carries grad_fn back to x, which is what NDTM needs.
        device = self.vision_device
        image_embeds = _per_image_tokens(self.model.get_image_features(
            pixel_values=inputs["pixel_values"].to(device),
            image_grid_thw=inputs["image_grid_thw"].to(device),
        ))
        return torch.stack([tokens.mean(dim=0) for tokens in image_embeds])

    @torch.no_grad()
    def get_description(self, x, instruction=None):
        """What the full VLM answers for these images (the claim Figure 10 tests)."""
        if instruction is None:
            instruction = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
                "Describe this image.<|im_end|>\n<|im_start|>assistant\n")
        inputs = self.processor(
            images=self._to_processor_space(x) * 255,
            text=[instruction] * len(x),
            padding=True,
            return_tensors="pt",
            input_data_format="channels_first",
        ).to(self.vision_device)
        self.model.to(self.vision_device)      # the language side may be offloaded
        generated = self.model.generate(**inputs)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def decode(self, y):
        raise NotImplementedError("Qwen does not support decoding.")


def _vision_tower(model):
    """The vision encoder, wherever this transformers version keeps it.

    Up to transformers 4.x it hung off the top-level class as `.visual`; from 5.0
    it lives on the inner `Qwen2VLModel`. Accessing the old path on a new install
    raises AttributeError rather than falling back, so resolve it explicitly.
    """
    for holder in (model, getattr(model, "model", None)):
        tower = getattr(holder, "visual", None)
        if tower is not None:
            return tower
    raise AttributeError("could not locate the Qwen vision tower")


def _per_image_tokens(features):
    """Patch tokens per image, as a sequence of (n_tokens, dim) tensors.

    transformers 4.x returned that sequence directly; 5.0 wraps it in a
    BaseModelOutputWithPooling and puts it on `.pooler_output`. Indexing the
    wrapper positionally does not fail -- it returns a different field -- so this
    has to be distinguished by attribute, not by trying to subscript.
    """
    pooled = getattr(features, "pooler_output", None)
    return features if pooled is None else pooled


def _grayscale_stat(stat, grayscale):
    """Per-channel ImageNet stat, collapsed to one channel for grayscale inputs."""
    stat = stat.view(1, 3, 1, 1)
    return stat.mean(dim=1, keepdim=True) if grayscale else stat


def normalize(img, global_mean, global_std, value_range=(0, 1)):
    #Bring to 0, 1
    img = (img + value_range[0])/(value_range[1] - value_range[0])
    img = (img - global_mean.to(img.device)) / global_std.to(img.device)
    return img

def denormalize(img, global_mean, global_std, clamp=True, value_range=(0, 1)):
    img = img * global_std.to(img.device) + global_mean.to(img.device)
    # Bring into value_range
    img = img*(value_range[1] - value_range[0]) + value_range[0]
    if clamp:
        img = torch.clamp(img, *value_range)
    return img
