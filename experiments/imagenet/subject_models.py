"""Subject models audited on ImageNet and the cue conflict dataset.

Each takes images in [-1, 1] at the diffusion model's 256x256 resolution and
returns the representation whose fiber we sample from.
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DinoSubjectModel(nn.Module):
    """DINOv2 ViT-B/14 backbone (paper Sections 4.2 and 4.3)."""

    def __init__(self, model_name="dinov2_vitb14"):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", model_name)
        self.model.eval()
        self.preprocess = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def forward(self, x):
        return self.model(self.preprocess((x + 1) / 2))


class InceptionSubjectModel(nn.Module):
    """The PyTorch InceptionV3 pool3 features that FID is computed on (Section 4.2)."""

    def __init__(self):
        super().__init__()
        from fff.evaluate.fid_old import InceptionV3Features

        self.model = InceptionV3Features(torch.device("cpu"))
        self.model.eval()

    def forward(self, x):
        # InceptionV3Features does not resize, and the network expects 299.
        # Feeding it the diffusion model's 256 changes the features, and so the
        # fiber loss, by more than the guidance does.
        x = torch.nn.functional.interpolate(
            (x + 1) / 2, size=299, mode="bilinear", align_corners=False)
        return self.model(x)


class ResNetSubjectModel(nn.Module):
    """ImageNet ResNet-50 after the final pooling layer (Section 4.3)."""

    def __init__(self):
        super().__init__()
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
        resnet = torchvision.models.resnet50(weights=weights)
        resnet.fc = nn.Identity()
        self.model = resnet
        self.model.eval()
        # The weights carry their own transform: resize 232, crop 224, bilinear.
        # Not the usual 256/bicubic -- the V2 recipe retrained at a different
        # scale, and substituting it shifts every feature.
        self.preprocess = weights.transforms()

    def forward(self, x):
        return self.model(self.preprocess((x + 1) / 2))


SUBJECT_MODELS = {
    "dinov2": DinoSubjectModel,
    "inception": InceptionSubjectModel,
    "resnet50": ResNetSubjectModel,
}


def build_subject_model(name: str, device):
    model = SUBJECT_MODELS[name]().to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model
