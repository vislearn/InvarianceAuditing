import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageNet
from warnings import warn
import albumentations as A
from fff.data.utils import TrainValTest
from albumentations.pytorch import ToTensorV2
from PIL import Image

def get_imagenet_dataset(root: str, **kwargs) -> TrainValTest:
    train_dataset = ImageNetDataset(mode="train", root=root, **kwargs)
    valid_dataset = ImageNetDataset(mode="valid", root=root, **kwargs)
    test_dataset = ImageNetDataset(mode="test", root=root, **kwargs)
    return train_dataset, valid_dataset, test_dataset

class ImageNetDataset(Dataset):
    """
    ImageNet dataset with the same interface as CueConflictDataset.
    """

    RESIZE_MODES = ("square", "shortest", "crop")

    def __init__(self, mode, root, resize_to=None, normalize=True,
                 return_tuple=True, resize_mode="square"):
        if mode not in ["train", "valid", "test"]:
            raise ValueError(f"Invalid mode: {mode}")

        # torchvision uses 'val' instead of 'valid'
        split = "val" if mode in ["valid", "test"] else "train"

        self.return_tuple = return_tuple
        try:
            self.dataset = ImageNet(root=root, split=split)
        except (RuntimeError, OSError):
            warn(f"Could not load split {split}, leaving class uninitialized")
        if resize_mode not in self.RESIZE_MODES:
            raise ValueError(f"resize_mode must be one of {self.RESIZE_MODES}, "
                             f"not {resize_mode!r}")
        transforms = []
        if resize_to is not None:
            if resize_mode == "square":
                # What the sampling runs used. ImageNet validation images are not
                # square -- 375x500 is typical -- so this squashes them, and the
                # aspect ratio is gone before any subject model's own resize sees
                # the image. Harmless on a square dataset like cue conflict.
                transforms.append(A.Resize(resize_to, resize_to))
            elif resize_mode == "shortest":
                # Shortest side to resize_to, then centre crop: the convention
                # the ImageNet models were evaluated under.
                transforms.append(A.SmallestMaxSize(max_size=resize_to))
                transforms.append(A.CenterCrop(height=resize_to, width=resize_to))
            else:
                # No rescaling at all: take a resize_to window out of the middle
                # at native scale. Preserves aspect ratio like "shortest" but
                # keeps the original detail and discards more of the frame.
                # pad_if_needed because a minority of validation images are
                # smaller than 256 on a side.
                transforms.append(A.CenterCrop(height=resize_to, width=resize_to,
                                               pad_if_needed=True))

        if normalize:
            transforms.append(
                A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
            )
        transforms.append(ToTensorV2())

        self.transform = A.Compose(transforms)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]

        img = np.array(img.convert("RGB"))
        img = self.transform(image=img)["image"].float()

        if not self.return_tuple:
            return {
                "image": img,
                "label": label,
                "path": self.dataset.samples[idx][0],
            }
        else:
            return img, label
