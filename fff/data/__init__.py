from .paths import data_root, resolve
from .utils import TrainValTest


def load_dataset(name: str, **kwargs) -> TrainValTest:
    if "root" in kwargs:
        kwargs["root"] = resolve(kwargs["root"])
    kwargs.pop("subject_model_path", None)
    kwargs.pop("subject_model_type", None)
    if name in ["mnist", "mnist_ds", "mnist_split", "emnist", "h5mnist", "cifar10", "celeba"]:
        from .image import (
            get_celeba_datasets,
            get_cifar10_datasets,
            get_emnist_datasets,
            get_h5saved_mnist,
            get_mnist_datasets,
            get_mnist_downsampled,
            get_split_mnist,
        )

        return {
            "mnist": get_mnist_datasets,
            "mnist_ds": get_mnist_downsampled,
            "mnist_split": get_split_mnist,
            "emnist": get_emnist_datasets,
            "h5mnist": get_h5saved_mnist,
            "cifar10": get_cifar10_datasets,
            "celeba": get_celeba_datasets,
        }[name](**kwargs)
    elif name == "chexpert":
        from .chexpert import get_chexpert_dataset

        return get_chexpert_dataset(**kwargs)
    elif name == "imagenet":
        from .imagenet import get_imagenet_dataset

        return get_imagenet_dataset(**kwargs)
    elif name == "cue_conflict":
        from .cue_conflict import get_cue_conflict_dataset

        return get_cue_conflict_dataset(**kwargs)
    elif name == "precompiled_dataset":
        from .saved_datasets import get_saved_dataset

        return get_saved_dataset(**kwargs)

    raise ValueError(f"Unknown dataset {name!r}")


def get_model_path(**dataset_kwargs):
    if "subject_model_path" in dataset_kwargs:
        return resolve(dataset_kwargs["subject_model_path"])
    elif dataset_kwargs["name"] == "precompiled_dataset":
        from .saved_datasets import get_subject_model_path

        return get_subject_model_path(dataset_kwargs["root"])
