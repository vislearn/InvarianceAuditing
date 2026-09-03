"""Compatibility shims for the pinned-version drift in the training stack.

Imported for its side effects from `fff/__init__.py`, so every entry point that
builds an fff model gets them, including `lightning_trainable.launcher.fit`.
"""


def _accept_positional_mapping():
    """Let lightning_trainable's AttributeDict be built from a positional dict.

    `AttributeDict.__init__` (and `HParams.__init__` on top of it) is
    keyword-only, but lightning_utilities rebuilds nested mappings as
    `type(mapping)(OrderedDict(...))`. Lightning walks the hyperparameters that
    way before the TensorBoard logger writes hparams.yaml, so training any config
    with a nested block -- data_set, optimizer, loss_weights, lossless_ae, that
    is to say all of them -- dies at the first epoch with

        TypeError: AttributeDict.__init__() takes 1 positional argument but 2 were given

    depending on the lightning_utilities version. Accepting the positional form
    costs nothing and keeps the configs working across both.
    """
    from lightning_trainable.hparams.attribute_dict import AttributeDict
    from lightning_trainable.hparams import HParams

    for cls in (AttributeDict, HParams):
        original = cls.__dict__.get("__init__")
        if original is None or getattr(original, "_positional_mapping_ok", False):
            continue

        def __init__(self, *args, _original=original, **kwargs):
            if args:
                if len(args) > 1:
                    raise TypeError(f"{type(self).__name__} takes at most one "
                                    f"positional argument, got {len(args)}")
                kwargs = {**dict(args[0]), **kwargs}
            _original(self, **kwargs)

        __init__._positional_mapping_ok = True
        cls.__init__ = __init__


_accept_positional_mapping()
