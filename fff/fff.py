from math import prod

from .base import FreeFormBaseHParams, FreeFormBase, VolumeChangeResult
from .fif import FreeFormInjectiveFlow


class FreeFormFlowHParams(FreeFormBaseHParams):
    pass


class FreeFormFlow(FreeFormBase):
    """
    A FreeFormFlow is a normalizing flow consisting of a pair of free-form
    encoder and decoder.
    """
    hparams: FreeFormFlowHParams

    def __init__(self, hparams: FreeFormFlowHParams | dict):
        if not isinstance(hparams, FreeFormFlowHParams):
            hparams = FreeFormFlowHParams(**hparams)
        super().__init__(hparams)
        if self.data_dim != self.latent_dim:
            raise ValueError("Data and latent dimension must be equal for a FreeFormFlow.")

    def _encoder_volume_change(self, x, c, **kwargs) -> VolumeChangeResult:
        z, jac_enc = self._encoder_jac(x, c, **kwargs)
        jac_enc = jac_enc.reshape(x.shape[0], prod(z.shape[1:]), prod(x.shape[1:]))
        log_det = jac_enc.slogdet()[1]
        return VolumeChangeResult(z, log_det, {})

    def _decoder_volume_change(self, z, c, **kwargs) -> VolumeChangeResult:
        x1, jac_dec = self._decoder_jac(z, c, **kwargs)
        jac_dec = jac_dec.reshape(z.shape[0], prod(x1.shape[1:]), prod(z.shape[1:]))
        log_det = jac_dec.slogdet()[1]
        return VolumeChangeResult(x1, log_det, {})
