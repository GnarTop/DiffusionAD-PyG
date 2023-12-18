from __future__ import annotations

import logging

import torch
from omegaconf import DictConfig, ListConfig
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import Tensor
from torch.optim.optimizer import Optimizer

from anomalib.models.diffusion_ad.loss import DiffusionADLoss
from anomalib.models.diffusion_ad.torch_model import DifussionADModel
from anomalib.models.diffusion_ad.subnetwork import SegmentationSubNetwork
from anomalib.models.components import AnomalyModule


class DiffusionAD(AnomalyModule):
    def __init__(
        self,
        input_size: tuple[int, int],
        backbone: str,
        gamma_c: int = 1,
        gamma_d: int = 1,
        num_nearest_neighbors: int = 3,
        num_hard_negative_features: int = 3,
        radius: float = 1e-5,
    ) -> None:
        super().__init__()
        self.model: DifussionADModel = DifussionADModel(
            # TODO
            # --?--
        )


        self.loss = DiffusionADLoss(
            # TODO
            # --?--
        )


    def training_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        pass

    def validation_step(self, batch: dict[str, str | Tensor], *args, **kwargs) -> STEP_OUTPUT:
        pass



class DiffusionADLightning(DiffusionAD):
    def __init__(self, hparams: DictConfig | ListConfig) -> None:
        pass

    def configure_callbacks(self) -> list[EarlyStopping]:
        pass

    def configure_optimizers(self) -> torch.optim.Optimizer:
        pass