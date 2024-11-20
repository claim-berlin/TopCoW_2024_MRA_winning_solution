import torch
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainer_CE_DC_CLDC import nnUNetTrainer_CE_DC_CLDC


class nnUNetTrainer_CE_DC_CLDC_NoDeepSupervision(nnUNetTrainer_CE_DC_CLDC):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.enable_deep_supervision = False
    
