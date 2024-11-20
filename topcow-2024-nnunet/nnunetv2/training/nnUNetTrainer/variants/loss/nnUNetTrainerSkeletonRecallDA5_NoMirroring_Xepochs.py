import torch
from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerSkeletonRecallDA5_NoMirroring import nnUNetTrainerSkeletonRecallDA5_NoMirroring


class nnUNetTrainerSkeletonRecallDA5_NoMirroring_5epochs(nnUNetTrainerSkeletonRecallDA5_NoMirroring):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 5
