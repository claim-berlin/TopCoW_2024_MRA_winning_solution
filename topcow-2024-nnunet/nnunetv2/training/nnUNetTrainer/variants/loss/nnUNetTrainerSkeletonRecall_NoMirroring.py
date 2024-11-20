from nnunetv2.training.nnUNetTrainer.variants.loss.nnUNetTrainerSkeletonRecall import nnUNetTrainerSkeletonRecall


class nnUNetTrainerSkeletonRecall_NoMirroring(nnUNetTrainerSkeletonRecall):
    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes = \
            super().configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        mirror_axes = None
        self.inference_allowed_mirroring_axes = None
        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes
