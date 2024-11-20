import shutil
import numpy as np
import SimpleITK as sitk
import os
import torch
import nibabel as nib
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.ensembling.ensemble import ensemble_folders
import glob

from ultralytics import YOLO
import cv2

#######################################################################################
YOLO_FILE_PATH = './models/yolo-cow-detection.pt'
SEGMODEL_DIR_PATH = './models/topcow-claim-models'
IMG_FOLDER = './folder_to_predict'
SEGMENTATION_FOLDER = './segmentations'
#######################################################################################

def main():
    print("Starting segmentation...")
    print("Model:", SEGMODEL_DIR_PATH)

    if not os.path.exists(SEGMENTATION_FOLDER):
        os.makedirs(SEGMENTATION_FOLDER)

    # Load the NIfTI image paths
    img_files = sorted(os.listdir(IMG_FOLDER))
    img_files = [img_path for img_path in img_files if img_path.endswith(".nii.gz")]

    for img_file in img_files:
        # Load the NIfTI image
        img_file = os.path.join(IMG_FOLDER, img_file)
        print("------------------------------------")
        print(os.path.basename(img_file))

        try:
            # Perform segmentation
            seg_data = segment_nifti(img_file)

            # Save the segmentation as a NIfTI file
            seg_file = os.path.basename(img_file).split(".")[0] + "_seg.nii.gz"
            seg_file = os.path.join(SEGMENTATION_FOLDER, seg_file)
            save_nifti_file(seg_data, seg_file, img_file)

            # Load and check saved segmentation
            load_seg, _ = load_nifti_file(seg_file)
            print("LOADED SAVED seg_data unique = ", np.unique(load_seg))
    
        except Exception as e:
            # Print the error and continue to the next file
            print(f"Error encountered while processing {img_file}: {str(e)}")

    print("Segmentation done!")


def load_nifti_file(nifti_path):
    nifti_img = nib.load(nifti_path)
    nifti_data = nifti_img.get_fdata()
    return nifti_data, nifti_img


def save_nifti_file(data, output_path, reference_nifti_path):
    """
    Save the 3D segmentation volume as a NIfTI file.
    Use the reference NIfTI file to copy the affine and header information.
    """
    reference_img = nib.load(reference_nifti_path)
    affine = reference_img.affine
    header = reference_img.header

    nifti_img = nib.Nifti1Image(data, affine, header)
    nib.save(nifti_img, output_path)


def window_CTA(CTA_array, WL=300, WW=1000):
    # Calculate the lower and upper bounds based on WL and WW
    lower_bound = WL - (WW / 2)
    upper_bound = WL + (WW / 2)

    # Clip the CTA array to the calculated bounds
    windowed_CTA_array = np.clip(CTA_array, lower_bound, upper_bound)

    return windowed_CTA_array


def process_nifti_with_yolo(model, nifti_path, output_crop_path):
    """
    Perform YOLO inference on each 2D slice of a NIfTI volume and output the cropped 3D volume.
    The 3D crop is centered around the median bounding box center, and the crop size is based on the median bounding box size.
    """
    # Load the NIfTI image
    nifti_data, nifti_img = load_nifti_file(nifti_path)

    # Squeeze the nifti_data to remove any singleton dimensions
    nifti_data = np.squeeze(nifti_data)

    # Initialize lists to store bounding box centers and sizes for each slice
    bbox_centers = []
    bbox_widths = []
    bbox_heights = []
    slices_with_roi = []

    # Perform inference on each slice of the NIfTI volume
    for slice_idx in range(nifti_data.shape[2]):
        # Get the 2D slice from the NIfTI volume
        slice_data = nifti_data[:, :, slice_idx]

        # Normalize and convert the slice to uint8 format for YOLO
        slice_min = np.min(slice_data)
        slice_max = np.max(slice_data)
        if slice_max > slice_min:
            normalized_slice = (slice_data - slice_min) / (slice_max - slice_min) * 255.0
        else:
            normalized_slice = np.zeros_like(slice_data)

        normalized_slice = normalized_slice.astype(np.uint8)

        # YOLO requires 3 channels, so we need to convert the 2D slice to a 3-channel image
        slice_rgb = cv2.cvtColor(normalized_slice, cv2.COLOR_GRAY2RGB)
        # slice_rgb = np.stack((normalized_slice, normalized_slice, normalized_slice), axis=-1)

        # Perform YOLO inference on the 2D slice
        results = model.predict(source=slice_rgb, verbose=False)

        # Loop through detected bounding boxes
        for result in results:
            if len(result.boxes.xyxy) > 0:
                # Mark the slice as containing a region of interest (ROI)
                slices_with_roi.append(slice_idx)

            for bbox in result.boxes.xyxy:
                # Extract bounding box coordinates (x_min, y_min, x_max, y_max)
                x_min, y_min, x_max, y_max = map(int, bbox)

                # Calculate the center of the bounding box
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2

                # Calculate the width and height of the bounding box
                width = x_max - x_min
                height = y_max - y_min

                # Store the center, width, and height
                bbox_centers.append((center_x, center_y, slice_idx))
                bbox_widths.append(width)
                bbox_heights.append(height)

    if "ct" in os.path.basename(nifti_path).lower():
        print("CT modality detected. Windowing CTA volume!")
        nifti_data = window_CTA(nifti_data)

    # If no bounding boxes were detected, return the original NIfTI data
    if not bbox_centers:
        print("No bounding boxes detected. Using full volume!")
        cropped_volume = nifti_data
        save_nifti_file(cropped_volume, output_crop_path, nifti_path)
        print(f"Cropped volume saved with shape: {cropped_volume.shape}")

        crop_dict = {"size": [cropped_volume.shape[0], cropped_volume.shape[1], cropped_volume.shape[2]],
                     "location": [0, 0, 0]
                     }
        print("Cropped dict:", crop_dict)
        print("Original size:", nifti_data.shape)

        return cropped_volume, crop_dict, nifti_data.shape

    print("Bounding box detected. Cropping volume!")

    # Compute the median center of all bounding boxes
    median_center_x = int(np.median([center[0] for center in bbox_centers]))
    median_center_y = int(np.median([center[1] for center in bbox_centers]))
    median_center_z = int(np.median([center[2] for center in bbox_centers]))

    # Compute the median width and height of all bounding boxes
    median_width = int(np.median(bbox_widths))
    median_height = int(np.median(bbox_heights))

    # Determine the crop size in x and y directions based on the median width and height
    crop_x_size = median_width
    crop_y_size = median_height

    # Determine the crop size in the z direction based on the number of slices with ROIs
    crop_z_size = len(set(slices_with_roi))

    # Define the crop region around the median center
    crop_x_min = max(0, median_center_x - crop_x_size // 2)
    crop_x_max = min(nifti_data.shape[0], median_center_x + crop_x_size // 2)

    crop_y_min = max(0, median_center_y - crop_y_size // 2)
    crop_y_max = min(nifti_data.shape[1], median_center_y + crop_y_size // 2)

    crop_z_min = max(0, median_center_z - crop_z_size // 2)
    crop_z_max = min(nifti_data.shape[2], median_center_z + crop_z_size // 2)

    f_xy = 10
    f_z = 5
    cropped_volume = nifti_data[crop_y_min - f_xy : crop_y_max + f_xy,
                     crop_x_min - f_xy : crop_x_max + f_xy,
                     crop_z_min - f_z : crop_z_max + f_z ]

    # Save the cropped 3D volume as a NIfTI file
    save_nifti_file(cropped_volume, output_crop_path, nifti_path)
    print(f"Cropped volume saved with shape: {cropped_volume.shape}")

    crop_dict = {"size": [crop_y_max + f_xy - (crop_y_min - f_xy),
                          crop_x_max + f_xy - (crop_x_min - f_xy),
                          crop_z_max + f_z - (crop_z_min - f_z)],
                 "location": [crop_y_min - f_xy, crop_x_min - f_xy, crop_z_min - f_z]}
    print("Cropped dict:", crop_dict)
    print("Original size:", nifti_data.shape)

    return cropped_volume, crop_dict, nifti_data.shape


def segment_nifti(img_file) -> np.array:
    """
    args:
    returns:
        np.array - prediction
    """

    ### YOLO DETECTION ###
    yolo_model = YOLO(YOLO_FILE_PATH)  # Load the trained YOLO model

    # Create a temporary folders to store the cropped NIfTI volume
    temp_save_path = "./temp_save"
    shutil.rmtree(temp_save_path, ignore_errors=True)
    os.makedirs(temp_save_path)
    ROI_path = os.path.join(temp_save_path, "ROI_0000.nii.gz")

    _, ROI_dict, original_size = process_nifti_with_yolo(yolo_model, img_file, ROI_path)

    print(f"list temp_save_path = {os.listdir(temp_save_path)}")

    #### SEGMENTATION ###
    # Create a temporary folder to store the segmentation
    temp_save_path_seg_best = "./temp_save_seg_best"
    shutil.rmtree(temp_save_path_seg_best, ignore_errors=True)
    os.makedirs(temp_save_path_seg_best)
    temp_save_path_seg_final = "./temp_save_seg_final"
    shutil.rmtree(temp_save_path_seg_final, ignore_errors=True)
    os.makedirs(temp_save_path_seg_final)
    temp_save_path_seg_ensemble = "./temp_save_seg_ensemble"
    shutil.rmtree(temp_save_path_seg_ensemble, ignore_errors=True)
    os.makedirs(temp_save_path_seg_ensemble)

    predictor_best = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    predictor_final = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )

    # initializes the network architecture, loads the checkpoint
    predictor_best.initialize_from_trained_model_folder(SEGMODEL_DIR_PATH,
                                                   use_folds=(0, 1, 2, 3, 4),
                                                   checkpoint_name='checkpoint_best.pth',
                                                   # select whether to use the best validation checkpoint or the model from the final epoch.
                                                   )
    predictor_final.initialize_from_trained_model_folder(SEGMODEL_DIR_PATH,
                                                        use_folds=(0, 1, 2, 3, 4),
                                                        checkpoint_name='checkpoint_final.pth',
                                                        # select whether to use the best validation checkpoint or the model from the final epoch.
                                                        )

    predictor_best.predict_from_files(temp_save_path,
                                 temp_save_path_seg_best,
                                 save_probabilities=True, overwrite=False,
                                 num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                 folder_with_segs_from_prev_stage=None)
    predictor_final.predict_from_files(temp_save_path,
                                      temp_save_path_seg_final,
                                      save_probabilities=True, overwrite=False,
                                      num_processes_preprocessing=2, num_processes_segmentation_export=2,
                                      folder_with_segs_from_prev_stage=None)

    print(f"list temp_save_path = {os.listdir(temp_save_path)}")
    print(f"list temp_save_path_seg_best = {os.listdir(temp_save_path_seg_best)}")
    print(f"list temp_save_path_seg_final = {os.listdir(temp_save_path_seg_final)}")

    # Ensamble the segmentations from checkpoint best and final
    ensemble_folders([temp_save_path_seg_best, temp_save_path_seg_final], temp_save_path_seg_ensemble, num_processes=4)

    print(f"list temp_save_path_seg_final = {os.listdir(temp_save_path_seg_ensemble)}")

    # Load the ROI segmentation prediction
    pred_array, _ = load_nifti_file(os.path.join(temp_save_path_seg_ensemble, "ROI.nii.gz"))
    # Make sure the classes are integers
    pred_array = np.round(pred_array).astype(np.uint8)

    # Resize the cropped prediction to the original size
    pred_array_resized = np.zeros(original_size, dtype=np.uint8)
    pred_array_resized[ROI_dict["location"][0]:ROI_dict["location"][0] + ROI_dict["size"][0],
                       ROI_dict["location"][1]:ROI_dict["location"][1] + ROI_dict["size"][1],
                       ROI_dict["location"][2]:ROI_dict["location"][2] + ROI_dict["size"][2]] = pred_array
    pred_array = pred_array_resized

    # Translate the class 13 to class 15
    pred_array[pred_array == 13] = 15

    # The output np.array needs to have the same shape as track modality input
    print("pred_array.shape = ", pred_array.shape)
    print("original shape = ", original_size)

    # Remove the temporary folders
    shutil.rmtree(temp_save_path)
    shutil.rmtree(temp_save_path_seg_best)
    shutil.rmtree(temp_save_path_seg_final)
    shutil.rmtree(temp_save_path_seg_ensemble)

    return pred_array


if __name__ == "__main__":
    main()
