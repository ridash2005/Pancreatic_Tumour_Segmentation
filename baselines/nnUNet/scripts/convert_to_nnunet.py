import os
import json
import shutil
import SimpleITK as sitk
from tqdm import tqdm

def convert_to_nnunet_v2(src_dir, output_dir, dataset_id, dataset_name, labels_dict):
    """
    Converts PANTHER .mha files to .nii.gz in nnU-Net v2 format.
    """
    folder_name = f"Dataset{dataset_id:03d}_{dataset_name}"
    dataset_dir = os.path.join(output_dir, folder_name)
    
    images_tr = os.path.join(dataset_dir, "imagesTr")
    labels_tr = os.path.join(dataset_dir, "labelsTr")
    images_ts = os.path.join(dataset_dir, "imagesTs")
    
    os.makedirs(images_tr, exist_ok=True)
    os.makedirs(labels_tr, exist_ok=True)
    os.makedirs(images_ts, exist_ok=True)
    
    src_images_tr = os.path.join(src_dir, "ImagesTr")
    src_labels_tr = os.path.join(src_dir, "LabelsTr")
    
    if not os.path.exists(src_images_tr):
        print(f"Error: Source images directory not found at {src_images_tr}")
        return

    # 1. Process Labeled Training Data
    img_files = [f for f in os.listdir(src_images_tr) if f.endswith(".mha") and not f.startswith(".")]
    
    converted_tr_count = 0
    print(f"Converting labeled training data for {dataset_name}...")
    
    for img_file in tqdm(img_files):
        # Case ID extraction. 
        # Filename example: 10000_0001_0000.mha
        if "_0000.mha" in img_file:
            case_id = img_file.replace("_0000.mha", "")
        else:
            case_id = img_file.replace(".mha", "")
        
        lbl_file = f"{case_id}.mha"
        lbl_path = os.path.join(src_labels_tr, lbl_file)
        
        if os.path.exists(lbl_path):
            try:
                # Convert Image
                img_sitk = sitk.ReadImage(os.path.join(src_images_tr, img_file))
                # nnU-Net expects images to have _0000.nii.gz suffix (for modality 0)
                sitk.WriteImage(img_sitk, os.path.join(images_tr, f"{case_id}_0000.nii.gz"))
                
                # Convert Label
                lbl_sitk = sitk.ReadImage(lbl_path)
                sitk.WriteImage(lbl_sitk, os.path.join(labels_tr, f"{case_id}.nii.gz"))
                
                converted_tr_count += 1
            except Exception as e:
                print(f"Failed to convert {case_id}: {e}")

    # 2. Process Unlabeled Data (Test set)
    # Check if ImagesTr_unlabeled exists
    src_images_ts = os.path.join(src_images_tr, "ImagesTr_unlabeled")
    converted_ts_count = 0
    if os.path.exists(src_images_ts):
        print(f"Converting unlabeled test data for {dataset_name}...")
        ts_files = [f for f in os.listdir(src_images_ts) if f.endswith(".mha") and not f.startswith(".")]
        for img_file in tqdm(ts_files):
            if "_0000.mha" in img_file:
                case_id = img_file.replace("_0000.mha", "")
            else:
                case_id = img_file.replace(".mha", "")
            
            try:
                img_sitk = sitk.ReadImage(os.path.join(src_images_ts, img_file))
                sitk.WriteImage(img_sitk, os.path.join(images_ts, f"{case_id}_0000.nii.gz"))
                converted_ts_count += 1
            except Exception as e:
                print(f"Failed to convert test case {case_id}: {e}")
    else:
        # Some tasks might have ImagesTs
        src_images_ts_alt = os.path.join(src_dir, "ImagesTs")
        if os.path.exists(src_images_ts_alt):
             print(f"Converting ImagesTs for {dataset_name}...")
             ts_files = [f for f in os.listdir(src_images_ts_alt) if f.endswith(".mha") and not f.startswith(".")]
             for img_file in tqdm(ts_files):
                if "_0000.mha" in img_file:
                    case_id = img_file.replace("_0000.mha", "")
                else:
                    case_id = img_file.replace(".mha", "")
                
                try:
                    img_sitk = sitk.ReadImage(os.path.join(src_images_ts_alt, img_file))
                    sitk.WriteImage(img_sitk, os.path.join(images_ts, f"{case_id}_0000.nii.gz"))
                    converted_ts_count += 1
                except Exception as e:
                    print(f"Failed to convert test case {case_id}: {e}")

    # 3. Create dataset.json
    dataset_json = {
        "channel_names": {
            "0": "MRI"
        },
        "labels": labels_dict,
        "numTraining": converted_tr_count,
        "file_ending": ".nii.gz",
        "name": dataset_name,
        "overwrite_image_reader_writer": "SimpleITKIO"
    }
    
    json_path = os.path.join(dataset_dir, "dataset.json")
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)
        
    print(f"Finished {dataset_name}.")
    print(f"  Training cases: {converted_tr_count}")
    print(f"  Test cases: {converted_ts_count}")
    print(f"Dataset stored at: {dataset_dir}")

def main():
    # Use absolute paths
    base_repo = r"d:\GitHub\my_repo\Pancreatic_Tumour_Segmentation"
    src_panther = os.path.join(base_repo, "Dataset", "PANTHER")
    output_nnunet = os.path.join(base_repo, "nnUNet_raw")
    
    labels = {
        "background": 0,
        "tumor": 1
    }
    
    # Task 1
    t1_src = os.path.join(src_panther, "PANTHER_Task1")
    convert_to_nnunet_v2(t1_src, output_nnunet, 501, "PantherTask1", labels)
    
    # Task 2
    t2_src = os.path.join(src_panther, "PANTHER_Task2")
    convert_to_nnunet_v2(t2_src, output_nnunet, 502, "PantherTask2", labels)

    print("\nAll conversions complete!")

if __name__ == "__main__":
    main()
