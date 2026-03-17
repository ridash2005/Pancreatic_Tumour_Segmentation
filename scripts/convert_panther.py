import os
import json
import SimpleITK as sitk
from tqdm import tqdm

def convert_panther_to_nifti(src_dir, output_dir, dataset_name, labels_dict):
    """
    Converts PANTHER .mha files to .nii.gz and stores them in output_dir.
    """
    images_out = os.path.join(output_dir, "imagesTr")
    labels_out = os.path.join(output_dir, "labelsTr")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    src_images = os.path.join(src_dir, "ImagesTr")
    src_labels = os.path.join(src_dir, "LabelsTr")

    if not os.path.exists(src_images):
        print(f"Directory not found: {src_images}")
        return

    img_files = [f for f in os.listdir(src_images) if f.endswith(".mha") and not f.startswith(".")]
    
    print(f"Converting {dataset_name} to NIfTI ({len(img_files)} images)...")
    
    for img_file in tqdm(img_files):
        # Image name: 10000_0001_0000.mha or 10303_0000.mha
        # Case ID: 10000_0001 or 10303
        case_id = img_file.replace("_0000.mha", "")
        
        # Convert Image
        try:
            img_mha = sitk.ReadImage(os.path.join(src_images, img_file))
            sitk.WriteImage(img_mha, os.path.join(images_out, f"{case_id}_0000.nii.gz"))
            
            # Convert Label
            # Label name: 10000_0001.mha or 10303.mha
            lbl_file = f"{case_id}.mha"
            lbl_path = os.path.join(src_labels, lbl_file)
            if os.path.exists(lbl_path):
                lbl_mha = sitk.ReadImage(lbl_path)
                sitk.WriteImage(lbl_mha, os.path.join(labels_out, f"{case_id}.nii.gz"))
        except Exception as e:
            print(f"Error converting {case_id}: {e}")

    # Create dataset.json in the output directory
    dataset_json = {
        "channel_names": {"0": "MRI"},
        "labels": labels_dict,
        "numTraining": len(img_files),
        "file_ending": ".nii.gz"
    }
    
    with open(os.path.join(output_dir, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=4)
        
    print(f"Successfully stored NIfTI files in: {output_dir}")

if __name__ == "__main__":
    # Get absolute path of current repo
    base_repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_panther_src = os.path.join(base_repo, "Dataset", "PANTHER")
    base_nifti_out = os.path.join(base_repo, "Dataset", "PANTHER_NIfTI")
    
    panther_labels = {"background": 0, "tumor": 1}
    
    # Task 1
    t1_src = os.path.join(base_panther_src, "PANTHER_Task1")
    t1_out = os.path.join(base_nifti_out, "Task1")
    convert_panther_to_nifti(t1_src, t1_out, "Panther_Task1", panther_labels)
    
    # Task 2
    t2_src = os.path.join(base_panther_src, "PANTHER_Task2")
    t2_out = os.path.join(base_nifti_out, "Task2")
    convert_panther_to_nifti(t2_src, t2_out, "Panther_Task2", panther_labels)
