#preprocessing flir dataset to keep only person class annotations in COCO format

import os
from pathlib import Path
from tqdm import tqdm
import json

def filter_coco_annotations(json_path):
    """Filter COCO annotations to keep only person class"""
    print(f"Processing annotations in: {json_path}")
    
    try:
        # Read the COCO JSON file
        with open(json_path, 'r') as f:
            coco_data = json.load(f)
        
        filtered_annotations = [
            ann for ann in coco_data['annotations'] 
            if ann['category_id'] == 1  # Person class
        ]
        
        # Update annotations in the COCO data
        coco_data['annotations'] = filtered_annotations
        
        # save filtered annotations back to the same file
        with open(json_path, 'w') as f:
            json.dump(coco_data, f, indent=2)
            
        print(f"Filtered annotations: {len(filtered_annotations)} person annotations kept")
            
    except Exception as e:
        print(f"Error processing {json_path}: {str(e)}")

if __name__ == "__main__":
    BASE_PATH = "/Users/isha.yerramilli-rao/FLIR_ADAS_v2"  # Base path where your dataset is located
    
    # Process each split with the new directory structure
    splits = {
        'test': 'images_thermal_test'
    }
    
    for split_name, dir_name in splits.items():
        json_path = Path(BASE_PATH) / dir_name / 'coco.json'  # coco.json is directly in the split directory
        print(f"\nProcessing {split_name} split...")
        filter_coco_annotations(json_path) 