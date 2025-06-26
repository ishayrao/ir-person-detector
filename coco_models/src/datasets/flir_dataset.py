import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
import torch.nn as nn
import logging
log = logging.getLogger(__name__)
class FLIRDataset(Dataset):
    def __init__(self, json_file, thermal_dir, transform=None):
        self.thermal_dir = Path(thermal_dir)
        self.transform = transform
        log.info(f"thermal_dir: {self.thermal_dir}")
        
        # Load annotations
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        # Create image_id to annotations mapping
        self.annotations = {}
        for ann in data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)
        
        # Keep only images that have annotations
        self.images = [img for img in data['images'] if img['id'] in self.annotations]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_info = self.images[idx]
        img_path = self.thermal_dir / img_info['file_name']
        #assert str(img_path).startswith("/root/ir-person-detector/"), f"BAD PATH: {img_path}"
        if not img_path.exists():
            raise RuntimeError(f"TRYING TO OPEN NON-EXISTENT FILE: {img_path}")
        image = Image.open(img_path)
        
        # Get annotations for this image
        img_anns = self.annotations[img_info['id']]
        
        # Extract bounding boxes
        boxes = []
        for ann in img_anns:
            x, y, w, h = ann['bbox']
            # Convert from [x,y,w,h] to [x1,y1,x2,y2] format
            boxes.append([x, y, x+w, y+h])
        
        # Convert to tensor
        boxes = torch.tensor(boxes, dtype=torch.float32)
        
        # Create target dictionary
        target = {
            'boxes': boxes,
            'labels': torch.ones(boxes.shape[0], dtype=torch.int64),  # all 1s for person class
            'image_id': torch.tensor([img_info['id']])
        }

        log.info(f"image dims before transform: {image.shape}")  
        for box_idx, box in enumerate(target['boxes']):
            log.info(f"box {box_idx}: {box}")
        log.info(f"image id: {target['image_id'].item()}")
        
        # Apply transforms if any
        if self.transform:
            if isinstance(self.transform, nn.Module):  # torchvision model transforms
                image, target = self.transform(image, target)
            else:  # custom transforms
                image, target = self.transform(image, target)
        
        return image, target 