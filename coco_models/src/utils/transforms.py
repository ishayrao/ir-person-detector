import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from typing import Dict, List, Union, Tuple
import torch.nn.functional as F_nn
import math

import logging

log = logging.getLogger(__name__)

def custom_collate_fn(batch): #called in train.py
    """
    - images: Tensor[batch_size, C, H, W]
    - targets: List[Dict] where each dict has:
        - boxes: Tensor[N, 4]
        - labels: Tensor[N] (all 1s for person class)
        - image_id: Tensor[1] (single tensor for each image)
    """

    images = []
    targets = []
    
    #for image, target in batch:
    for i, (image, target) in enumerate(batch):
        images.append(image)
        # Create target dict with single tensors (not lists)
        target_dict = {
            'boxes': target['boxes'],  # already a tensor from dataset
            'labels': torch.ones(target['boxes'].shape[0], dtype=torch.int64),  # all 1s for person class
            'image_id': target['image_id']  # already a tensor from dataset
        }
        targets.append(target_dict)
    
    # Stack images into a single tensor (they should all be the same size)
    images = torch.stack(images, dim=0)
    
    return images, targets

class DetectionTransform:
    # transform that can be applied to images and bounding boxes
    def __init__(self, transforms: List[Dict]):
        self.transforms = []
        for t in transforms:
            if t['name'] == 'Resize':
                self.transforms.append(('resize', t['params']['size']))
            elif t['name'] == 'Normalize':
                self.transforms.append(('normalize', t['params']['mean'], t['params']['std']))
            elif t['name'] == 'RandomHorizontalFlip':
                self.transforms.append(('flip_h', t['params']['p']))
            elif t['name'] == 'RandomVerticalFlip':
                self.transforms.append(('flip_v', t['params']['p']))
            elif t['name'] == 'RandomRotation':
                self.transforms.append(('rotate', t['params']['degrees']))
    
    def __call__(self, image, target):
        #transforms to both image and bounding boxes
        image = F.to_tensor(image) 
        
        # checking image has 1 channel
        if image.shape[0] != 1:
            raise ValueError(f"Expected 1 channel (thermal), got {image.shape[0]} channels")
        
       # image = image.repeat(3, 1, 1)  # Convert 1-channel to 3-channel (trial to see if this improves overfitting!!)
        for t_name, *params in self.transforms:
            if t_name == 'resize':
                size = params[0]
                # resize with padding
                image, resize_info = resize_with_padding(image, size)

                target_h, target_w = size
                
                # Transform boxes 
                if target is not None and 'boxes' in target:
                    boxes = target['boxes']
                    boxes = transform_boxes(
                        boxes,
                        resize_info['scale'],
                        resize_info['pad_left'],
                        resize_info['pad_top'],
                        target_w,
                        target_h
                    )
                    target['boxes'] = boxes

            elif t_name == 'normalize':
                mean, std = params
                image = F.normalize(image, mean=mean, std=std)
            
            elif t_name == 'flip_h':
                p = params[0]
                if torch.rand(1) < p:
                    image = F.hflip(image)
                    if target is not None and 'boxes' in target:
                        boxes = target['boxes']
                        # new_x = width - old_x
                        boxes[:, [0, 2]] = image.shape[-1] - boxes[:, [2, 0]]
                        target['boxes'] = boxes
            
            elif t_name == 'flip_v':
                p = params[0]
                if torch.rand(1) < p:
                    image = F.vflip(image)
                    if target is not None and 'boxes' in target:
                        boxes = target['boxes']
                        # new_y = height - old_y 
                        boxes[:, [1, 3]] = image.shape[-2] - boxes[:, [3, 1]]
                        target['boxes'] = boxes
            
            elif t_name == 'rotate':
                degrees= params[0]
                if torch.rand(1) < 0.5:  # 50% chance to apply rotation
                    angle = float(torch.empty(1).uniform_(-degrees, degrees).item())
                    image, boxes = rotate_image_and_boxes(image, target['boxes'], angle, expand=False) if target is not None else (image, None)
                    if boxes is not None:
                        target['boxes'] = boxes
        
        return image, target

def build_transforms(cfg: Dict, is_train: bool = True, test: bool = False) -> DetectionTransform:
    if is_train:
        transforms = cfg.dataset.transform.train
    elif test:
        transforms = cfg.dataset.transform.test
    else:
        transforms = cfg.dataset.transform.val
    
    return DetectionTransform(transforms) 

def resize_with_padding(image: torch.Tensor, target_size: Tuple[int, int]) -> Tuple[torch.Tensor, Dict[str, float]]:
   #maintain aspect ratio and add padding
    c, h, w = image.shape
    target_h, target_w = target_size
    
    # calc scaling factors (to find limiting facotr )
    scale_h = target_h / h
    scale_w = target_w / w
    scale = min(scale_h, scale_w)
    
    # new size
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Resize image
    resized_image = F_nn.interpolate(image.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0) #using interpolate since they are already tensors (not PIL)
    
    # padding
    pad_h = target_h - new_h
    pad_w = target_w - new_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    
    # Add padding
    padded_image = F_nn.pad(resized_image, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    return padded_image, {
        'scale': scale,
        'pad_left': pad_left,
        'pad_top': pad_top #resize info
    }

def transform_boxes(boxes: torch.Tensor, scale: float, pad_left: int, pad_top: int, target_w: int, target_h: int) -> torch.Tensor:
    #bounding boxes tranformed in line wiht image tranforms 
    #chnaged (adding clamp to ensure boxes are within image bounds)
    transformed_boxes = boxes.clone()
    transformed_boxes[:, [0, 2]] = (transformed_boxes[:, [0, 2]] * scale + pad_left).clamp(0, max=target_w) # x1 and x2 (since already converted in flir_dataset.py)
    transformed_boxes[:, [1, 3]] = (transformed_boxes[:, [1, 3]] * scale + pad_top).clamp(0, max=target_h)   # y1 and y2
    
    return transformed_boxes

def rotate_image_and_boxes(image: torch.Tensor, boxes: torch.Tensor, angle: float, expand: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    #rotation to images and bounding boxes
    # image dims
    c, h, w = image.shape
    
    # Convert to PIL =
    image_pil = F.to_pil_image(image)
    
    # Rotate image
    rotated_image = F.rotate(image_pil, angle, expand=expand)
    rotated_image = F.to_tensor(rotated_image)
    
    if boxes is None or len(boxes) == 0:
        return rotated_image, boxes
    
    # Convert boxes to points 
    points = boxes_to_points(boxes)
    
    # Rotate points
    center = (w / 2, h / 2)
    rotated_points = rotate_points(points, center, -angle)  # PIL rotation is anticlockwise -- neg angle 
    
    rotated_boxes = points_to_boxes(rotated_points)

    # checks to ensure boxes are valid (CHANGED)
    if rotated_boxes.shape[0] > 0:
        x1, y1, x2, y2 = rotated_boxes.unbind(-1)
        valid = (x2 > x1) & (y2 > y1)
        rotated_boxes = rotated_boxes[valid] #filtering out any invalid boxes
    
    return rotated_image, rotated_boxes

def boxes_to_points(boxes: torch.Tensor) -> torch.Tensor:
    #Convert boxes to 4 corner points
    x1, y1, x2, y2 = boxes.unbind(-1)
    points = torch.stack([
        torch.stack([x1, y1], dim=1),  # top left
        torch.stack([x2, y1], dim=1),  # top right
        torch.stack([x2, y2], dim=1),  # bottom right
        torch.stack([x1, y2], dim=1),  # bottom left
    ], dim=1)
    return points

def points_to_boxes(points: torch.Tensor) -> torch.Tensor:
    #Convert corner points to boxes
    min_coords, _ = torch.min(points, dim=1)
    max_coords, _ = torch.max(points, dim=1)
    return torch.cat([min_coords, max_coords], dim=1)

def rotate_points(points: torch.Tensor, center: Tuple[float, float], angle: float) -> torch.Tensor:
    #Rotate points around center by angle (in degrees)
    angle_rad = math.radians(angle)
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    
    cx, cy = center
    points = points.clone()
    
    # Translate to origin
    points[..., 0] -= cx
    points[..., 1] -= cy
    
    x_prime = points[..., 0] * cos_theta - points[..., 1] * sin_theta #rotation matrix multiplication 
    y_prime = points[..., 0] * sin_theta + points[..., 1] * cos_theta
    
    # Translate back
    points[..., 0] = x_prime + cx
    points[..., 1] = y_prime + cy
    
    return points