from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm
import numpy as np
import json
from PIL import Image
import pickle
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import logging
from pathlib import Path
from hydra.utils import get_original_cwd, to_absolute_path
import torch
# Import model implementations and dataset
from models.custom_detector import ThermalDetector
from models.faster_rcnn_detector import FasterRCNNDetector
from models.effnet_detector import EfficientNetDetector
from models.ssdlite_detector import SSDLiteDetector
from datasets.flir_dataset import FLIRDataset
from utils.transforms import build_transforms, custom_collate_fn

log = logging.getLogger(__name__)

def evaluate_model(model, data_loader, device, cfg: DictConfig):
    """
    Evaluate model on test set and compute COCO metrics
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for (data, targets) in tqdm(data_loader):
            data = data.to(device)
            predictions = model(data)
            
            for pred, target in zip(predictions, targets):
                image_id = target['image_id'].item()
                boxes = pred['boxes']
                scores = pred['scores']
                
                if len(boxes) > 0:
                    # Convert to COCO format [x,y,w,h]
                    boxes_coco = torch.cat([
                        boxes[:, :2],
                        boxes[:, 2:] - boxes[:, :2]
                    ], dim=1)
                    
                    # Add all detections for this image
                    results.extend([
                        {
                            'image_id': image_id,
                            'category_id': 1,  # person
                            'bbox': box.tolist(),
                            'score': score.item()
                        }
                        for box, score in zip(boxes_coco, scores)
                    ])

    return results

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    base_dir = Path(get_original_cwd())
    
    device = cfg.model.device
    
    # Create model
    if cfg.model.name == "custom_detector":
        model = ThermalDetector(cfg).to(device)
    elif cfg.model.name == "faster_rcnn":
        model = FasterRCNNDetector(cfg).to(device)
    elif cfg.model.name == "effnet":
        model = EfficientNetDetector(cfg).to(device)
    elif cfg.model.name == "ssdlite":
        model = SSDLiteDetector(cfg).to(device)
    else:
        raise ValueError(f"Unknown model type: {cfg.model.name}")

    # load best model 
    #checkpoint_path = Path(cfg.logging.save_dir) / 'best_model.pth'
    checkpoint_path = "/root/ir-person-detector/multirun/2025-06-23/faster_rcnn_optimized/0/best_model.pth"
    #if checkpoint_path.exists():
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    log.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    #else:
    #    raise ValueError(f"No checkpoint found at {checkpoint_path}")

    # Set up transforms as in dataset
    #if cfg.model.name in ["ssdlite", "faster_rcnn"]:
   #     transform = model.transforms
   # else:
    transform = build_transforms(cfg, is_train=False)

    #log.info(f"test annotations: {Path(get_original_cwd()) / cfg.dataset.data.test_annotations}")
    #log.info(f"test images: {Path(get_original_cwd()) / cfg.dataset.data.test_images}")
    # Create test dataset and dataloader
    test_dataset = FLIRDataset(
        json_file= base_dir / cfg.dataset.data.test_annotations,
        thermal_dir= base_dir / cfg.dataset.data.test_images,
        transform=transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers= 0, #cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        collate_fn=custom_collate_fn
    )

    # Eval
    results = evaluate_model(model, test_loader, device, cfg)

    # predictions
    output_dir = Path(cfg.logging.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = output_dir / f"{cfg.model.name}_predictions.json"
    with open(predictions_file, "w") as f:
        json.dump(results, f)

    # COCO metrics
    coco_gt = COCO(cfg.dataset.data.test_annotations)
    coco_dt = coco_gt.loadRes(str(predictions_file))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    
    # Only evaluate person class (category_id = 1)
    coco_eval.params.catIds = [1]
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # Save metrics
    metrics = { #AP = average precision
        'AP': coco_eval.stats[0],  # AP at IoU=0.50:0.95
        'AP50': coco_eval.stats[1],  # AP at IoU=0.50
        'AP75': coco_eval.stats[2],  # AP at IoU=0.75
        'APs': coco_eval.stats[3],   # AP for small objects
        'APm': coco_eval.stats[4],   # AP for medium objects
        'APl': coco_eval.stats[5],   # AP for large objects
    }
    
    metrics_file = output_dir / f"{cfg.model.name}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)
    
    log.info(f"AP50: {metrics['AP50']:.3f}")

if __name__ == "__main__":
    main()
