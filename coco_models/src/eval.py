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
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
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
    results = []
    total_predictions = 0
    total_boxes = 0
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(tqdm(data_loader)):
            data = data.to(device)
            predictions = model(data)
            #log.info(f"Scores before filtering: {predictions[0]['scores']}")
            #log.info(f"Boxes before filtering: {predictions[0]['boxes'].shape}")
                        
            for pred, target in zip(predictions, targets):
                image_id = target['image_id'].item()
                boxes = pred['boxes']
                scores = pred['scores']
                total_predictions += 1
                total_boxes += len(boxes)
    
                if len(boxes) > 0:
                    # convert from [x1,y1,x2,y2] to COCO format [x,y,w,h]
                    boxes_coco = torch.zeros_like(boxes)
                    boxes_coco[:, 0] = boxes[:, 0]  # x
                    boxes_coco[:, 1] = boxes[:, 1]  # y
                    boxes_coco[:, 2] = boxes[:, 2] - boxes[:, 0]  # w
                    boxes_coco[:, 3] = boxes[:, 3] - boxes[:, 1]  # h
                    
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
                    
           # if batch_idx == 0:  # Log detailed info for first batch
                    #log.info(f"First batch complete. Total boxes found: {total_boxes}")
                    #log.info(f"Results so far: {results}")
   
    #log.info(f"Evaluation complete. Total predictions: {total_predictions}")
   # log.info(f"Total boxes detected: {total_boxes}")
    return results

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    base_dir = Path(get_original_cwd())
    
    device = cfg.model.device
    
    #Create model
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

    #load best model 
    #checkpoint_path = Path(cfg.logging.save_dir) / 'best_model.pth'
    checkpoint_path = "/root/ir-person-detector/multirun/2025-06-26/14-41-38/0/best_model.pth"
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    log.info(f"Loaded checkpoint frm experiment: {checkpoint_path}")


    #Load pretrained model as test
    #weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
    #model = fasterrcnn_mobilenet_v3_large_fpn(weights=weights)

    #model.to(device)
    model.eval()
    # Add model verification
    log.info(f"Model device: {next(model.parameters()).device}")
    log.info(f"Model training mode: {model.training}")

    test_transform = build_transforms(cfg, is_train=False, test=True)

    # Create test dataset and dataloader
    test_dataset = FLIRDataset(
        json_file= base_dir / cfg.dataset.data.test_annotations,
        thermal_dir= base_dir / cfg.dataset.data.test_images,
        transform=test_transform
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers= cfg.training.num_workers,
        pin_memory=cfg.training.pin_memory,
        collate_fn=custom_collate_fn
    )

    # Eval
    results = evaluate_model(model, test_loader, device, cfg)

    # saving predictions
    output_dir = Path(cfg.logging.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_file = output_dir / f"{cfg.model.name}_predictions.json"
    with open(predictions_file, "w") as f:
        json.dump(results, f)

    # COCO metrics
    gt_path = base_dir / cfg.dataset.data.test_annotations
    # Load and check contents
    with open(predictions_file, 'r') as f:
        pred_data = json.load(f)
    log.info(f"no of predictions: {len(pred_data)}")
    
    with open(gt_path, 'r') as f:
        gt_data = json.load(f)
    log.info(f"ground truth images: {len(gt_data['images'])}")
    log.info(f"ground truth annotations: {len(gt_data['annotations'])}")

    # Check if there are any matching image IDs
    pred_img_ids = set(p['image_id'] for p in pred_data)
    gt_img_ids = set(ann['image_id'] for ann in gt_data['annotations'])
    matching_ids = pred_img_ids.intersection(gt_img_ids)
    log.info(f"matching image IDs: {len(matching_ids)}")
    
    coco_gt = COCO(gt_path)
    coco_dt = coco_gt.loadRes(str(predictions_file))
    
    coco_eval = COCOeval(cocoGt=coco_gt, cocoDt=coco_dt)
    #coco_eval.params.catIds = [1]
    coco_eval.params.iouType = 'bbox'


    # compare with your predictions for image 0
    img0_preds = [p for p in results if p['image_id'] == 0]
    log.info(f"Predictions for image 0: {img0_preds}")
    coco_eval.evaluate()
    #log.info(f"evalImgs: {[x for x in coco_eval.evalImgs if x is not None]}")
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
