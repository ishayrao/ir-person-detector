import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from omegaconf import DictConfig
import logging
log = logging.getLogger(__name__)
#resizes to 800x1333 (will resize any image to this size)
class FasterRCNNDetector(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(FasterRCNNDetector, self).__init__()
        
        # Load pretrained model withOUT default weights
        weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        self.model = fasterrcnn_mobilenet_v3_large_fpn(
            weights=None,
            box_nms_thresh=0.5,     # NMS IoU threshold
            box_detections_per_img=10  # Max detections per image
        )
        
        # Modify classifier for single class detection
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = (torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
            in_features, cfg.model.num_classes + 1))
        
        self.transforms = weights.transforms()
    
    def forward(self, data, targets=None):
        outputs = self.model(data, targets)
        return outputs
