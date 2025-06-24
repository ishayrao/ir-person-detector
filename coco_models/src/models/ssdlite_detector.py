import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from torchvision.models.detection import _utils as det_utils
import hydra
from omegaconf import DictConfig
import logging
log = logging.getLogger(__name__)
#requires image of size 320x320

class SSDLiteDetector(nn.Module):
    def __init__(self, cfg: DictConfig):
        super(SSDLiteDetector, self).__init__()
        
        # Load pretrained model withOUT default weights
        weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
        self.model = ssdlite320_mobilenet_v3_large(weights=None)

        #CHECK THIS!! 
        # Modify first conv layer for 1-channel input
        # original_conv = self.model.backbone.features[0][0]
        # new_conv = nn.Conv2d(1, original_conv.out_channels,
        #                   kernel_size=original_conv.kernel_size,
        #                   stride=original_conv.stride,
        #                   padding=original_conv.padding,
        #                   bias=original_conv.bias is not None)
        
        # # average the weights across input channels --not sure if this is the best method/correct way to do this 
        # with torch.no_grad():
        #     new_conv.weight = nn.Parameter(original_conv.weight.sum(dim=1, keepdim=True))
        #     if original_conv.bias is not None:
        #         new_conv.bias = original_conv.bias
        
        # # Replace the first conv layer wtih new 1 channel conv layer 
        # self.model.backbone.features[0][0] = new_conv

        # Get number of anchors
        num_anchors = self.model.anchor_generator.num_anchors_per_location()
        size = (320, 320)
        in_channels = det_utils.retrieve_out_channels(self.model.backbone, size) #retrievng out channels from backbone --> last layer of backbone is first layer of classification head (320 is the expected size for ssdlite)
        # Create new classification head
        self.model.head.classification_head = torchvision.models.detection.ssd.SSDClassificationHead(
            in_channels=in_channels,
            num_anchors=num_anchors,
            num_classes=cfg.model.num_classes + 1
        )
        
        self.transforms = weights.transforms()

    def forward(self, data, targets=None):
        return self.model(data, targets)

    