from torchinfo import summary
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights

rcnn_weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
rcnn_model = fasterrcnn_mobilenet_v3_large_fpn(weights=None)

ssd_weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
ssd_model = ssdlite320_mobilenet_v3_large(weights=None)

batch_size = 64
summary(rcnn_model, input_size=(batch_size, 1, 1000, 800))
summary(ssd_model, input_size=(batch_size, 1, 320, 320))