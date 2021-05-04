import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class DetectionModel():
    def __init__(self, num_classes: int):
        self.__num_classes = num_classes

    def get_detection_model(self):
        # load model
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        # replace the pretrained head with a new head
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.__num_classes)
        
        return model

    @property
    def num_classes(self):
        return self.__num_classes

    @num_classes.setter
    def num_classes(self, value: int):
        self.__num_classes = value
    

"""
# fasterrcnn_resnet50_fpn
(roi_heads): RoIHeads(
    (box_roi_pool): MultiScaleRoIAlign()
    (box_head): TwoMLPHead(
        (fc6): Linear(in_features=12544, out_features=1024, bias=True)
        (fc7): Linear(in_features=1024, out_features=1024, bias=True)
    )
    (box_predictor): FastRCNNPredictor(
        (cls_score): Linear(in_features=1024, out_features=91, bias=True)
        (bbox_pred): Linear(in_features=1024, out_features=364, bias=True)
    )
)
"""