from FlowDesign.processor import ThinkProcessor
from torchcam.methods import *
from .tools import *

class ExtractCAM(ThinkProcessor):
    modifies = ('saliency', 'maskedcam', 'prediction', 'heatmap')
    def __init__(self, cam_class, model, layer=0, slope=25, position=0.4, model_type='classification'):
        super().__init__()
        self.cam_class = cam_class
        self.model = model
        self.layer = layer
        self.slope = slope
        self.position = position
        self.model_type = model_type

    def process(self, image, target_class_idx=None):
        return generate_cam(image,
                            self.model,
                            self.cam_class,
                            self.layer,
                            slope=self.slope,
                            position=self.position,
                            class_id=target_class_idx,
                            model_type=self.model_type)
  