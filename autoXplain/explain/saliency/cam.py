from typing import Dict, Any, List

from torchcam.methods import (
    GradCAM, SmoothGradCAMpp, GradCAMpp,
    CAM as CAMMethod, ScoreCAM, LayerCAM, XGradCAM, ISCAM, SSCAM,
)

from autoXplain.explain.base import BaseExplainer
from autoXplain.explain.saliency.base import saliency
from autoXplain.utils.score import generate_cam

CAM_METHODS = {
    'gradcam': GradCAM,
    'smoothgradcampp': SmoothGradCAMpp,
    'gradcampp': GradCAMpp,
    'cam': CAMMethod,
    'scorecam': ScoreCAM,
    'layercam': LayerCAM,
    'xgradcam': XGradCAM,
    'iscam': ISCAM,
    'sscam': SSCAM,
}


@saliency
class CAMSaliency(BaseExplainer):
    """Batch-capable CAM saliency explainer.

    Args:
        model: torch.nn.Module to explain.
        cam: name of the CAM method (e.g. ``'GradCAM'``).
        layer: target layer index for CAM extraction.
        model_type: ``'classification'``, ``'segmentation'``, or ``'detection'``.
        slope / position: sigmoid mask parameters.

    Input dict:
        ``{'image_paths': [str, ...]}``

    Output dict keys:
        saliency_images, masked_images, heatmaps, original_images,
        prediction, img_tensors, outputs, cam_arrays
    """

    def __init__(self, model, cam: str = 'GradCAM', layer: int = 0,
                 model_type: str = 'classification', labels: List[str] = None,
                 slope: float = 25, position: float = 0.4, **kwargs):
        super().__init__(model)
        key = cam.lower()
        if key not in CAM_METHODS:
            raise ValueError(f"Unknown CAM method: {cam}. Available: {list(CAM_METHODS.keys())}")
        self.cam_class = CAM_METHODS[key]
        self.layer = layer
        self.model_type = model_type
        self.slope = slope
        self.position = position
        self.labels = labels

    def explain(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        out = {
            'id': [],
            'saliency_images': [], 'masked_images': [], 'heatmaps': [],
            'original_images': [], 'prediction': [],
            'img_tensors': [], 'outputs': [], 'cam_arrays': [],
        }
        for path in inputs['image_paths']:
            r = generate_cam(
                path, self.model, self.cam_class, self.layer,
                model_type=self.model_type,
                slope=self.slope, position=self.position,
            )
            r['id'] = str(path).split('/')[-1].split('.')[0]
            out['id'].append(r['id'])
            out['saliency_images'].append(r['cam_image'])
            out['masked_images'].append(r['masked_image'])
            out['heatmaps'].append(r['heatmap'])
            out['original_images'].append(r['original_image'])
            out['prediction'].append(r['pred_info'])
            out['img_tensors'].append(r['img_tensor'])
            out['outputs'].append(r['output'])
            out['cam_arrays'].append(r['cam_array'])
            if self.labels:
                out['prediction'][-1]['top_labels'] = [self.labels[i] for i in r['pred_info']['top_predictions'][:20]]
                out['prediction'][-1]['predicted_label'] = self.labels[r['pred_info']['class_idx']]
        return out
