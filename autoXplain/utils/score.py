import re
import tempfile
from functools import partial
from typing import List, Optional, Tuple
import pathlib
import numpy as np
import torch
from torch import Tensor, nn
from PIL import Image
from torchvision.transforms.functional import resize, normalize, to_tensor
from torchcam.utils import overlay_mask
from torchcam.methods import CAM as CAMMethod

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
#  Model wrappers — make segmentation / detection models CAM-compatible
# ---------------------------------------------------------------------------

class SegmentationWrapper(nn.Module):
    """Wraps a segmentation model to return (class_scores, seg_map)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        seg_map = out['out'] if isinstance(out, dict) else out
        class_scores = seg_map.mean(dim=[2, 3])
        return class_scores, seg_map


class DetectionWrapper(nn.Module):
    """Wraps a detection model's backbone for CAM extraction.

    Only the backbone is registered as a submodule so torchcam hooks into it.
    The full detection model is kept as a plain attribute to avoid torchcam
    hooking into non-differentiable detection layers.
    """
    def __init__(self, det_model):
        super().__init__()
        self.backbone = det_model.backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self._det_model = [det_model]
        self.detections = None

    def forward(self, x):
        features = self.backbone(x)
        feats = list(features.values())
        combined = sum(
            torch.nn.functional.adaptive_avg_pool2d(f, feats[0].shape[2:])
            for f in feats
        )
        return self.pool(combined).flatten(1)

    def run_detection(self, img_tensor):
        det = self._det_model[0]
        det.eval()
        with torch.no_grad():
            self.detections = det([xi for xi in img_tensor])


# ---------------------------------------------------------------------------
#  Small utilities
# ---------------------------------------------------------------------------

def load_image(image) -> Image.Image:
    if isinstance(image, str) or isinstance(image, pathlib.Path):
        return Image.open(image).convert('RGB')
    return image.convert('RGB')


def preprocess(img: Image.Image, size=(224, 224)) -> Tensor:
    return normalize(to_tensor(resize(img, size)), IMAGENET_MEAN, IMAGENET_STD).unsqueeze(0)


def pil_to_tempfile(pil_img, suffix=".png") -> str:
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    pil_img.save(f.name)
    return f.name


def sigmoid_mask(x, slope=25, position=0.4):
    return 1.0 / (1.0 + np.exp(slope * (position - x)))


def locate_candidate_layer(mod: nn.Module, input_shape=(3, 224, 224), index=None) -> List[str]:
    mod.eval()
    shapes: List[Tuple[Optional[str], Tuple[int, ...]]] = []

    def _hook(module, inp, out, name=None):
        if isinstance(out, Tensor) and 'aux' not in (name or ''):
            shapes.append((name, out.shape))

    handles = [m.register_forward_hook(partial(_hook, name=n)) for n, m in mod.named_modules()]
    with torch.no_grad():
        mod(torch.zeros((1, *input_shape), device=next(mod.parameters()).device))
    for h in handles:
        h.remove()

    candidates = []
    for name, shape in reversed(shapes):
        if len(shape) == len(input_shape) + 1 and any(v != 1 for v in shape[2:]):
            candidates.append(name)

    if index is not None and len(candidates) > index:
        return [candidates[index]]
    return candidates[::-1]


def locate_linear_layer(mod: nn.Module, index=None) -> List[str]:
    layers = [n for n, m in mod.named_modules() if isinstance(m, nn.Linear)]
    if index is not None:
        return [layers[index]]
    return layers[::-1]


# ---------------------------------------------------------------------------
#  Prediction-info builders (one per model type)
# ---------------------------------------------------------------------------

def _pred_classification(output, class_id):
    probs = torch.softmax(output, dim=1)[0]
    idx = class_id if class_id is not None else output.argmax().item()
    return {
        'class_idx': idx,
        'confidence': probs[idx].item(),
        '_cam_output': output,
        'top_predictions': probs.topk(min(20, len(probs))).indices.tolist()
    }


def _pred_segmentation(output, class_id, orig_size, img):
    scores, seg_map = output
    idx = class_id if class_id is not None else scores.argmax().item()
    probs = torch.softmax(scores, dim=1)[0]

    pred_mask = seg_map[0].argmax(0).cpu().numpy().astype(np.uint8)
    binary = (pred_mask == idx)
    seg_out = Image.fromarray((binary * 255).astype(np.uint8)).convert("L")
    resized = np.array(Image.fromarray(binary.astype(np.uint8)).resize(orig_size, Image.NEAREST))
    gray = np.array(img.convert('L').convert('RGB'))
    overlay = Image.fromarray(
        np.where(resized[..., None] == 1, np.array(img), gray).astype(np.uint8)
    )
    return {
        'class_idx': idx,
        'confidence': probs[idx].item(),
        'segment_mask': seg_out,
        'segment_overlay': overlay,
        '_cam_output': scores,
        'top_predictions': probs.topk(min(20, len(probs))).indices.tolist()
    }


def _pred_detection(wrapper, output, class_id):
    dets = wrapper.detections[0]
    boxes, labels, scores = dets['boxes'], dets['labels'], dets['scores']

    if len(boxes) == 0:
        return {
            'class_idx': class_id if class_id is not None else 0,
            'confidence': 0.0,
            'boxes': [],
            '_cam_output': output,
        }

    if class_id is not None:
        mask = labels == class_id
        if mask.any():
            best = scores[mask].argmax()
            pick = mask.nonzero(as_tuple=True)[0][best].item()
        else:
            pick = scores.argmax().item()
    else:
        pick = scores.argmax().item()

    all_boxes = [
        {'box': boxes[i].tolist(), 'label': labels[i].item(), 'score': scores[i].item()}
        for i in range(len(boxes))
    ]
    return {
        'class_idx': labels[pick].item(),
        'confidence': scores[pick].item(),
        'box': boxes[pick].tolist(),
        'boxes': all_boxes,
        '_cam_output': output,
        'top_predictions': scores.topk(min(20, len(scores))).indices.tolist()
    }


# ---------------------------------------------------------------------------
#  Main CAM generation function
# ---------------------------------------------------------------------------

INPUT_SIZES = {
    'classification': (224, 224),
    'segmentation': 520,
    'detection': (224, 224),
}


def generate_cam(image, model, cam_class, layer=0, class_id=None,
                 slope=25, position=0.4, model_type='classification'):
    """Generate CAM visualizations for classification, segmentation, or detection.

    Returns a dict with keys:
        cam_image, masked_image, heatmap, original_image,
        pred_info, img_tensor, output, cam_array
    """
    img = load_image(image)
    orig_size = img.size

    if model_type == 'segmentation':
        wrapped = SegmentationWrapper(model)
    elif model_type == 'detection':
        wrapped = DetectionWrapper(model)
    else:
        wrapped = model

    input_size = INPUT_SIZES.get(model_type, (224, 224))
    img_tensor = preprocess(img, input_size).to(next(wrapped.parameters()).device)
    img_tensor.requires_grad_(True)

    if model_type == 'detection':
        wrapped.run_detection(img_tensor.detach())

    # Old code:
    # target_layer = (locate_linear_layer(wrapped, index=layer)[0] if cam_class == CAMMethod
    #                 else locate_candidate_layer(wrapped, index=layer)[0])
    # Use feature map layers for CAM extraction. Using a linear layer here can
    # cause channel-size mismatches (e.g., 512 vs 1000) for CAMMethod.
    target_layer = locate_candidate_layer(wrapped, index=layer)[0]
    extractor = cam_class(wrapped, target_layer)

    output = wrapped(img_tensor)

    if model_type == 'segmentation':
        pred_info = _pred_segmentation(output, class_id, orig_size, img)
    elif model_type == 'detection':
        pred_info = _pred_detection(wrapped, output, class_id)
    else:
        pred_info = _pred_classification(output, class_id)

    cam_output = pred_info.pop('_cam_output')
    class_idx = pred_info['class_idx']
    activation = extractor(class_idx, cam_output)[0].squeeze().cpu().numpy()
    extractor.remove_hooks()

    activation = (activation - activation.min()) / (activation.max() - activation.min() + 1e-8)
    heatmap = Image.fromarray(activation.astype(np.float32), mode='F').resize(orig_size, Image.BICUBIC)
    cam_image = overlay_mask(img, heatmap, alpha=0.5)
    mask_arr = sigmoid_mask(np.array(heatmap), slope, position)
    masked = Image.fromarray((np.array(img) * mask_arr[..., np.newaxis]).astype(np.uint8))

    t_h, t_w = img_tensor.shape[2], img_tensor.shape[3]
    cam_array = np.array(
        Image.fromarray(activation.astype(np.float32), mode='F').resize((t_w, t_h), Image.BICUBIC)
    )

    return {
        'cam_image': cam_image,
        'masked_image': masked,
        'heatmap': heatmap,
        'original_image': img,
        'pred_info': pred_info,
        'img_tensor': img_tensor.detach(),
        'output': cam_output.detach(),
        'cam_array': cam_array,
    }


# ---------------------------------------------------------------------------
#  Text parsing helpers (for VLMJudge)
# ---------------------------------------------------------------------------

def get_first_number(text: str):
    """Extract the most likely score digit (0-9) from text."""
    if not text:
        return None
    m = re.search(r'^\s*(\d)\s*$', text.strip(), re.MULTILINE)
    if m:
        return int(m.group(1))
    all_digits = re.findall(r'\d', text)
    return int(all_digits[-1]) if all_digits else None


def extract_function_calls(text):
    funcs = []
    for match in re.finditer(r'<function>(.*?)</function>', text, re.DOTALL):
        params = {}
        for p in re.finditer(r'<parameter\s+(\w+)>(.*?)</parameter>', match.group(1), re.DOTALL):
            params[p.group(1)] = p.group(2).strip()
        funcs.append(params)
    return funcs
