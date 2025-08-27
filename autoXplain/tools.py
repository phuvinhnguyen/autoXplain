# Copyright (C) 2020-2023, François-Guillaume Fernandez.

# This program is licensed under the Apache License 2.0.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0> for full license details.

from functools import partial
from typing import List, Optional, Tuple, Union
import torch
from torch import Tensor, nn
import numpy as np
from PIL import Image
from torchvision.transforms.functional import resize, normalize, to_tensor
from torchcam.utils import overlay_mask
from torchcam.methods import *
import tempfile, re
from collections import OrderedDict

class SegmentationWrapper(torch.nn.Module):
    def __init__(self, segmentation_model):
        super().__init__()
        self.segmentation_model = segmentation_model
        
    def forward(self, x):
        out = self.segmentation_model(x)
        # Extract the segmentation output
        if not isinstance(out, torch.Tensor):
            seg_out = out['out']
        else:
            seg_out = out
        
        # Convert 4D segmentation output to 2D class scores
        # Average pool across spatial dimensions to get per-class scores
        class_scores = seg_out.mean(dim=[2, 3])  # Shape: [batch, num_classes]
        return class_scores, seg_out

def get_first_number(text: str) -> int:
    match = re.search(r"\d", text)
    return int(match.group()) if match else None

def extract_function_calls(text):
    # Regular expression to capture the content inside <function> tags
    pattern = re.compile(r'<function>(.*?)</function>', re.DOTALL)
    
    # Regular expression to capture the <parameter> tags and their contents
    param_pattern = re.compile(r'<parameter\s*(\w+)>((.|\n)*?)</parameter>', re.DOTALL)
    
    functions = []
    
    for match in pattern.finditer(text):
        func_content = match.group(1)
        
        # Extract parameters (description, justification, and score) inside the <function> content
        params = {}
        for param_match in param_pattern.finditer(func_content):
            param_name = param_match.group(1)  # The parameter name (e.g., description, justification, score)
            param_value = param_match.group(2).strip()  # The parameter value (the actual text content)
            params[param_name] = param_value
        
        # Add the extracted function and its parameters to the list of functions
        functions.append({'name': 'function', 'args': params})
    
    return functions

def pil_to_tempfile_path(pil_img, suffix=".png"):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    pil_img.save(temp_file.name)
    return temp_file.name

def locate_candidate_layer(mod: nn.Module, input_shape: Tuple[int, ...] = (3, 224, 224), index=None) -> Optional[List[str]]:
    """Attempts to find a candidate layer to use for CAM extraction
    Args:
        mod: the module to inspect
        input_shape: the expected shape of input tensor excluding the batch dimension
        index: optional index to select specific candidate layer
    Returns:
        List[str]: the candidate layers for CAM
    """
    # Set module in eval mode
    module_mode = mod.training
    mod.eval()
    
    output_shapes: List[Tuple[Optional[str], Tuple[int, ...]]] = []
    
    def _record_output_shape(module: nn.Module, input: Tensor, output: Union[Tensor, OrderedDict], name: Optional[str] = None) -> None:
        """Activation hook."""
        if isinstance(output, Tensor) and 'aux' not in name:
            # Handle regular tensor outputs
            output_shapes.append((name, output.shape))
    
    hook_handles: List[torch.utils.hooks.RemovableHandle] = []
    
    # Forward hook on all layers
    for n, m in mod.named_modules():
        hook_handles.append(m.register_forward_hook(partial(_record_output_shape, name=n)))
    
    # Forward empty tensor
    with torch.no_grad():
        _ = mod(torch.zeros((1, *input_shape), device=next(mod.parameters()).data.device))
    

    # Remove all temporary hooks
    for handle in hook_handles:
        handle.remove()
    
    # Put back the model in the corresponding mode
    mod.training = module_mode
    
    # Check output shapes
    candidate_layer = []
    for layer_name, output_shape in reversed(output_shapes):
        # Stop before flattening or global pooling
        if len(output_shape) == (len(input_shape) + 1) and any(v != 1 for v in output_shape[2:]):
            candidate_layer.append(layer_name)
    
    if index is not None and len(candidate_layer) > index:
        candidate_layer = [candidate_layer[index]]
    
    return candidate_layer[::-1]

def locate_linear_layer(mod: nn.Module, index=None) -> Optional[str]:
    """Attempts to find a fully connecter layer to use for CAM extraction

    Args:
        mod: the module to inspect
        index: the index of the candidate layer (None means take all layers)

    Returns:
        str: the candidate layer
    """
    candidate_layer = []
    for layer_name, m in mod.named_modules():
        if isinstance(m, nn.Linear):
            candidate_layer.append(layer_name)

    if index is not None:
        candidate_layer = [candidate_layer[index]]
    return candidate_layer[::-1]

def acti(x, slope=25, position=0.4): return 1 / (1 + np.exp(slope*(position-x)))

# def generate_cam(
#         image,
#         model,
#         cam_extractor,
#         layer=0,
#         class_id=None,
#         slope=25,
#         position=0.4,
#         model_type='classification'):
#     # Load image
#     if isinstance(image, str):
#         img = Image.open(image).convert('RGB')
#     else:
#         img = image.convert('RGB')
#     orig_size = img.size

#     if model_type == 'segmentation':
#         model = SegmentationWrapper(model)

#     # Load cam
#     if cam_extractor == CAM:
#         cam_extractor = cam_extractor(model, locate_linear_layer(model, index=layer)[0])
#     else:
#         cam_extractor = cam_extractor(model, locate_candidate_layer(model, index=layer)[0])

#     img_tensor = normalize(
#             to_tensor(resize(img, 520)), 
#             [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
#         ).unsqueeze(0)

#     # Move to model's device
#     device = next(model.parameters()).device
#     img_tensor = img_tensor.to(device)
    
#     # Forward pass
#     output = model(img_tensor)
#     if model_type == 'segmentation':
#         output, seg_out = output
#         class_idx = class_id if class_id is not None else output.argmax().item()
        
#         pred_mask = seg_out[0].argmax(0).cpu().detach().numpy()
#         binary_mask = (pred_mask == class_idx).astype(np.uint8)
        
#         seg_out = Image.fromarray(binary_mask * 255).convert("L")
#     else:
#         class_idx = class_id if class_id is not None else output.argmax().item()
#         seg_out = None
    
#     # Generate CAM
#     activation_map = cam_extractor(class_idx, output)[0].squeeze().cpu().numpy()
#     cam_extractor.remove_hooks()
    
#     # Normalize and resize activation map
#     activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8)
#     heatmap = Image.fromarray(activation_map.astype(np.float32), mode='F').resize(orig_size, Image.BICUBIC)
    
#     # Create overlay
#     cam_image = overlay_mask(img, heatmap, alpha=0.5)
    
#     # Create masked image
#     mask = acti(np.array(heatmap), slope, position)
#     masked = Image.fromarray((np.array(img) * mask[..., np.newaxis]).astype(np.uint8))
    
#     if model_type == 'segmentation':
#         return cam_image, masked, {'class_idx': class_idx, 'segment_mask': seg_out}, heatmap
#     else:
#         return cam_image, masked, {'class_idx': class_idx}, heatmap
    
def generate_cam(
        image,
        model,
        cam_extractor,
        layer=0,
        class_id=None,
        slope=25,
        position=0.4,
        model_type='classification'):
    # Load image
    if isinstance(image, str):
        img = Image.open(image).convert('RGB')
    else:
        img = image.convert('RGB')
    orig_size = img.size

    if model_type == 'segmentation':
        model = SegmentationWrapper(model)

    # Load cam
    if cam_extractor == CAM:
        cam_extractor = cam_extractor(model, locate_linear_layer(model, index=layer)[0])
    else:
        cam_extractor = cam_extractor(model, locate_candidate_layer(model, index=layer)[0])

    if model_type == 'segmentation':
        img_tensor = normalize(
                to_tensor(resize(img, 520)), 
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ).unsqueeze(0)
    else:
        img_tensor = normalize(
                to_tensor(resize(img, (224, 224))), 
                [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            ).unsqueeze(0)

    # Move to model's device
    device = next(model.parameters()).device
    img_tensor = img_tensor.to(device)
    
    # Forward pass
    output = model(img_tensor)
    if model_type == 'segmentation':
        output, seg_out = output
        class_idx = class_id if class_id is not None else output.argmax().item()
        
        pred_mask = seg_out[0].argmax(0).cpu().detach().numpy()
        binary_mask = (pred_mask == class_idx).astype(np.uint8)
        
        # Create segmentation mask at model output resolution
        seg_out = Image.fromarray(binary_mask * 255).convert("L")
        
        # Create segment_overlay (color where mask=1, gray elsewhere)
        # Resize binary mask to original image size
        resized_mask = np.array(Image.fromarray(binary_mask).resize(orig_size, Image.NEAREST))
        gray_img = img.convert('L').convert('RGB')
        img_arr = np.array(img)
        gray_arr = np.array(gray_img)
        # Create overlay using mask
        segment_overlay = Image.fromarray(
            np.where(resized_mask[..., None] == 1, img_arr, gray_arr).astype(np.uint8)
        )
    else:
        class_idx = class_id if class_id is not None else output.argmax().item()
        seg_out = None
    
    # Generate CAM
    activation_map = cam_extractor(class_idx, output)[0].squeeze().cpu().numpy()
    cam_extractor.remove_hooks()
    
    # Normalize and resize activation map
    activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min() + 1e-8)
    heatmap = Image.fromarray(activation_map.astype(np.float32), mode='F').resize(orig_size, Image.BICUBIC)
    
    # Create overlay
    cam_image = overlay_mask(img, heatmap, alpha=0.5)
    
    # Create masked image
    mask = acti(np.array(heatmap), slope, position)
    masked = Image.fromarray((np.array(img) * mask[..., np.newaxis]).astype(np.uint8))
    
    if model_type == 'segmentation':
        return cam_image, masked, {'class_idx': class_idx, 'segment_mask': seg_out, 'segment_overlay': segment_overlay}, heatmap
    else:
        return cam_image, masked, {'class_idx': class_idx}, heatmap