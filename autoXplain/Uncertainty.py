from .xtractcam import *
from .tools import *
from torchcam import *
import torch
import numpy as np

class Uncertainty(ExtractCAM):
    modifies = ('pred_prob', 'true_prob', 'prediction', 'saliency')

    def __init__(self, cam_class, model, layer=0, labels=[], **kwargs):
        super().__init__(cam_class, model, layer, **kwargs)
        self.labels = labels
        self.model.eval()
        self.device = next(model.parameters()).device

    def process(self, image, true_label=None):
        # --- Step 1: Load & preprocess image ---
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")
        orig_size = img.size

        img_tensor = normalize(
            to_tensor(resize(img, (224, 224))),
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ).unsqueeze(0).to(self.device)

        # --- Step 2: Model inference ---
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)[0]
            
        class_idx = output.argmax().item()
        prediction = self.labels[class_idx] if self.labels else str(class_idx)
        pred_prob = probs[class_idx].item()
        
        # Get true label probability if provided
        if true_label is not None:
            if isinstance(true_label, str) and self.labels:
                # Convert string label to index
                true_idx = self.labels.index(true_label) if true_label in self.labels else None
            else:
                # Assume true_label is already an index
                true_idx = true_label if isinstance(true_label, int) else None
            
            true_prob = probs[true_idx].item() if true_idx is not None else None
        else:
            true_prob = None

        # --- Step 3: Generate CAM for visualization ---
        cam_extractor = self.cam_class(self.model, None)
        # Need to re-run forward pass for CAM extraction
        img_tensor.requires_grad_()
        output = self.model(img_tensor)
        cam = cam_extractor(class_idx, output)[0].squeeze().cpu().numpy()
        cam_extractor.remove_hooks()

        # Normalize CAM
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        # --- Step 4: Visualization ---
        heatmap = Image.fromarray(cam.astype(np.float32), mode='F').resize(orig_size, Image.BICUBIC)
        saliency = overlay_mask(img, heatmap, alpha=0.5)

        return pred_prob, true_prob, prediction, saliency
