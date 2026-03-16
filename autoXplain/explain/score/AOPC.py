import numpy as np
import torch
from typing import Dict, Any

from autoXplain.explain.score.base import score, BaseScoreExplainer


@score
class AOPC(BaseScoreExplainer):
    """Area Over the Perturbation Curve score (classification only)."""

    def __init__(self, model, saliency_config=None, steps: int = 30, **kwargs):
        super().__init__(model, saliency_config, **kwargs)
        self.steps = steps

    def explain(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        sal = self.get_saliency(inputs)
        results = []

        for i in range(len(sal['img_tensors'])):
            img_t = sal['img_tensors'][i].to(self.device)
            cam = sal['cam_arrays'][i]
            pred = sal['predictions'][i]
            class_idx = pred['class_idx']

            with torch.no_grad():
                full_score = torch.softmax(self.model(img_t), dim=1)[0, class_idx].item()

            flat = cam.flatten()
            order = np.argsort(-flat)
            h, w = cam.shape
            total = h * w

            scores = []
            for step in range(self.steps + 1):
                keep = int(total * (self.steps - step) / self.steps)
                mask = torch.zeros(total)
                if keep > 0:
                    mask[order[:keep]] = 1
                mask = mask.reshape(h, w).unsqueeze(0).repeat(3, 1, 1).to(self.device)
                with torch.no_grad():
                    sk = torch.softmax(self.model(img_t * mask), dim=1)[0, class_idx].item()
                scores.append(sk)

            aopc = float(np.mean([full_score - s for s in scores]))
            results.append({
                'score': aopc,
                'prediction': pred,
                'saliency_image': sal['saliency_images'][i],
            })
        return {'results': results}
