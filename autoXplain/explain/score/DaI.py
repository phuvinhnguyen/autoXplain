import numpy as np
import torch
from sklearn.metrics import auc
from typing import Dict, Any

from autoXplain.explain.score.base import score, BaseScoreExplainer


@score
class DaI(BaseScoreExplainer):
    """Deletion and Insertion AUC score (classification only)."""

    def __init__(self, model, saliency_config=None, steps: int = 30, labels=None, **kwargs):
        super().__init__(model, saliency_config, **kwargs)
        self.steps = steps
        self.labels = labels or []

    def explain(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        sal = self.get_saliency(inputs)
        results = []

        for i in range(len(sal['img_tensors'])):
            img_t = sal['img_tensors'][i].to(self.device)
            cam = sal['cam_arrays'][i]
            pred = sal['prediction'][i]
            class_idx = pred['class_idx']
            pred['predicted_label'] = self.labels[class_idx]

            flat = cam.flatten()
            order = np.argsort(-flat)
            h, w = cam.shape
            total = h * w
            x = np.linspace(0, 1, self.steps + 1)

            ins_scores, del_scores = [], []
            blank = torch.zeros_like(img_t)

            for step in range(self.steps + 1):
                k = int(total * step / self.steps)
                mask = torch.zeros(total)
                if k > 0:
                    mask[order[:k]] = 1
                mask = mask.reshape(h, w).unsqueeze(0).repeat(3, 1, 1).to(self.device)

                with torch.no_grad():
                    ins = torch.softmax(self.model(img_t * mask + blank * (1 - mask)), 1)[0, class_idx].item()
                    dele = torch.softmax(self.model(img_t * (1 - mask)), 1)[0, class_idx].item()
                ins_scores.append(ins)
                del_scores.append(dele)

            ins_auc = auc(x, ins_scores)
            del_auc = auc(x, del_scores)
            results.append({
                'id': str(inputs['image_paths'][i]).split('/')[-1].split('.')[0],
                'score': (ins_auc + (1 - del_auc)) / 2,
                'insertion_auc': ins_auc,
                'deletion_auc': del_auc,
                'prediction': pred,
                'saliency_image': sal['saliency_images'][i],
                'original_image': inputs['image_paths'][i],
            })
        
        # convert list of dict to dict of list and return
        return {k: [result[k] for result in results] for k in results[0].keys()}
