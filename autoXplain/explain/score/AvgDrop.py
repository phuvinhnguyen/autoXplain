import torch
from typing import Dict, Any

from autoXplain.explain.score.base import score, BaseScoreExplainer


@score
class AverageDrop(BaseScoreExplainer):
    """Average Drop in confidence score (classification only)."""

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

            mask = torch.from_numpy(cam).unsqueeze(0).repeat(3, 1, 1).to(self.device)
            with torch.no_grad():
                masked_score = torch.softmax(self.model(img_t * mask), dim=1)[0, class_idx].item()

            drop = max(0.0, full_score - masked_score)
            avg_drop = 100.0 * drop / (full_score + 1e-8)
            results.append({
                'score': -avg_drop,
                'full_confidence': full_score,
                'masked_confidence': masked_score,
                'prediction': pred,
                'saliency_image': sal['saliency_images'][i],
            })
        return {'results': results}
