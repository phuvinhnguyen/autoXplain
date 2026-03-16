import torch
from typing import Dict, Any

from autoXplain.explain.score.base import score, BaseScoreExplainer
from autoXplain.utils.score import load_image, preprocess


@score
class Uncertainty(BaseScoreExplainer):
    """Prediction confidence / uncertainty score.

    Works for all model types — uses the saliency method's prediction
    confidence plus optionally the true-label probability.
    """

    def __init__(self, model, saliency_config=None, labels=None, **kwargs):
        super().__init__(model, saliency_config, **kwargs)
        self.labels = labels or []

    def explain(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        sal = self.get_saliency(inputs)
        results = []

        for i, path in enumerate(inputs['image_paths']):
            pred = sal['predictions'][i]
            pred_prob = pred['confidence']

            label = path.split('/')[-1].split('_')[-1].split('.')[0]
            true_prob = None
            if label and self.labels and label in self.labels:
                output = sal['outputs'][i].unsqueeze(0) if sal['outputs'][i].dim() == 1 else sal['outputs'][i]
                probs = torch.softmax(output, dim=1)[0]
                true_prob = probs[self.labels.index(label)].item()

            results.append({
                'score': pred_prob,
                'pred_prob': pred_prob,
                'true_prob': true_prob,
                'prediction': pred,
                'saliency_image': sal['saliency_images'][i],
            })
        return {'results': results}
