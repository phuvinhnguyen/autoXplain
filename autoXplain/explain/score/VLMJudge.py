import re
from typing import Dict, Any

from autoXplain.explain.score.base import score, BaseScoreExplainer
from autoXplain.utils.score import pil_to_tempfile, get_first_number
from autoXplain.utils.vlm import VLM_REGISTRY

MASKED_CAM_PROMPT = '''\
Read the masked CAM image and use this tool to evaluate the model's attention mechanism. This task assesses how well the model's attention aligns with its own prediction.

A masked Grad-CAM image shows only the regions the model focused on when making its prediction; all other areas are blacked out. The model predicted "{object}" based on these visible regions.

Your task: Determine whether the visible regions actually contain evidence for "{object}".

<function>
<parameter description>
- Describe what is actually visible in the masked image (objects, shapes, colors, textures).
- Is "{object}" present in the visible regions? (not at all / partially / fully)
- Which parts of "{object}" can be seen, if any?
- If "{object}" is not visible: what did the model focus on instead?
- Are there any distracting or irrelevant regions in the visible areas?
</parameter>
<parameter justification>
Explain whether the visible content supports the prediction of "{object}". 
If the model attended to regions unrelated to "{object}", explain what 
captured its attention and why this represents strong/weak alignment.
</parameter>
<parameter score>
Give a single number from 0 to 5 based on attention-prediction alignment:

5: Visible regions clearly and fully depict "{object}" — perfect alignment
4: Visible regions mostly show "{object}" with minor gaps or distractions  
3: Visible regions partially show "{object}" mixed with other content
2: Visible regions barely show "{object}" — mostly focused elsewhere
1: Visible regions show no "{object}" at all — completely misaligned
0: Visible regions are scattered, empty, or incoherent — no meaningful focus
</parameter>
</function>

<IMPORTANT>
- Evaluate based on what is ACTUALLY visible, not what should be there.
- The model uses this masked image as "proof" of "{object}" — judge if this proof is valid.
- Function calls MUST follow the specified format, start with <function and end with </function>.
- Required parameters MUST follow the specified format.
- You can only call one function each turn.
- Provide reasoning in natural language BEFORE the function call, but NOT after.
'''

ORIGINAL_CAM_PROMPT = '''\
Analyze the provided CAM heatmap to evaluate the model's attention mechanism. This assessment tests whether the model's visual attention aligns with its prediction.

## Input Description
- **Source**: ImageNet 1K dataset (standard research benchmark)
- **Visualization**: Warm colors (orange, red, yellow) = high attention; cool colors (blue, purple, black) = low/no attention
- **Model's claim**: The model predicts "{object}" based on this attention pattern

## Your Task
Evaluate whether the warm-colored regions (high attention) actually correspond to "{object}" in the original image. Determine if the model's attention supports its own prediction.

<function>
<parameter description>
- What regions show warm colors (high attention)? Describe their location, shape, and extent.
- Is "{object}" located within these warm-colored regions? (not at all / partially / fully)
- Which parts of "{object}" are covered by warm colors, if any?
- Are there warm-colored regions that do NOT correspond to "{object}"? If so, what are they?
- Are cool-colored regions (low attention) appropriately placed on background/irrelevant areas?
- What visual features might explain the model's attention pattern (e.g., distinctive textures, confusing backgrounds, partial occlusion, similar-colored objects)?
</parameter>
<parameter justification>
Explain whether the warm-colored regions provide valid evidence for "{object}". 
Assess if the model's attention pattern justifies its prediction, or if it focused 
on misleading features. Justify why this level of attention-prediction alignment 
warrants your assigned score.
</parameter>
<parameter score>
Rate attention-prediction alignment from 0 to 5:

5: Warm colors fully cover "{object}" with minimal background attention — perfect alignment
4: Warm colors mostly cover "{object}" with minor spillover to nearby regions  
3: Warm colors partially cover "{object}" mixed with significant irrelevant areas
2: Warm colors barely touch "{object}" — mostly focused on wrong regions
1: Warm colors completely miss "{object}" — attention on unrelated objects/background
0: Warm colors scattered randomly or absent — no coherent attention pattern
</parameter>
</function>

<IMPORTANT>
- Base your evaluation SOLELY on what the heatmap reveals about attention placement.
- The model claims this heatmap proves "{object}" is present — assess if this claim is valid.
- Function calls MUST start with <function and end with </function>.
- Required parameters MUST use <parameter name> and </parameter> tags.
- One function call per turn only.
- Provide reasoning BEFORE the function call, NOT after.
'''

def _parse_vlm_response(text):
    desc_match = re.search(r'<parameter description>(.*?)</parameter>', text, re.DOTALL)
    description = desc_match.group(1).strip() if desc_match else None

    just_match = re.search(r'<parameter justification>(.*?)</parameter>', text, re.DOTALL)
    justification = just_match.group(1).strip() if just_match else None

    score_match = re.search(r'<parameter score>(.*?)</parameter>', text, re.DOTALL)
    score_val = get_first_number(score_match.group(1).strip()) if score_match else None

    if score_val is None:
        m = re.search(r'(?:score|rating|rate)[:\s]*(\d)', text, re.IGNORECASE)
        if m:
            score_val = int(m.group(1))
        else:
            score_val = get_first_number(text)

    return {
        'description': description,
        'justification': justification,
        'score': score_val,
    }

@score
class VLMJudge(BaseScoreExplainer):
    """VLM-based CAM quality judge.

    Uses a Vision-Language Model to evaluate whether the model's
    attention (CAM) aligns with its prediction.
    """

    def __init__(self, model, saliency_config=None, vlm=None,
                 cam_mode='masked', threshold=2.5, labels=None, **kwargs):
        super().__init__(model, saliency_config, **kwargs)
        self.vlm = VLM_REGISTRY[vlm['name']](**(vlm.get('kwargs') or {}))
        self.cam_mode = cam_mode
        self.threshold = threshold
        self.labels = labels or []
        self.prompt_template = MASKED_CAM_PROMPT if cam_mode == 'masked' else ORIGINAL_CAM_PROMPT

    def explain(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        sal = self.get_saliency(inputs)
        queries = []
        meta = []

        for i, path in enumerate(inputs['image_paths']):
            pred = sal['prediction'][i]
            class_idx = pred['class_idx']
            label = self.labels[class_idx] if self.labels else str(class_idx)
            pred['label_name'] = label

            send_img = sal['masked_images'][i] if self.cam_mode == 'masked' else sal['saliency_images'][i]
            img_path = pil_to_tempfile(send_img)
            prompt = self.prompt_template.format(object=label)
            queries.append({'image_path': img_path, 'text': prompt})
            meta.append({
                'pred': pred,
                'label': str(path).split('/')[-1].split('_')[-1].split('.')[0],
                'saliency_image': sal['saliency_images'][i],
                'masked_image': sal['masked_images'][i],
                'top_labels': sal['prediction'][i]['top_labels'],
                'predicted_label': sal['prediction'][i]['predicted_label'],
            })

        responses = self.vlm.query_batch(queries, max_tokens=1024)

        results = []
        for i, raw in enumerate(responses):
            parsed = _parse_vlm_response(raw)
            s = parsed['score']
            results.append({
                'id': str(inputs['image_paths'][i]).split('/')[-1].split('.')[0],
                'score': s,
                'description': parsed['description'],
                'justification': parsed['justification'],
                'above_threshold': s >= self.threshold if s is not None else None,
                'threshold': self.threshold,
                'prediction': meta[i]['pred'],
                'label': meta[i]['label'],
                'saliency_image': meta[i]['saliency_image'],
                'masked_image': meta[i]['masked_image'],
                'raw_response': raw,
            })
        
        # convert list of dict to dict of list and return
        return {k: [result[k] for result in results] for k in results[0].keys()}
