from typing import Any, Dict, List, Optional
from PIL import Image

from autoXplain.explain.aggregation.base import BaseAggregationExplainer, aggregation
from autoXplain.utils.vlm import VLM_REGISTRY
import re
from typing import List, Dict, Callable

import nltk
from nltk.stem import WordNetLemmatizer

def parser(text: str) -> Optional[str]:
    match = re.search(r'<prediction>(.*?)</prediction>', text, re.DOTALL | re.IGNORECASE)
    if match: return match.group(1).strip()
    return 'None'

# ============================================
# 1. PREDICT FROM ORIGINAL IMAGE (chỉ có ảnh gốc)
# ============================================

def predict_from_original_image(is_prompt: bool = True) -> Callable:
    def prompt(original_image: str, labels: List[str], **kwargs) -> Dict:
        labels_str = ", ".join([f'"{label}"' for label in labels])
        prompt_text = f"""You are a meta-predictor trying to predict how a black-box classifier will label this image.

Task: Look at the provided image and predict which label the classifier will assign to it.

Available labels: [{labels_str}]

Analyze the image carefully and predict the classifier's output. The classifier may use subtle visual features that humans might not immediately notice.

Provide your prediction in the following XML format:
<prediction>your predicted label</prediction>

Important: Only output the XML tag with the exact label from the available labels list. Do not add any explanation."""

        return {"image_path": original_image, "text": prompt_text}

    return prompt if is_prompt else parser


# ============================================
# 2. PREDICT FROM ORIGINAL TEXT (chỉ có text mô tả)
# ============================================

def predict_from_original_text(is_prompt: bool = True) -> Callable:
    def prompt(original_text: Optional[str] = None, labels: List[str] = None, **kwargs) -> Dict:
        # Some datasets may pass the relevant text under `explain_text` / `description`
        # instead of `original_text`. Keep this robust and minimal.
        if original_text is None:
            original_text = (
                kwargs.get("original_text")
                or kwargs.get("explain_text")
                or kwargs.get("description")
                or ""
            )
        labels = labels or []
        labels_str = ", ".join([f'"{label}"' for label in labels])
        prompt_text = f"""You are a meta-predictor trying to predict how a black-box classifier will label an image based on its text description.

Description: "{original_text}"

Task: Based on this description, predict which label the classifier will assign to the corresponding image.

Available labels: [{labels_str}]

The classifier makes decisions based on visual features. Use the description to infer what visual elements might be present and predict the classifier's output.

Provide your prediction in the following XML format:
<prediction>your predicted label</prediction>

Important: Only output the XML tag with the exact label from the available labels list. Do not add any explanation."""

        # If an image is available, keep it so vLLM always gets an image_path.
        original_image = kwargs.get("original_image")
        return {"image_path": original_image, "text": prompt_text}

    return prompt if is_prompt else parser


# ============================================
# 3. PREDICT FROM EXPLANATION IMAGE (chỉ có ảnh giải thích - heatmap)
# ============================================

def predict_from_explanation_image(is_prompt: bool = True) -> Callable:
    def prompt(explain_image: str, labels: List[str], **kwargs) -> Dict:
        labels_str = ", ".join([f'"{label}"' for label in labels])
        prompt_text = f"""You are a meta-predictor trying to predict how a black-box classifier will label an image based on its explanation visualization.

You are shown an explanation image (e.g., attribution map, heatmap, or saliency map) that highlights which regions the classifier focuses on when making decisions.

Task: Analyze this explanation visualization and predict which label the classifier will assign to the original image.

Available labels: [{labels_str}]

The highlighted regions in the explanation indicate what features are important to the classifier. Use this information to infer the classifier's decision.

Provide your prediction in the following XML format:
<prediction>your predicted label</prediction>

Important: Only output the XML tag with the exact label from the available labels list. Do not add any explanation."""

        return {"image_path": explain_image, "text": prompt_text}

    return prompt if is_prompt else parser


# ============================================
# 4. PREDICT FROM EXPLANATION TEXT (ảnh gốc + text giải thích)
# ============================================

def predict_from_explanation_text(is_prompt: bool = True) -> Callable:
    def prompt(explain_text: str, labels: List[str], **kwargs) -> Dict:
        labels_str = ", ".join([f'"{label}"' for label in labels])
        prompt_text = f"""You are a meta-predictor trying to predict how a black-box classifier will label an image based on a textual explanation of its decision process.

You are provided with:
1. The original image
2. A textual explanation describing what features the classifier focuses on

Textual explanation: "{explain_text}"

Task: Look at the image and use the textual explanation to understand how the classifier makes decisions, then predict which label the classifier will assign.

Available labels: [{labels_str}]

The explanation describes which visual features are important to the classifier. Use this information to predict the classifier's output.

Provide your prediction in the following XML format:
<prediction>your predicted label</prediction>

Important: Only output the XML tag with the exact label from the available labels list. Do not add any explanation."""

        # vllm backend expects an `image_path` string.
        # This explainer variant uses `explain_text` but still uses the original image.
        original_image = kwargs.get("original_image")
        return {"image_path": original_image, "text": prompt_text}

    return prompt if is_prompt else parser


# ============================================
# 5. PREDICT FROM EXPLANATION IMAGE AND TEXT (ảnh giải thích + text giải thích)
# ============================================

def predict_from_explanation_image_and_text(is_prompt: bool = True) -> Callable:
    def prompt(explain_image: str, explain_text: str, labels: List[str], **kwargs) -> Dict:
        labels_str = ", ".join([f'"{label}"' for label in labels])
        prompt_text = f"""You are a meta-predictor trying to predict how a black-box classifier will label an image based on both visual and textual explanations.

You are provided with:
1. An explanation visualization (e.g., heatmap, attribution map) showing where the classifier focuses
2. A textual explanation describing the reasoning process

Textual explanation:
{explain_text}

Task: Analyze both the visual explanation map and the textual description to understand how the classifier makes decisions, then predict which label the classifier will assign to the original image.

Available labels: [{labels_str}]

The visual explanation shows which regions are important, while the text explains why. Combine both sources of information to predict the classifier's output.

Provide your prediction in the following XML format:
<prediction>your predicted label</prediction>

Important: Only output the XML tag with the exact label from the available labels list. Do not add any explanation."""

        return {"image_path": explain_image, "text": prompt_text}

    return prompt if is_prompt else parser

@aggregation
class UtilityK(BaseAggregationExplainer):
    target = ['summary', 'meta']
    def __init__(
        self,
        model,
        labels: Optional[List[str]] = None,
        vlm: Optional[Dict[str, Any]] = None,
        context_k: int = 5,
        n_trials: int = 3,
        n_query: Optional[int] = None,
        seed: int = 0,
        temperature: float = 0.0,
        max_tokens: int = 32,
        **kwargs,
    ):
        # We keep the BaseExplainer signature (expects a `model`) but this
        # aggregation variant only operates on *saved* explanation results.
        super().__init__(model)
        self.labels = labels or []
        self.vlm = None
        if vlm is not None:
            self.vlm = VLM_REGISTRY[vlm["name"]](**(vlm.get("kwargs") or {}))
        self.context_k = int(context_k)
        self.n_trials = int(n_trials)
        self.n_query = int(n_query) if n_query is not None else None
        self.seed = int(seed)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

    def explain(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # convert dict of list to list of dict
        list_inputs = [dict(zip(inputs.keys(), result)) for result in zip(*inputs.values())]

        # raw prediction
        has_original_image_key = bool(list_inputs) and 'original_image' in list_inputs[0]
        if has_original_image_key:
            fn_raw_prediction = predict_from_original_image
        else:
            fn_raw_prediction = predict_from_original_text

        # explanation prediction
        has_explain_image_key = bool(list_inputs) and 'explain_image' in list_inputs[0]
        has_explain_text_key = bool(list_inputs) and 'explain_text' in list_inputs[0]

        if has_explain_image_key and has_explain_text_key:
            fn_explain_prediction = predict_from_explanation_image_and_text
        elif has_explain_image_key:
            fn_explain_prediction = predict_from_explanation_image
        elif has_explain_text_key:
            fn_explain_prediction = predict_from_explanation_text
        else:
            raise ValueError("No explanation key found in inputs")

        prompts_raw_input = []
        prompts_explanation = []
        model_predictions = []
        existing_labels = set()
        for inp in list_inputs:
            if 'predicted_label' in inp['prediction']:
                existing_labels.add(inp['prediction']['predicted_label'])
            else:
                existing_labels.add(inp['prediction']['class_idx'])
        labels = list(existing_labels)

        for input in list_inputs:
            if 'top_labels' in input['prediction']:
                local_labels = input['prediction']['top_labels'][:20]
            elif 'top_predictions' in input['prediction']:
                local_labels = [self.labels[i] for i in input['prediction']['top_predictions'][:20]]
            else:
                local_labels = labels

            prediction = input.pop('prediction')
            if 'predicted_label' in prediction:
                model_predictions.append(prediction['predicted_label'])
            else:
                model_predictions.append(self.labels[prediction['class_idx']])
            raw_prompt = fn_raw_prediction(is_prompt=True)(**input, labels=local_labels)
            explain_prompt = fn_explain_prediction(is_prompt=True)(**input, labels=local_labels)
            prompts_raw_input.append(raw_prompt)
            prompts_explanation.append(explain_prompt)

        outputs = self.vlm.query_batch(prompts_raw_input + prompts_explanation, max_tokens=self.max_tokens)
        vlm_raw_prediction = [fn_raw_prediction(is_prompt=False)(output) for output in outputs[:len(prompts_raw_input)]]
        vlm_explanation_prediction = [fn_explain_prediction(is_prompt=False)(output) for output in outputs[len(prompts_raw_input):]]

        # compute utility k
        lemmatizer = WordNetLemmatizer()
        accurate_raw_prediction = len([0 for i, j in zip(model_predictions, vlm_raw_prediction) if lemmatizer.lemmatize(j.lower(), pos='n') == lemmatizer.lemmatize(i.lower(), pos='n')])
        accurate_explanation_prediction = len([0 for i, j in zip(model_predictions, vlm_explanation_prediction) if lemmatizer.lemmatize(j.lower(), pos='n') == lemmatizer.lemmatize(i.lower(), pos='n')])

        return inputs, {
            "model_predictions": model_predictions,
            "vlm_raw_output": outputs[:len(prompts_raw_input)],
            "vlm_explanation_output": outputs[len(prompts_raw_input):],
            "vlm_raw_prediction": vlm_raw_prediction,
            "vlm_explanation_prediction": vlm_explanation_prediction,
            'accurate_raw_prediction': accurate_raw_prediction,
            'accurate_explanation_prediction': accurate_explanation_prediction,
            'utility_k': accurate_explanation_prediction / (accurate_raw_prediction + 1e-10),
        }