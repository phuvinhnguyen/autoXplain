import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from autoXplain.explain.aggregation.base import BaseAggregationExplainer, aggregation
from autoXplain.utils.score import load_image, preprocess
from autoXplain.utils.vlm import VLM_REGISTRY

def _fuzzy_match(pred: str, gt: str) -> bool:
    if pred is None or gt is None:
        return False
    p = str(pred).strip().lower()
    g = str(gt).strip().lower()
    return (p == g) or (g in p) or (p in g)


def _score_to_text(score: Optional[float]) -> str:
    if score is None:
        return "No additional explanation signal is available."
    try:
        s = float(score)
    except Exception:
        return f"Explanation signal: {score}"
    if s >= 0.8:
        level = "HIGH"
        msg = "The model's attention/behavior appears strongly aligned with its prediction."
    elif s >= 0.5:
        level = "MODERATE"
        msg = "The model's attention/behavior partially aligns with its prediction; some noise may exist."
    else:
        level = "LOW"
        msg = "The model's attention/behavior does not align well with its prediction; it may use spurious cues."
    return f"Alignment score: {s:.3f} ({level}). {msg}"


@aggregation
class UtilityK(BaseAggregationExplainer):
    """VLM-simulated Utility-K evaluation.

    This is a *minimal* implementation inspired by `utilityk.md`.

    For each query image, we compare:
      - baseline condition: VLM predicts the classifier's label from the image only
      - explanation condition: same, but we add a lightweight explanation signal

    Utility for a query is: mean(correct_with_expl) - mean(correct_baseline) over trials.
    Ground truth is the *classifier prediction* (not the true label).

    Notes:
    - The VLM client interface only supports one image per query, so K-shot examples
      are provided as *text only* (labels + optional explanation text).
    - If `vlm` is None, we fall back to echoing the classifier prediction (utility=0),
      which is useful for smoke tests without a running VLM server.
    """

    def __init__(
        self,
        model,
        labels: List[str],
        vlm=None,
        explainer: Optional[Dict[str, Any]] = None,
        k: int = 5,
        n_trials: int = 3,
        n_query: Optional[int] = None,
        seed: int = 0,
        temperature: float = 0.0,
        max_tokens: int = 32,
        **kwargs,
    ):
        super().__init__(model)
        self.labels = labels or []
        self.vlm = VLM_REGISTRY[vlm['name']](**(vlm.get('kwargs') or {}))
        self.explainer_cfg = explainer or {}
        self.k = int(k)
        self.n_trials = int(n_trials)
        self.n_query = int(n_query) if n_query is not None else None
        self.seed = int(seed)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)

        self._explainer_type = self.explainer_cfg.get("type")  # "score" | "saliency" | None
        self._explainer_method = self.explainer_cfg.get("method")
        self._explainer_kwargs = dict(self.explainer_cfg.get("kwargs") or {})
        self._built_explainer = None

    def _build_inner_explainer(self):
        if not self._explainer_type or not self._explainer_method:
            return None
        if self._explainer_type == "score":
            from autoXplain.explain.score import SCORE_REGISTRY

            cls = SCORE_REGISTRY.get(self._explainer_method)
            if cls is None:
                raise ValueError(
                    f"Unknown score explainer: {self._explainer_method}. "
                    f"Available: {list(SCORE_REGISTRY.keys())}"
                )
            return cls(model=self.model, labels=self.labels, **self._explainer_kwargs)
        if self._explainer_type == "saliency":
            from autoXplain.explain.saliency import SALIENCY_REGISTRY

            cls = SALIENCY_REGISTRY.get(self._explainer_method)
            if cls is None:
                raise ValueError(
                    f"Unknown saliency explainer: {self._explainer_method}. "
                    f"Available: {list(SALIENCY_REGISTRY.keys())}"
                )
            return cls(model=self.model, model_type="classification", **self._explainer_kwargs)
        raise ValueError("explainer.type must be 'score' or 'saliency'")

    def _predict_classifier_label(self, image_path: str) -> Tuple[int, str, float]:
        img = load_image(image_path)
        x = preprocess(img).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
            conf, idx = probs.max(dim=0)
        class_idx = int(idx.item())
        name = self.labels[class_idx] if (self.labels and 0 <= class_idx < len(self.labels)) else str(class_idx)
        return class_idx, name, float(conf.item())

    def _build_prompt(
        self,
        example_items: List[Dict[str, Any]],
        include_expl: bool,
        query_expl_text: Optional[str],
    ) -> str:
        # Keep prompt short and deterministic; tell the VLM to predict what the *classifier* outputs.
        label_list = ", ".join(self.labels) if self.labels else "(unknown label set)"
        lines = [
            "You are simulating a naive user trying to predict what an image classifier will output.",
            "IMPORTANT: Predict the classifier's output label, NOT what the image actually shows.",
            f"Possible labels: {label_list}",
            "",
            "Here are some previous examples of classifier behavior:",
        ]
        for i, ex in enumerate(example_items, 1):
            if include_expl:
                expl = ex.get("expl_text") or "No explanation signal."
                lines.append(f"- Example {i}: classifier predicted '{ex['gt_label']}'. Explanation: {expl}")
            else:
                lines.append(f"- Example {i}: classifier predicted '{ex['gt_label']}'.")
        lines.append("")
        if include_expl and query_expl_text:
            lines.append("Now predict the classifier output for the NEW image. Additional explanation signal:")
            lines.append(query_expl_text)
        else:
            lines.append("Now predict the classifier output for the NEW image.")
        lines.append("Answer with ONLY the label string.")
        return "\n".join(lines)

    def _vlm_predict(self, image_path: str, prompt: str) -> str:
        if self.vlm is None:
            # Smoke-test fallback: echo classifier prediction.
            _, gt, _ = self._predict_classifier_label(image_path)
            return gt
        resp = self.vlm.query_batch(
            [{"image_path": image_path, "text": prompt}],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )[0]
        return str(resp).strip()

    def explain(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        rng = random.Random(self.seed)
        image_paths = list(inputs.get("image_paths") or [])
        if self.n_query is not None:
            image_paths = image_paths[: self.n_query]

        if len(image_paths) < max(2, self.k + 1):
            raise ValueError(f"Need at least K+1 images to run UtilityK. Got {len(image_paths)} (K={self.k}).")

        # Precompute classifier predictions (ground truth is classifier behavior)
        gt = {}
        for p in image_paths:
            _, name, conf = self._predict_classifier_label(p)
            gt[p] = {"gt_label": name, "confidence": conf}

        # Precompute explanation signals if configured
        expl_signal: Dict[str, Dict[str, Any]] = {}
        if self._built_explainer is None:
            self._built_explainer = self._build_inner_explainer()

        if self._built_explainer is not None:
            out = self._built_explainer({"image_paths": image_paths})
            # Score explainers return {'results': [...]}; saliency returns tensors/images arrays
            if "results" in out:
                for i, p in enumerate(image_paths):
                    item = out["results"][i]
                    expl_signal[p] = {
                        "score": item.get("score"),
                        "description": item.get("description"),
                        "justification": item.get("justification"),
                        "saliency_image": item.get("saliency_image"),
                        "masked_image": item.get("masked_image"),
                    }
            else:
                # Saliency output
                for i, p in enumerate(image_paths):
                    expl_signal[p] = {
                        "saliency_image": out.get("saliency_images", [None] * len(image_paths))[i],
                        "masked_image": out.get("masked_images", [None] * len(image_paths))[i],
                    }

        results = []
        for q_path in image_paths:
            q_gt = gt[q_path]["gt_label"]

            # Build pool excluding query
            pool = [p for p in image_paths if p != q_path]
            if len(pool) < self.k:
                continue

            baseline_correct = []
            expl_correct = []
            baseline_preds = []
            expl_preds = []

            for _ in range(self.n_trials):
                ex_paths = rng.sample(pool, self.k)
                example_items = []
                for p in ex_paths:
                    ex = {"gt_label": gt[p]["gt_label"]}
                    sig = expl_signal.get(p, {})
                    # Prefer score; fall back to short description.
                    expl_text = None
                    if "score" in sig and sig.get("score") is not None:
                        expl_text = _score_to_text(sig.get("score"))
                    elif sig.get("description"):
                        expl_text = str(sig["description"])[:300]
                    ex["expl_text"] = expl_text
                    example_items.append(ex)

                # Baseline: original image
                prompt_base = self._build_prompt(example_items, include_expl=False, query_expl_text=None)
                pred_base = self._vlm_predict(q_path, prompt_base)
                baseline_preds.append(pred_base)
                baseline_correct.append(_fuzzy_match(pred_base, q_gt))

                # Explanation condition:
                q_sig = expl_signal.get(q_path, {})
                q_expl_text = None
                send_image_path = q_path

                if self._explainer_type == "saliency":
                    # If the explainer provides a masked image, send it as the query image.
                    masked = q_sig.get("masked_image") or q_sig.get("saliency_image")
                    if isinstance(masked, Image.Image):
                        # Save to a temp file via PIL to bytes is heavy; simplest: keep original image path
                        # and add a textual hint. (Minimal implementation)
                        q_expl_text = "A saliency/masked attention visualization was computed for this image."
                    else:
                        q_expl_text = "A saliency/masked attention visualization was computed for this image."
                else:
                    if q_sig.get("score") is not None:
                        q_expl_text = _score_to_text(q_sig.get("score"))
                    elif q_sig.get("description"):
                        q_expl_text = str(q_sig.get("description"))[:300]

                prompt_expl = self._build_prompt(example_items, include_expl=True, query_expl_text=q_expl_text)
                pred_expl = self._vlm_predict(send_image_path, prompt_expl)
                expl_preds.append(pred_expl)
                expl_correct.append(_fuzzy_match(pred_expl, q_gt))

            base_acc = float(np.mean(baseline_correct)) if baseline_correct else 0.0
            expl_acc = float(np.mean(expl_correct)) if expl_correct else 0.0
            utility = float(expl_acc - base_acc)

            result = {
                "score": utility,
                "utility_k": utility,
                "baseline_accuracy": base_acc,
                "explanation_accuracy": expl_acc,
                "n_trials": self.n_trials,
                "k": self.k,
                "ground_truth_model_label": q_gt,
                "baseline_predictions": baseline_preds,
                "explanation_predictions": expl_preds,
                "prediction": {"label_name": q_gt, "confidence": gt[q_path]["confidence"]},
            }

            # Attach any images if available for saving by process_image
            sig = expl_signal.get(q_path, {})
            if isinstance(sig.get("saliency_image"), Image.Image):
                result["saliency_image"] = sig.get("saliency_image")
            if isinstance(sig.get("masked_image"), Image.Image):
                result["masked_image"] = sig.get("masked_image")

            results.append(result)

        return {"results": results}

