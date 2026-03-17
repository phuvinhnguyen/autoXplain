"""
Evaluate scoring methods against PASTA golden scores.

Metrics (aligned with PASTA project evaluation):
- Pearson correlation: linear correlation between method scores and PASTA golden
- Spearman correlation: rank correlation (robust to scale)
- MSE: mean squared error (both normalized to [0,1])
- R2: coefficient of determination
- Cohen's Kappa (quadratic weighted): agreement when binning scores to 5 classes
- Pseudo-classification accuracy: % of predictions in same interval as reference
"""

from argparse import ArgumentParser
import json
import os, glob
from typing import Any, Dict, Tuple

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score, mean_squared_error

from autoXplain.evaluate.base import BaseEvaluator, evaluate


def _align_dicts_simple(
    reference_dict: Dict[str, float], prediction_dict: Dict[str, float]
) -> Tuple[np.ndarray, np.ndarray]:
    """Align by common keys (stem = filename without extension). Return (ref_arr, pred_arr)."""
    def stem(s):
        b = os.path.basename(str(s))
        return b.rsplit(".", 1)[0] if "." in b else b

    ref_by_stem = {stem(k): v for k, v in reference_dict.items()}
    pred_by_stem = {stem(k): v for k, v in prediction_dict.items()}
    common = sorted(set(ref_by_stem.keys()) & set(pred_by_stem.keys()))
    ref_arr = np.array([ref_by_stem[k] for k in common], dtype=float)
    pred_arr = np.array([pred_by_stem[k] for k in common], dtype=float)
    return ref_arr, pred_arr


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    r = np.corrcoef(x, y)[0, 1]
    return float(r) if not np.isnan(r) else 0.0


def _normalize_to_01(x: np.ndarray, x_min: float = None, x_max: float = None) -> np.ndarray:
    if x_min is None:
        x_min = np.nanmin(x)
    if x_max is None:
        x_max = np.nanmax(x)
    span = x_max - x_min
    if span <= 0:
        return np.zeros_like(x)
    return (x - x_min) / span


def _normalize_ref_to_01(x: np.ndarray) -> np.ndarray:
    """PASTA golden scores are in [1, 5]. Map to [0,1]."""
    return (x - 1.0) / 4.0


def _scale_pred_to_15(pred: np.ndarray) -> np.ndarray:
    """Min-max scale prediction to [1, 5] to match PASTA scale."""
    pmin, pmax = np.nanmin(pred), np.nanmax(pred)
    if pmax <= pmin:
        return np.full_like(pred, 3.0)
    return 1.0 + 4.0 * (pred - pmin) / (pmax - pmin)


def _bin_to_5_classes(x: np.ndarray) -> np.ndarray:
    """Bin continuous scores to integers 1..5."""
    x = np.clip(x, 1.0, 5.0)
    return np.round(x).astype(int)


@evaluate
class ScoreImageEvaluator(BaseEvaluator):
    """Evaluate scoring method outputs against PASTA golden scores."""

    def pearson_correlation(self, ref: np.ndarray, pred: np.ndarray) -> float:
        return _safe_corrcoef(ref, pred)

    def spearman_correlation(self, ref: np.ndarray, pred: np.ndarray) -> float:
        r, _ = spearmanr(ref, pred)
        return float(r) if not np.isnan(r) else 0.0

    def mse(self, ref: np.ndarray, pred: np.ndarray) -> float:
        ref_n = _normalize_ref_to_01(ref)
        pred_n = _normalize_to_01(pred)
        return float(mean_squared_error(ref_n, pred_n))

    def r2_score(self, ref: np.ndarray, pred: np.ndarray) -> float:
        ref_n = _normalize_ref_to_01(ref)
        pred_n = _normalize_to_01(pred)
        ss_res = np.sum((ref_n - pred_n) ** 2)
        ss_tot = np.sum((ref_n - np.mean(ref_n)) ** 2)
        if ss_tot <= 0:
            return 0.0
        return float(1.0 - ss_res / ss_tot)

    def cohens_kappa_quadratic(self, ref: np.ndarray, pred: np.ndarray) -> float:
        """Quadratic weighted Cohen's Kappa (PASTA metric). Bins to 5 classes."""
        ref_bin = _bin_to_5_classes(ref)
        pred_scaled = _scale_pred_to_15(pred)
        pred_bin = _bin_to_5_classes(pred_scaled)
        k = cohen_kappa_score(ref_bin, pred_bin, weights="quadratic")
        return float(k) if not np.isnan(k) else 0.0

    def pseudo_classification_accuracy(self, ref: np.ndarray, pred: np.ndarray, radius: float = 0.5) -> float:
        """% of predictions within ±radius of reference (PASTA-style)."""
        pred_scaled = _scale_pred_to_15(pred)
        correct = np.sum(np.abs(pred_scaled - ref) <= radius)
        return float(correct / len(ref)) if len(ref) > 0 else 0.0

    def evaluate(self, prediction_path: str, reference_path: str) -> Dict[str, Any]:
        with open(reference_path) as f:
            reference = json.load(f)
        ref_dict = {os.path.basename(k).split(".")[0]: float(v) for k, v in reference.items()}

        pred_dict = {}
        summary_paths = [f for f in glob.glob(os.path.join(prediction_path, '**', 'summary.jsonl'), recursive=True)]
        for summary_path in summary_paths:
            with open(summary_path) as f:
                prediction = [json.loads(line) for line in f]

            for item in prediction:
                img = item.get("id", "").split(".")[0]
                score = item.get("score")
                if score is None:
                    continue
                stem = os.path.basename(str(img)).rsplit(".", 1)[0] if "." in os.path.basename(str(img)) else str(img)
                pred_dict[stem] = float(score)

        ref_arr, pred_arr = _align_dicts_simple(ref_dict, pred_dict)
        if len(ref_arr) < 2:
            return {
                "pearson_correlation": 0.0,
                "spearman_correlation": 0.0,
                "mse": 0.0,
                "r2": 0.0,
                "cohens_kappa_quadratic": 0.0,
                "pseudo_classification_accuracy": 0.0,
                "n_matched": len(ref_arr),
            }

        return {
            "pearson_correlation": self.pearson_correlation(ref_arr, pred_arr),
            "spearman_correlation": self.spearman_correlation(ref_arr, pred_arr),
            "mse": self.mse(ref_arr, pred_arr),
            "r2": self.r2_score(ref_arr, pred_arr),
            "cohens_kappa_quadratic": self.cohens_kappa_quadratic(ref_arr, pred_arr),
            "pseudo_classification_accuracy": self.pseudo_classification_accuracy(ref_arr, pred_arr),
            "n_matched": len(ref_arr),
        }


def main():
    parser = ArgumentParser()
    parser.add_argument("--prediction_path", "-p", type=str, required=True)
    parser.add_argument("--reference_path", "-r", type=str, required=True)
    parser.add_argument("--output_path", "-o", type=str, default=None)
    args = parser.parse_args()
    evaluator = ScoreImageEvaluator()
    result = evaluator.evaluate(args.prediction_path, args.reference_path)
    result["prediction_path"] = args.prediction_path
    result["reference_path"] = args.reference_path
    if args.output_path:
        os.makedirs(args.output_path, exist_ok=True)
        with open(os.path.join(args.output_path, "evaluation.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
