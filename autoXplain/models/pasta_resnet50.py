"""PASTA-trained ResNet50 model loader.

Loads dataset-specific ResNet50 weights from the PASTA project.
Each PASTA dataset (catsdogscars, coco, monumai, pascalpart) has its own
trained head with a different number of output classes.
"""

from pathlib import Path
import sys
from typing import Any, Dict

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from autoXplain.models.base import model

PASTA_ROOT = Path("/home/phng1216/Thesis/PASTA") # default PASTA root for local run, but can be changed in the config
PASTA_WEIGHTS_DIR = PASTA_ROOT / "models_pkl"

PASTA_LABELS = {
    "catsdogscars": ["cats", "dogs", "cars"],
    "monumai": ["Baroque", "Gothic", "Hispanic-Muslim", "Renaissance"],
    "coco": [
        "shopping_and_dining",
        "workplace",
        "home_or_hotel",
        "transportation",
        "sports_and_leisure",
        "cultural",
    ],
    "pascalpart": [
        "aeroplane", "bicycle", "bird", "bottle", "bus", "car",
        "cat", "cow", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "train", "tvmonitor",
    ],
}


class _PastaResnet50(nn.Module):
    """Architecture matching PASTA's Resnet50: backbone + linear head."""

    def __init__(self, output_dim: int, device: str = "cpu"):
        super().__init__()
        base = resnet50(weights=ResNet50_Weights.DEFAULT).to(device)
        self.resnet50 = nn.Sequential(*(list(base.children())[:-1]))
        self.linear = nn.Linear(2048, output_dim).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet50(x)
        x = x.squeeze(-1).squeeze(-1)
        return self.linear(x)

@model
def pasta_resnet50(dataset: str, model_type: str = "classification", pasta_root: Path = PASTA_ROOT, **kwargs: Any) -> Dict[str, Any]:
    """Load a PASTA-trained ResNet50 for a specific dataset.

    Args:
        dataset: One of 'catsdogscars', 'coco', 'monumai', 'pascalpart'.
        model_type: Always 'classification'.

    Returns:
        {'model': nn.Module, 'model_type': str, 'labels': list[str]}
    """
    PASTA_ROOT = pasta_root
    PASTA_WEIGHTS_DIR = PASTA_ROOT / "models_pkl"

    if dataset not in PASTA_LABELS:
        raise ValueError(
            f"Unknown PASTA dataset: {dataset}. "
            f"Available: {list(PASTA_LABELS.keys())}"
        )
    labels = PASTA_LABELS[dataset]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    net = _PastaResnet50(len(labels), device)

    ckpt_path = PASTA_WEIGHTS_DIR / f"best_resnet50_{dataset}.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"PASTA weights not found: {ckpt_path}")

    pasta_in_path = str(PASTA_ROOT) in sys.path
    if not pasta_in_path:
        sys.path.insert(0, str(PASTA_ROOT))
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    finally:
        if not pasta_in_path and str(PASTA_ROOT) in sys.path:
            sys.path.remove(str(PASTA_ROOT))

    if isinstance(ckpt, dict):
        net.load_state_dict(ckpt)
    else:
        net.load_state_dict(ckpt.state_dict())

    net.eval().to(device)
    return {"model": net, "model_type": model_type, "labels": labels}
