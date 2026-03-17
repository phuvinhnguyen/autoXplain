from .base import BaseExplainer
from .saliency import SALIENCY_REGISTRY
from .score import SCORE_REGISTRY
from .aggregation import AGGREGATION_REGISTRY

EXPLAIN_REGISTRY = {
    **SALIENCY_REGISTRY,
    **SCORE_REGISTRY,
    **AGGREGATION_REGISTRY
    }