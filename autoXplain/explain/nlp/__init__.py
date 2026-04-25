from .base import NLP_REGISTRY, BaseNLPExplainer, nlp
from . import agent  # noqa: F401 — registers NLPAgent in NLP_REGISTRY

__all__ = ["NLP_REGISTRY", "BaseNLPExplainer", "nlp"]

