from abc import ABC, abstractmethod
from typing import Any, Dict
import torch

EVALUATE_REGISTRY = {}

def evaluate(cls):
    EVALUATE_REGISTRY[cls.__name__] = cls
    return cls


@evaluate
class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, prediction_path: str, reference_path: str) -> Dict[str, Any]:
        """Evaluate the prediction and reference paths.
        Args:
            prediction_path: str
            reference_path: str
        Returns:
            Dict[str, Any]
        """
        ...