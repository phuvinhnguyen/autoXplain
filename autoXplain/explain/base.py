from abc import ABC, abstractmethod
from typing import Any, Dict
import torch

class BaseExplainer(ABC):
    target = 'summary'
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device

    @abstractmethod
    def explain(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Explain model on a single input. Returns dict with explanation results.
        Args:
            inputs: Dict[str, Any]
        Returns:
            Dict[str, Any]
        """
        ...

    def run(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self.explain(inputs)

    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return self.explain(inputs)
