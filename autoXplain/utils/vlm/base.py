from abc import ABC, abstractmethod
from typing import List, Dict, Any

VLM_REGISTRY = {}

def vlm(cls):
    VLM_REGISTRY[cls.__name__] = cls
    return cls

class VLMBase(ABC):
    @abstractmethod
    def query_batch(self, queries: List[Dict[str, Any]], **kwargs) -> List[str]:
        """
        Args:
            queries: list of {"image_path": str, "text": str}
        Returns:
            list of response strings in same order
        """
        ...