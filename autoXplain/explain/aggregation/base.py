from autoXplain.explain.base import BaseExplainer

AGGREGATION_REGISTRY = {}


def aggregation(cls):
    AGGREGATION_REGISTRY[cls.__name__] = cls
    return cls


class BaseAggregationExplainer(BaseExplainer):
    """Base for aggregation-style explainers that operate over a dataset.

    This follows the same pattern as score/saliency explainers: implement `explain(inputs)`.
    Inputs are typically `{'image_paths': [str, ...]}`.
    """

