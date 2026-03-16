from autoXplain.explain.base import BaseExplainer

SCORE_REGISTRY = {}


def score(cls):
    SCORE_REGISTRY[cls.__name__] = cls
    return cls


class BaseScoreExplainer(BaseExplainer):
    """Base for score methods that use a saliency explainer internally.

    ``saliency_config`` is a dict like::

        {'method': 'CAMSaliency', 'cam': 'GradCAM', 'layer': 0,
         'model_type': 'classification', 'slope': 25, 'position': 0.4}

    The saliency explainer is instantiated in ``__init__`` and stored as
    ``self.saliency``.  Score methods call ``self.get_saliency(inputs)`` to
    obtain saliency maps before computing their own metrics.
    """

    def __init__(self, model, saliency_config=None, **kwargs):
        super().__init__(model)
        self.saliency_config = saliency_config or {}
        self.saliency = self._build_saliency() if saliency_config else None

    def _build_saliency(self):
        from autoXplain.explain.saliency import SALIENCY_REGISTRY
        cfg = dict(self.saliency_config)
        method_name = cfg.pop('method')
        if method_name not in SALIENCY_REGISTRY:
            raise ValueError(f"Unknown saliency: {method_name}. Available: {list(SALIENCY_REGISTRY.keys())}")
        return SALIENCY_REGISTRY[method_name](model=self.model, **cfg)

    def get_saliency(self, inputs):
        """Run the saliency explainer on inputs and return its output dict."""
        if self.saliency is None:
            raise RuntimeError("No saliency method configured")
        return self.saliency(inputs)
