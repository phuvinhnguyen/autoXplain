from autoXplain.models import MODEL_REGISTRY
from autoXplain.explain import EXPLAIN_REGISTRY

def build_model(model_cfg):
    """Build model via the registry. Returns {'model', 'model_type', 'labels'}."""
    name = model_cfg.get('name', 'torchvision')
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model name: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    kwargs = {k: v for k, v in model_cfg.items() if k != 'name'}
    return MODEL_REGISTRY[name](**kwargs)

def build_explainer(explain_cfg, model_info):
    method_name = explain_cfg['name']
    if method_name not in EXPLAIN_REGISTRY:
        raise ValueError(
            f"Unknown method: {method_name}. "
            f"Score: {list(EXPLAIN_REGISTRY.keys())}"
        )

    kwargs = dict(explain_cfg.get('kwargs') or {})
    kwargs['labels'] = model_info['labels']
    kwargs['model_type'] = model_info['model_type']
    kwargs['model'] = model_info['model']

    return EXPLAIN_REGISTRY[method_name](**kwargs)