from typing import Any, Dict

from torchvision.models import get_model, get_model_weights

from autoXplain.models.base import model


@model
def torchvision(name: str = 'resnet18', model_type: str = 'classification',
                weight: str = 'DEFAULT', **kwargs: Any) -> Dict[str, Any]:
    """Load a torchvision model with auto-detected labels.

    Args:
        name: model function name (e.g. ``'resnet50'``, ``'deeplabv3_resnet50'``,
              ``'fasterrcnn_resnet50_fpn'``).
        model_type: ``'classification'``, ``'segmentation'``, or ``'detection'``.
        weight: weight variant (default ``'DEFAULT'``). Pass a specific enum
                name like ``'IMAGENET1K_V2'`` to pick a variant.

    Returns:
        ``{'model': nn.Module, 'model_type': str, 'labels': list[str]}``
    """
    weights_enum = get_model_weights(name)
    weights = getattr(weights_enum, weight)
    net = get_model(name, weights=weights, **kwargs).eval()
    labels = list(weights.meta.get('categories', []))
    return {'model': net, 'model_type': model_type, 'labels': labels}
