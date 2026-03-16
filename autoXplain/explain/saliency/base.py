SALIENCY_REGISTRY = {}

def saliency(cls):
    SALIENCY_REGISTRY[cls.__name__] = cls
    return cls
