MODEL_REGISTRY = {}

def model(cls):
    MODEL_REGISTRY[cls.__name__] = cls
    return cls
