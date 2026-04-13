MODEL_REGISTRY = {}

def model(cls):
    MODEL_REGISTRY[cls.__name__] = cls
    return cls

@model
def empty(*args, **kwargs):
    '''
    Empty model that returns an empty dictionary.
    Useful for testing or when model is not inputted in this way (e.g. for NLP models).
    '''
    return {"model": None, "model_type": "empty", "labels": []}