from .base import VLM_REGISTRY, VLMBase, vlm

def _try_import(mod):
    try:
        __import__(mod, globals(), locals(), level=0)
    except ImportError:
        pass

_try_import('autoXplain.utils.vlm.vllm')
_try_import('autoXplain.utils.vlm.openai')
_try_import('autoXplain.utils.vlm.gemini')

__all__ = ['VLM_REGISTRY', 'VLMBase', 'vlm']
