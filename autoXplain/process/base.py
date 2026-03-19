from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple
from autoXplain.explain.base import BaseExplainer
from autoXplain.utils.general import build_model, build_explainer
import os
from PIL import Image
import json

PROCESS_REGISTRY = {}

def process(cls):
    PROCESS_REGISTRY[cls.__name__] = cls
    return cls

class BaseProcess(ABC):
    @classmethod
    def is_dataset(cls, ds_cfg) -> bool:
        return ds_cfg.get('name') == cls.__name__

    def process(self, ds_cfg, explain_cfg, output_root) -> List[Dict[str, Any]]:
        ds_name = ds_cfg['ds_name']
        model_name = '-'.join([str(i) for i in ds_cfg['model'].values()])
        model_info = build_model(ds_cfg.pop('model'))
        exp = build_explainer(explain_cfg, model_info)
        target = exp.target
        output_path = os.path.join(output_root, ds_name, model_name)
        os.makedirs(output_path, exist_ok=True)
        return [self.save(i, output_path, target) for i in self._process(ds_cfg, exp)]

    @abstractmethod
    def _process(self, ds_cfg, exp: BaseExplainer) -> List[Dict[str, Any]]:  # use yield to return results
        ...

    def save(self, result: Dict[str, Any], output_root: str, target: str = 'summary') -> Dict[str, Any]:
        def serialize(k, v, index=None):
            if isinstance(v, Image.Image):
                index = '-' + str(index) if index is not None else ''
                os.makedirs(os.path.join(output_root, *k.split('/')), exist_ok=True)
                path = os.path.join(output_root, *k.split('/'), result.get('id', 'unknown') + f'{index}.jpg')
                v.save(path)
                return path
            elif isinstance(v, list):
                return [serialize(k, i, index) for index, i in enumerate(v)]
            elif isinstance(v, dict):
                return {nk: serialize(k + '/' + nk, v) for nk, v in v.items()}
            return v

        if isinstance(result, Tuple):
            for index, item in enumerate(result):
                normal_data = {k: serialize(k, v) for k, v in item.items()}
                with open(os.path.join(output_root, f'{target[index]}.jsonl'), 'a') as f:
                    f.write(json.dumps(normal_data) + '\n')
            return result
        else:
            normal_data = {k: serialize(k, v) for k, v in result.items()}

            with open(os.path.join(output_root, f'{target}.jsonl'), 'a') as f:
                f.write(json.dumps(normal_data) + '\n')
            return result