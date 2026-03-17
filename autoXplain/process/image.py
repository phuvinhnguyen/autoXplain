from autoXplain.process.base import *
from typing import List, Dict, Any
from pathlib import Path

@process
class ImageProcess(BaseProcess):
    def _process(self, ds_cfg, exp: BaseExplainer) -> List[Dict[str, Any]]:
        prefix = ds_cfg.get('prefix', None)
        input_dir = Path(ds_cfg['path'])
        images = sorted(f for f in input_dir.iterdir()
                        if f.suffix.lower() in ('.jpg', '.jpeg', '.png'))
        if prefix:
            images = [f for f in images if f.name.startswith(prefix)]

        output = exp.run({'image_paths': images}) # dict of list
         
        # convert dict of list to list of dict
        return [dict(zip(output.keys(), result)) for result in zip(*output.values())]
