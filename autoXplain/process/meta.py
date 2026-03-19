from autoXplain.process.base import *
from typing import List, Dict, Any
from pathlib import Path
import json, os, glob
import numpy as np

@process
class MetaProcess(BaseProcess):
    """Process that feeds saved explanation results into aggregation methods.

    The dataset config should point to a `summary.jsonl` produced by another
    explainer run (or a directory containing that file).
    """

    def _load_items(self, ds_cfg) -> List[Dict[str, Any]]:
        """Load all summary.jsonl files in the dataset directory and return a list of items."""
        summary_paths = [f for f in glob.glob(os.path.join(ds_cfg['path'], '**', 'summary.jsonl'), recursive=True)]
        items = [] # list of dicts
        targeted_keys = ds_cfg.get('targeted_keys', {})
        for summary_path in summary_paths:
            print(summary_path)
            with open(summary_path) as f:
                items.extend([json.loads(line) for line in f])

        # return dict of list
        try:
            return {k: [item[v] for item in items] for k, v in targeted_keys.items()}
        except Exception as e:
            print(f"Error loading items: {e}")
            raise e

    def _process(self, ds_cfg, exp: BaseExplainer) -> List[Dict[str, Any]]:
        items = self._load_items(ds_cfg)
        out = exp.run(items)

        if isinstance(out, Tuple):
            outputs = []
            for o in out:
                # Check equal length of keys (List)
                size = [len(i) if isinstance(i, list) else i for i in o.values()]
                try:
                    if all(s == size[0] for s in size[1:]):
                        outputs.append([dict(zip(o.keys(), result)) for result in zip(*o.values())])
                    else:
                        outputs.append(o)
                except:
                    outputs.append(o)
            return tuple(outputs)
        else:
            return [dict(zip(out.keys(), result)) for result in zip(*out.values())]

