#!/usr/bin/env python
"""
Run explanation pipeline from a YAML config.

Usage:
    python -m autoXplain.process_image --config configs/example.yaml
"""
import argparse
import json
from pathlib import Path

import yaml
from PIL import Image

from autoXplain.models import MODEL_REGISTRY
from autoXplain.explain.score import SCORE_REGISTRY
from autoXplain.utils.vlm import VLM_REGISTRY


def build_model(model_cfg):
    """Build model via the registry. Returns {'model', 'model_type', 'labels'}."""
    source = model_cfg.get('source', 'torchvision')
    if source not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model source: {source}. Available: {list(MODEL_REGISTRY.keys())}")
    kwargs = {k: v for k, v in model_cfg.items() if k != 'source'}
    return MODEL_REGISTRY[source](**kwargs)


def build_vlm(vlm_cfg):
    if vlm_cfg is None:
        return None
    name = vlm_cfg['client']
    if name not in VLM_REGISTRY:
        raise ValueError(f"Unknown VLM: {name}. Available: {list(VLM_REGISTRY.keys())}")
    return VLM_REGISTRY[name](**(vlm_cfg.get('kwargs') or {}))


def build_explainer(explain_cfg, model_info):
    """Build the score explainer from config and model info dict."""
    method_name = explain_cfg['method']
    if method_name not in SCORE_REGISTRY:
        raise ValueError(f"Unknown score method: {method_name}. Available: {list(SCORE_REGISTRY.keys())}")

    saliency_config = dict(explain_cfg.get('saliency', {}))
    saliency_config['model_type'] = model_info['model_type']

    kwargs = dict(explain_cfg.get('kwargs') or {})
    kwargs['saliency_config'] = saliency_config
    kwargs['labels'] = model_info['labels']

    if 'vlm' in explain_cfg:
        kwargs['vlm'] = build_vlm(explain_cfg['vlm'])

    return SCORE_REGISTRY[method_name](model=model_info['model'], **kwargs)


def save_result(result, img_stem, output_dir):
    for key in ('saliency_image', 'masked_image'):
        img = result.get(key)
        if isinstance(img, Image.Image):
            d = output_dir / key
            d.mkdir(exist_ok=True)
            img.save(d / f"{img_stem}.jpg")
            result[f'{key}_path'] = str(d / f"{img_stem}.jpg")

    serializable = {k: v for k, v in result.items() if not isinstance(v, Image.Image)}
    meta_dir = output_dir / 'metadata'
    meta_dir.mkdir(exist_ok=True)
    with open(meta_dir / f"{img_stem}.json", 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    return serializable


def process_dataset(ds_cfg, explain_cfg, output_root):
    ds_name = ds_cfg['name']
    model_cfg = ds_cfg['model']

    output_dir = Path(output_root) / ds_name
    output_dir.mkdir(parents=True, exist_ok=True)

    model_info = build_model(model_cfg)

    print(f"\n{'='*60}")
    print(f"Dataset: {ds_name}")
    print(f"  model={model_cfg.get('name')}  type={model_info['model_type']}  "
          f"labels={len(model_info['labels'])}  method={explain_cfg['method']}")
    print(f"  output -> {output_dir}")
    print(f"{'='*60}")

    exp = build_explainer(explain_cfg, model_info)

    input_dir = Path(ds_cfg['path'])
    images = sorted(f for f in input_dir.iterdir()
                    if f.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    print(f"  Processing {len(images)} images ...")

    image_paths = [str(p) for p in images]
    output = exp.run({'image_paths': image_paths})

    all_serializable = []
    for i, result in enumerate(output.get('results', [])):
        stem = images[i].stem
        result['image'] = str(images[i])
        ser = save_result(result, stem, output_dir)
        all_serializable.append(ser)

        score_str = f"score={result.get('score')}" if 'score' in result else ''
        pred = result.get('prediction', '')
        print(f"  [{i+1}/{len(images)}] {images[i].name}  {score_str}  prediction={pred}")

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(all_serializable, f, indent=2, default=str)

    print(f"  Done: {len(all_serializable)} results -> {output_dir}")
    return all_serializable


def run_config(config):
    explain_cfg = config['explain']
    datasets = config['datasets']
    output_root = config.get('output', 'outputs')

    all_results = {}
    for ds_cfg in datasets:
        results = process_dataset(ds_cfg, explain_cfg, output_root)
        all_results[ds_cfg['name']] = results
    return all_results


def main():
    parser = argparse.ArgumentParser(description='autoXplain explanation pipeline')
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_config(config)


if __name__ == '__main__':
    main()
