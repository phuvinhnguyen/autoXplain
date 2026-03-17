#!/usr/bin/env python
"""
Run explanation pipeline from a YAML config.

Usage:
    python -m autoXplain.process --config configs/example.yaml
"""
import argparse
import yaml
from autoXplain.process import PROCESS_REGISTRY


def run_config(config):
    explain_cfg = config['explain']
    datasets = config['datasets']
    output_root = config.get('output', 'outputs')

    for ds_cfg in datasets:
        print(f"Processing {ds_cfg}")
        if ds_cfg.get('name') not in PROCESS_REGISTRY:
            for process in PROCESS_REGISTRY.values():
                if process.is_dataset(ds_cfg):
                    process().process(ds_cfg, explain_cfg, output_root)
                    break
        else:
            PROCESS_REGISTRY[ds_cfg.get('name')]().process(ds_cfg, explain_cfg, output_root)

def main():
    parser = argparse.ArgumentParser(description='autoXplain explanation pipeline')
    parser.add_argument('--config', required=True, help='Path to YAML config file')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_config(config)

if __name__ == '__main__':
    main()
