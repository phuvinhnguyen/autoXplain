## autoXplain: VLM-based automatic explanations for vision models

autoXplain is a framework that combines Vision Language Models (VLMs) with Class Activation Mapping (CAM) methods to automatically explain and evaluate vision model predictions. It produces explanations, saliency maps, and quantitative scores for how well a model’s focus aligns with the ground-truth label.

### Installation

- **From GitHub (editable install, recommended when working on this repo locally):**

```bash
git clone https://github.com/phuvinhnguyen/autoXplain.git
cd autoXplain
pip install -e .
```

- **Direct install:**

```bash
pip install git+https://github.com/phuvinhnguyen/autoXplain.git
```

You may also want to create and use a dedicated conda/virtual environment (e.g. with CUDA-compatible PyTorch and vLLM) before installing.

### Main features

- **Multiple CAM methods**:
  - GradCAM, SmoothGradCAM++, GradCAM++, CAM, ScoreCAM, LayerCAM, XGradCAM
- **Vision-Language Model (VLM) based evaluation**:
  - Uses a VLM (via vLLM) to judge whether a saliency map focuses on the correct region
- **Batch processing**:
  - Run over folders/datasets of images
- **Rich outputs**:
  - Explanations, masked images, and JSON metadata
- **Flexible configuration**:
  - Experiments are fully driven by YAML configs (model, CAM type, VLM, dataset, thresholds, etc.)

### Repository structure (high level)

- `autoXplain/` – core Python package:
  - `models/` – vision model definitions and wrappers
  - `explain/` – explanation and scoring methods (e.g. VLM-based judge)
  - `utils/vlm/` – VLM client utilities (vLLM-based)
  - `process_image.py` – main entry point for running experiments from a config
- `configs/` – example experiment configs (e.g. `vlm_judge.yaml`, `multi_dataset.yaml`)
- `datasets/` – (optional) local datasets used in experiments (e.g. `test_imgs`)
- `outputs/` – results from runs (masked images, metadata, summaries)
- `test/` – small notebooks and scripts for local testing

### Quick start: run a VLMJudge experiment

1. **Prepare a config**  
   Use or copy an existing config, for example:
   - `configs/vlm_judge.yaml` – runs the VLMJudge pipeline on a dataset.

2. **Point the config to your dataset**  
   In the YAML file:
   - Set `dataset.name` (a label for your dataset).
   - Set `dataset.path` to the folder containing your images.

3. **Choose the VLM and model**  
   In the same config:
   - `explain.method` – typically `VLMJudge` for VLM-based evaluation.
   - `explain.vlm.kwargs.model_name` – name of the VLM served by vLLM (e.g. a Qwen or LLaVA model).
   - Other keys control CAM type, thresholds, output directory, etc.

4. **Run the pipeline**

From the repository root:

```bash
python -m autoXplain.process_image --config configs/vlm_judge.yaml
```

This will:
- Load the vision model and CAM method
- Start or connect to a VLM server (via vLLM)
- Iterate over all images in the configured dataset
- Save explanations, saliency maps, and VLM-based scores into `outputs/`

### Configuration files

Configs in `configs/` define complete experiments. Common fields include:

- **Dataset section**:
  - `dataset.name` – human-readable name (e.g. `test_imgs`)
  - `dataset.path` – absolute or relative path to your image folder

- **Vision model section**:
  - Model architecture and weights to load (e.g. ResNet, MaxViT)

- **Explain / VLM section**:
  - `explain.method` – which explanation/evaluation method to use
  - `explain.vlm` – parameters for the VLM client (e.g. model name, host/port)

You can create new YAML files in `configs/` to define your own experiments, then pass them to `process_image.py` using the `--config` flag.

### Outputs

For a run configured with a dataset (e.g. `test_imgs`), you will typically see:

- `outputs/<dataset_name>/metadata/*.json` – per-image metadata files containing:
  - Model prediction
  - VLM-generated explanation/justification
  - VLM score and other evaluation metrics
- `outputs/<dataset_name>/masked_image/*.jpg` – masked CAM images highlighting the regions the model focuses on
- `outputs/<dataset_name>/summary.json` – aggregate statistics and summary over the dataset

The exact structure may vary slightly by config, but all outputs live under `outputs/`.

### Extending autoXplain

- **New datasets** – create a new folder with your images and point `dataset.path` to it.
- **New vision models** – add model definitions under `autoXplain/models/` and wire them into the model factory.
- **New VLMs** – if supported by vLLM, update the config’s `explain.vlm.kwargs.model_name` and ensure the model is available to vLLM.
- **Custom explainers** – implement new methods under `autoXplain/explain/` and reference them via `explain.method` in a config.

### Citation

If you use this work in your research, please cite:

```bib
@article{nguyen2025novel,
  title={A Novel Framework for Automated Explain Vision Model Using Vision-Language Models},
  author={Nguyen, Phu-Vinh and Pham, Tan-Hanh and Ngo, Chris and Hy, Truong Son},
  journal={arXiv preprint arXiv:2508.20227},
  year={2025}
}
```
