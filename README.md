# autoXplain: A VLM-based framework to automatically explain vision model with ease

autoXplain is a framework that combines Vision Language Models (VLMs) with various Class Activation Mapping (CAM) methods to automatically explain and evaluate vision model predictions. It provides detailed explanations, saliency maps, and quantitative evaluations of model performance.

## Installation

Install the autoXplain package:
```bash
pip install git+https://github.com/phuvinhnguyen/autoXplain.git
```

Or you can clone and install it
```bash
git clone https://github.com/phuvinhnguyen/autoXplain.git
cd autoXplain
pip install -e .
```

## Features

- Multiple CAM methods support:
  - GradCAM
  - SmoothGradCAM++
  - GradCAM++
  - CAM
  - ScoreCAM
  - LayerCAM
  - XGradCAM
- Automatic evaluation using Vision Language Models (VLMs)
- Batch processing of images
- Comprehensive result analysis and reporting
- Support for different vision models (ResNet18, MaxViT)
- Detailed performance metrics and visualizations

## Usage

### Basic Usage

```python
from autoXplain.evaluating import CamJudge
from FlowDesign.litellm import LLMInference
from torchcam.methods import GradCAM
from torchvision.models import resnet18
import torchvision
import json
import urllib.request

# Load labels of ImageNet models
url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
with urllib.request.urlopen(url) as response:
    class_idx = json.load(response)
labels = [class_idx[str(i)][1] for i in range(1000)]

# Load LLM
bot = LLMInference("gemini/gemini-1.5-flash", api_key='<API_TOKEN>')

# Load vision model
model = resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)

# Create workflow
agent = CamJudge(bot, GradCAM, model, labels=labels)

# Run framework
output = agent({'image': 'path/to/image.png', 'label': 'label_of_the_image'})

print(output)
```

### Batch Processing

For processing multiple images and generating XAI confusion matrix, use the `examples/process_folder.py` script:

```bash
python examples/process_folder.py input_folder \
    --save_dir autoXplain_results \
    --model resnet18 \
    --cam_type gradcam \
    --threshold 2.5 \
    --vlm_model gemini/gemini-1.5-flash \
    --api_key YOUR_API_KEY_1,YOUR_API_KEY_2
```

#### Command Line Arguments

- `input_folder`: Path to folder containing images (required)
- `--save_dir`: Path to save processed results (default: 'autoXplain_results')
- `--model`: Model to use (choices: 'resnet18', 'maxvit_t', ..., default: 'resnet18')
- `--cam_type`: Type of CAM method to use (choices: 'gradcam', 'smoothgradcam', 'gradcamplusplus', 'cam', 'scorecam', 'layercam', 'xgradcam', default: 'gradcam')
- `--threshold`: Threshold for VLM score (higher means good, default: 2.5)
- `--vlm_model`: name of VLM used for evaluating (default: 'gemini/gemini-1.5-flash')
- `--api_key`: Google API key for Gemini model (required)

#### Image Naming Convention

Images should be named in the format: `id_label.extension`
Example: `001_cat.jpg`, `002_dog.png`

## Output

The framework provides comprehensive outputs:

### For Single Image Processing
- Saliency map
- Masked CAM image
- Description
- Justification
- Score
- Prediction

### For Batch Processing
- All individual image results
- Final report categorizing results into four cases:
  - Correct predictions with high VLM score
  - Correct predictions with low VLM score
  - Wrong predictions with high VLM score
  - Wrong predictions with low VLM score
- Summary statistics
- Detailed analysis of each case

## Pipeline

The pipeline follows these steps:
1. Takes model and images as input
2. Computes attention (saliency map) using the selected CAM method
3. Uses VLMs to evaluate and score samples
4. Computes confusion matrix of VLMs' judgment and accuracy
5. Generates comprehensive reports and visualizations

## Supported CAM Methods

- **GradCAM**: Standard gradient-based class activation mapping
- **SmoothGradCAM++**: Improved version with noise reduction
- **GradCAM++**: Enhanced version with better localization
- **CAM**: Original Class Activation Mapping
- **ScoreCAM**: Score-weighted class activation mapping
- **LayerCAM**: Layer-wise class activation mapping
- **XGradCAM**: Extended gradient-based class activation mapping

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- torchcam
- FlowDesign
- Google API key for Gemini model (by default)

<span style="float: right; font-size: 10px; color: gray;">V-003</span>