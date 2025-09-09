import os
import json
import hashlib
import shutil
from pathlib import Path
from time import sleep
import traceback

from autoXplain.evaluating import CamJudge, PROMPT_1, PROMPT_2, PROMPT_3
from autoXplain.old_evaluating import OldMaskedCamJudge, OldOriginalCamJudge
from FlowDesign.litellm import LLMInference
from torchcam.methods import (
    GradCAM,
    SmoothGradCAMpp,
    GradCAMpp,
    CAM,
    ScoreCAM,
    LayerCAM,
    XGradCAM,
)
from torchvision import models
import urllib.request
import PIL.Image

METHODS = {
    'maskedcam': CamJudge,
    'oldmaskedcam': OldMaskedCamJudge,
    'oldoriginalcam': OldOriginalCamJudge,
}


def get_image_hash(image_path):
    """Generate MD5 hash for image content"""
    # with open(image_path, 'rb') as f:
    #     return hashlib.md5(f.read()).hexdigest()
    return os.path.basename(image_path).split('.')[0] # currently dont need to hash the image

def create_directory_structure(save_dir):
    """Create necessary subdirectories in save directory"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for subdir in ['original', 'saliency', 'maskedcam', 'metadata']:
        (Path(save_dir) / subdir).mkdir(exist_ok=True)

def get_cam_method(cam_type):
    """Get the appropriate CAM method class based on the input type"""
    cam_methods = {
        'gradcam': GradCAM,
        'smoothgradcam': SmoothGradCAMpp,
        'gradcamplusplus': GradCAMpp,
        'cam': CAM,
        'scorecam': ScoreCAM,
        'layercam': LayerCAM,
        'xgradcam': XGradCAM,
    }
    
    cam_type = cam_type.lower()
    if cam_type not in cam_methods:
        raise ValueError(f"Unsupported CAM type: {cam_type}. Supported types are: {', '.join(cam_methods.keys())}")
    
    return cam_methods[cam_type]

def process_image(
    image_path,
    save_dir,
    bot,
    model,
    labels,
    api_tokens,
    cam_type,
    prompt,
    method='maskedcam',
    api_index=0,
    slope=25,
    position=0.6,
    model_type='classification',
    delay_seconds=7):
    """Process a single image with structured saving"""
    create_directory_structure(save_dir)
    image_hash = get_image_hash(image_path)
    metadata_path = Path(save_dir) / 'metadata' / f"{image_hash}.json"

    # Check if already processed
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    # Copy original image
    original_dest = Path(save_dir) / 'original' / f"{image_hash}.jpg"
    shutil.copy(image_path, original_dest)
    
    # Process image
    result = None
    cam_method = get_cam_method(cam_type)
    agent = METHODS[method](bot, cam_method, model, labels=labels, slope=slope, position=position, model_type=model_type)
    agent.PROMPT = PROMPT_1 if prompt == 1 else PROMPT_2 if prompt == 2 else PROMPT_3

    for _ in range(7):
        try:
            sleep(delay_seconds)  # prevent exceed quota
            if len(api_tokens) != 0:
                api_index = api_index % len(api_tokens)
                agent.bot.api_key = api_tokens[api_index]
                api_index += 1
            if model_type != 'classification':
                label = os.path.basename(image_path).split('_')[1].split('.')[0]
            else:
                label = 'unk'
            result = agent({'image': image_path, 'label': label})
            break
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            print(f"Full traceback: {traceback.format_exc()}")
    
    if not result: return None #mean the img not present in labels
    else:
        # Save processed images
        saliency_path = Path(save_dir) / 'saliency' / f"{image_hash}.jpg"
        result['saliency'].save(saliency_path)
        maskedcam_path = Path(save_dir) / 'maskedcam' / f"{image_hash}.jpg"
        result['maskedcam'].save(maskedcam_path)
        for key, value in result['prediction'].items():
            if isinstance(value, PIL.Image.Image):
                os.makedirs(Path(save_dir) / key, exist_ok=True)
                value.save(Path(save_dir) / key / f"{image_hash}.jpg")
                result['prediction'][key] = str(Path(save_dir) / key / f"{image_hash}.jpg")
        
        result.update({
            'original': image_path,
            'saliency': str(saliency_path),
            'maskedcam': str(maskedcam_path),
            'image': str(original_dest),
            'cam_type': cam_type
        })
        
    result['image_hash'] = image_hash
    
    with open(metadata_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    return result

def generate_final_report(results):
    """Generate a final report categorizing results into 4 cases"""
    cases = {
        'correct_high': [],  # correct prediction + high VLM score
        'correct_low': [],   # correct prediction + low VLM score
        'wrong_high': [],    # wrong prediction + high VLM score
        'wrong_low': []      # wrong prediction + low VLM score
    }
    
    for result in results:
        is_correct = result['prediction']['prediction'] == result['label']
        is_high_score = result['score'] >= result['threshold']
        
        if is_correct and is_high_score:
            cases['correct_high'].append(result)
        elif is_correct and not is_high_score:
            cases['correct_low'].append(result)
        elif not is_correct and is_high_score:
            cases['wrong_high'].append(result)
        else:
            cases['wrong_low'].append(result)
    
    report = {
        'summary': {
            'correct_high': len(cases['correct_high']),
            'correct_low': len(cases['correct_low']),
            'wrong_high': len(cases['wrong_high']),
            'wrong_low': len(cases['wrong_low']),
            'total': len(results)
        },
        'cases': cases
    }
    
    return report

def process_folder(input_folder,
                   save_dir,
                   method='maskedcam',
                   model_name='resnet18',
                   model=None,
                   api_key=None,
                   threshold=2.5,
                   cam_type='gradcam',
                   vlm_model='gemini/gemini-1.5-flash',
                   prompt=1,
                   slope=25, position=0.6, labels=None, model_type='classification', delay_seconds=7):
    """
    Process all images in a folder using the autoXplain pipeline
    
    Args:
        input_folder (str): Path to folder containing images
        save_dir (str): Path to save processed results
        model_name (str): Model to use (choices: 'resnet18', 'maxvit_t', ..., default: 'resnet18')
        api_key (str): Google API key for Gemini model
        threshold (float): Threshold for VLM score (higher means good, lower means bad)
        cam_type (str): Type of CAM method to use
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Initialize models
    if model is None:
        model_ = models
        if model_type != 'classification':
            model_ = getattr(models, model_type)
        if hasattr(model_, model_name):
            model = getattr(model_, model_name)(pretrained=True).eval()
        else:
            raise ValueError(f"Model must be one of {', '.join(model_.__dict__.keys())}")
        
    if labels is None:
        url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
        with urllib.request.urlopen(url) as response:
            class_idx = json.load(response)
        labels = [class_idx[str(i)][1] for i in range(1000)]

    # Initialize LLM
    api_tokens = api_key.split(',') if api_key else []
    bot = LLMInference(vlm_model, api_key)

    # Process all images in folder
    results = []

    print(f"Processing {len(list(Path(input_folder).glob('*')))} images... in {input_folder}")

    total = len(list(Path(input_folder).glob('*')))
    for count, image_file in enumerate(Path(input_folder).glob('*')):
        if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            # Extract label from filename (format: id_label.extension)
            filename = image_file.stem
            try:
                label = filename.split('_', 1)[1]  # Get everything after first underscore
            except IndexError:
                print(f"Warning: Could not extract label from filename {filename}, set label to 'unk'")
                label = 'unk'
                
            result = process_image(
                image_path=str(image_file),
                save_dir=save_dir,
                bot=bot,
                model=model,
                labels=labels,
                api_tokens=api_tokens,
                method=method,
                cam_type=cam_type, prompt=prompt, slope=slope, position=position, model_type=model_type, delay_seconds=delay_seconds)
            
            if result is None: continue
            result['label'] = label
            result['threshold'] = threshold
            results.append(result)
            print(f"{count}/{total}\t| {image_file} - Score: {result['score']}, Prediction: {result['prediction']}, Label: {label}")

    # Generate final report
    report = generate_final_report(results)
    
    # Save report
    report_path = Path(save_dir) / 'final_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    return results, report

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process images using autoXplain pipeline')
    parser.add_argument('input_folder', help='Path to folder containing images')
    parser.add_argument('--method', default='maskedcam', help='Method to use', choices=METHODS.keys())
    parser.add_argument('--save_dir', default='autoXplain_results', help='Path to save processed results')
    parser.add_argument('--model', default='resnet18', help='Model to use')
    parser.add_argument('--api_key', help='Google API key for Gemini model')
    parser.add_argument('--threshold', type=float, default=2.5, help='Threshold for VLM score (higher means good)')
    parser.add_argument('--vlm_model', type=str, default='gemini/gemini-1.5-flash', help='VLM model to use')
    parser.add_argument('--prompt', type=int, default=1, help='Prompt to use, from 1 to 3')
    parser.add_argument('--cam_type', default='gradcam', 
                      choices=['gradcam', 'smoothgradcam', 'gradcamplusplus', 'cam', 
                              'scorecam', 'layercam', 'xgradcam'],
                      help='Type of CAM method to use')
    
    args = parser.parse_args()
    
    results, report = process_folder(args.input_folder, args.save_dir, args.model, 
                                   args.api_key, args.threshold, args.cam_type, args.vlm_model, args.prompt, method=args.method)
    print(f"\nProcessed {len(results)} images")
    print(f"Results saved to {args.save_dir}")
    print("\nFinal Report Summary:")
    print(f"Correct predictions with high VLM score: {report['summary']['correct_high']}")
    print(f"Correct predictions with low VLM score: {report['summary']['correct_low']}")
    print(f"Wrong predictions with high VLM score: {report['summary']['wrong_high']}")
    print(f"Wrong predictions with low VLM score: {report['summary']['wrong_low']}") 