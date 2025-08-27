import json, os, shutil, hashlib
import gradio as gr
from pathlib import Path

from autoXplain.evaluating import CamJudge
from FlowDesign.litellm import LLMInference
from torchcam.methods import GradCAM
from torchvision.models import resnet18, maxvit_t
import urllib.request
from time import sleep
import matplotlib
matplotlib.use('Agg')

# Initialize model and labels once
url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
with urllib.request.urlopen(url) as response:
    class_idx = json.load(response)
labels = [class_idx[str(i)][1] for i in range(1000)]

api_tokens = [os.getenv('GOOGLE_API')]
API_INDEX = 0

bot = LLMInference("gemini/gemini-1.5-flash", os.getenv('GOOGLE_API'))
model_resnet = resnet18(pretrained=True).eval()
model_maxvit_t = maxvit_t(pretrained=True).eval()

def get_image_hash(image_path):
    """Generate MD5 hash for image content"""
    with open(image_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def create_directory_structure(save_dir):
    """Create necessary subdirectories in save directory"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for subdir in ['original', 'saliency', 'maskedcam', 'metadata']:
        (Path(save_dir) / subdir).mkdir(exist_ok=True)

def process_image(image_path, save_dir):
    global API_INDEX
    """Process image with structured saving"""
    create_directory_structure(save_dir)
    image_hash = get_image_hash(image_path)
    metadata_path = Path(save_dir) / 'metadata' / f"{image_hash}.json"

    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    
    # Copy original image
    original_dest = Path(save_dir) / 'original' / f"{image_hash}.jpg"
    shutil.copy(image_path, original_dest)
    
    # Process image
    result = None
    if 'resnet18' in image_path:
        agent = CamJudge(bot, GradCAM, model_resnet, labels=labels, slope=25, position=0.6)
    elif 'maxvit_t' in image_path:
        agent = CamJudge(bot, GradCAM, model_maxvit_t, labels=labels, slope=25, position=0.6)
    else: raise Exception('MODEL not existed')

    for _ in range(7):
        try:
            sleep(8) # prevent exceed quota
            if len(api_tokens) != 0:
                API_INDEX = API_INDEX % len(api_tokens)
                agent.bot.api_key = api_tokens[API_INDEX]
                API_INDEX += 1
            result = agent({'image': image_path, 'label': 'unk'})
            break
        except Exception as e:
            print(e)
    
    if not result:
        result = {
            'original': image_path,
            'image': str(original_dest),
            'label': 'unk',
            'description': '',
            'justification': '',
            'score': -1,
            'prediction': 'unk',
            'saliency': '',
            'maskedcam': ''
        }
    else:
        # Save processed images
        saliency_path = Path(save_dir) / 'saliency' / f"{image_hash}.jpg"
        result['saliency'].save(saliency_path)
        maskedcam_path = Path(save_dir) / 'maskedcam' / f"{image_hash}.jpg"
        result['maskedcam'].save(maskedcam_path)
        
        result.update({
            'original': image_path,
            'saliency': str(saliency_path),
            'maskedcam': str(maskedcam_path),
            'image': str(original_dest)
        })
        
    result['image_hash'] = image_hash
    
    with open(metadata_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    return result

def save_decision(result, decision, save_dir):
    """Save user decision to image metadata"""
    metadata_path = Path(save_dir) / 'metadata' / f"{result['image_hash']}.json"
    result['decision'] = decision
    with open(metadata_path, 'w') as f:
        json.dump(result, f, indent=2)

def load_existing_results(save_dir):
    """Load all processed results from save directory"""
    metadata_dir = Path(save_dir) / 'metadata'
    if not metadata_dir.exists():
        return []
    
    results = []
    for meta_file in metadata_dir.glob('*.json'):
        with open(meta_file, 'r') as f:
            result = json.load(f)
            results.append(result)
    
    return sorted(results, key=lambda x: x['image'])

def find_first_undecided(results):
    """Find first image without decision"""
    for i, res in enumerate(results):
        if 'decision' not in res:
            return i
    return 0

def create_ui():
    """Create Gradio interface with save directory support"""
    with gr.Blocks() as demo:
        state = gr.State({'results': [], 'current_index': 0})
        
        with gr.Row():
            save_dir_input = gr.Textbox(label="Save Directory", value="autoXplain_data")
            load_btn = gr.Button("Load Existing Results", variant="primary")
        
        with gr.Row():
            folder_input = gr.File(label="Image Folder", file_count="directory")
        
        with gr.Row():
            original = gr.Image(label="Original Image")
            saliency = gr.Image(label="Saliency Map")
            maskedcam = gr.Image(label="Masked CAM")
        
        with gr.Row():
            desc = gr.Textbox(label="Description", interactive=False)
            justify = gr.Textbox(label="Justification", interactive=False)
            score = gr.Number(label="Score", interactive=False)
        
        with gr.Row():
            accept_btn = gr.Button("Accept", variant="primary")
            reject_btn = gr.Button("Reject", variant="secondary")
            counter = gr.Label("0/0")

        def update_display(index, results):
            if not results or index >= len(results):
                return [None]*3 + ["", "", 0, "0/0"]
            
            result = results[index]
            return [
                result['image'],
                result['saliency'],
                result['maskedcam'],
                result['description'],
                result['justification'],
                result['score'],
                f"Prediction: {result['prediction']}\nLabel: {result['label']}\n{index+1}/{len(results)}"
            ]
        
        def handle_folder(folder, save_dir, state):
            results = []
            if folder:
                results = [process_image(p, save_dir) for p in folder]
            
            current_index = find_first_undecided(results)
            return [{'results': results, 'current_index': current_index}] + update_display(current_index, results)
        
        def handle_decision(state, save_dir, decision):
            if not state['results']:
                return [state] + update_display(0, [])
            
            current_index = state['current_index']
            result = state['results'][current_index]
            save_decision(result, decision, save_dir)
            
            # Update current index
            new_index = current_index + 1
            if new_index >= len(state['results']):
                new_index = current_index
            
            # Find next undecided image
            for i in range(new_index, len(state['results'])):
                if 'decision' not in state['results'][i]:
                    new_index = i
                    break
            
            new_state = {'results': state['results'], 'current_index': new_index}
            return [new_state] + update_display(new_index, state['results'])
        
        def handle_load(save_dir):
            results = load_existing_results(save_dir)
            current_index = find_first_undecided(results)
            return [{'results': results, 'current_index': current_index}] + update_display(current_index, results)

        # Event handlers
        load_btn.click(
            handle_load,
            [save_dir_input],
            [state] + [original, saliency, maskedcam, desc, justify, score, counter]
        )
        
        folder_input.change(
            handle_folder,
            [folder_input, save_dir_input, state],
            [state] + [original, saliency, maskedcam, desc, justify, score, counter]
        )
        
        accept_btn.click(
            lambda s, sd: handle_decision(s, sd, 'accept'),
            [state, save_dir_input],
            [state] + [original, saliency, maskedcam, desc, justify, score, counter]
        )
        
        reject_btn.click(
            lambda s, sd: handle_decision(s, sd, 'reject'),
            [state, save_dir_input],
            [state] + [original, saliency, maskedcam, desc, justify, score, counter]
        )
    
    return demo

demo = create_ui()
demo.launch()