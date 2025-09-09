from .evaluating import *
import re, json

def parse_bot_output(bot_output):
    result = {
        'description': None,
        'score': None,
        'justification': None,
        'bot_output': bot_output
    }
    
    # Find all JSON-like blocks in the text
    dict_pattern = r"\{(?:[^{}])*\}"  # Recursive regex for nested braces if needed (optional)
    matches = re.findall(dict_pattern, bot_output, re.DOTALL)

    for match in matches:
        try:
            output_dict = json.loads(match)
            if all(k in output_dict for k in ["evaluation", "score", "justification"]):
                result['description'] = output_dict['evaluation']
                result['score'] = output_dict['score']
                result['justification'] = output_dict['justification']
                break  # Stop at the first valid match
        except json.JSONDecodeError:
            continue  # Try next match if JSON parsing fails

    return result


class OldMaskedCamJudge(ExtractCAM):
    PROMPT = '''Task: Conduct an evaluation of the model's attention mechanism by analyzing its response to the supplied masked image. This assessment aims to test the model's capacity to effectively interpret and utilize attention when processing partially obscured visual data.

Image Description:

    1. The input image is sourced from a highly reputable dataset widely used in research, known for its quality and reliability. It originates from ImageNet 1K, a standard benchmark in the research community, ensuring it meets all necessary standards for academic and scientific work.
    2. The image is masked using a Grad-CAM heatmap, where only the areas the model is focusing on are visible, while all other regions are blacked out.
    3. The model's focus is on the {object}.
    4. The image may appear mostly black; identify the visible regions (non-black areas) and analyze what those regions represent in relation to the object of interest.

Evaluation Criteria:

    Focus Accuracy: Analyze which part of the image the Grad-CAM is highlighting. Is the model's attention placed accurately on the {object}, or is it scattered across other areas?
    Object Recognition: Determine if the model is correctly recognizing the {object}. Is the attention primarily on the correct object, or does the model focus on irrelevant areas?
    Object Coverage: Evaluate how much of the object is being captured by the model's attention. Is the entire object covered, only a small part, or none at all?
    Background and Irrelevant Focus: Check for any significant focus on the background or irrelevant objects. Does this distract the model from the primary object?
    Explanatory Analysis: Provide possible reasons for the model's attention pattern. Consider whether the model is being misled by similarly shaped or colored objects, complex backgrounds, or other visual challenges.

Scoring:

Assign a score between 0 and 5 based on the relevance and accuracy of the model's attention:

    0: The model's attention is scattered with no clear target, showing it does not understand the task or the object.
    1: The model consistently directs its attention to something unrelated to {object}, indicating a fundamental misunderstanding of the {object} it is supposed to recognize.
    2: Partial object recognition: The model captures only a small fragment of the {object}, missing most of its critical features. The attention is mostly misdirected, with just minor alignment to the actual object.
    3: The model identifies a limited area of {object}, but its attention still includes some irrelevant parts surrounding it.
    4: The model predominantly focuses on {object}, with only minor distractions or irrelevant attention in the background.
    5: The model accurately captures the entire {object} without any distractions from irrelevant areas or background elements.

Output Format:

    Evaluation: Provide a concise evaluation (5-6 sentences), discussing:
        Where the Grad-CAM is focusing.
        Whether the attention aligns with the {object}.
        Whether there is any significant focus on irrelevant areas or the background.
        Explain why the model might be focusing on specific regions.

    Score: Assign a score from 0 to 5, justifying your rating in a sentence.

    Your output format must be presented in a dictionary as follows, which is extremely important for the evaluation process to run without any error:
{{
    "evaluation": [your evaluation],
    "justification": [your justification],
    "score": [score]
}}'''
    modifies = ('description', 'justification', 'score', 'prediction', 'label', 'saliency', 'maskedcam')

    def __init__(self, bot, cam_class, model, layer=0, labels=[], slope=25, position=0.4, prediction_of_preprocesed_image=None, **kwargs):
        super().__init__(cam_class, model, layer, slope, position, **kwargs)
        self.bot = bot
        self.labels = labels
        self.prediction_of_preprocesed_image = prediction_of_preprocesed_image

    def extract_answer(self, text):
        output = parse_bot_output(text)
        return output

    def process(self, image, label):
        if self.prediction_of_preprocesed_image == None:
            if self.model_type != 'classification' and label in self.labels:
                target_class_idx = self.labels.index(label)
            else:
                target_class_idx = None
            saliency, masked_cam, prediction, _ = super().process(image, target_class_idx)
            prediction['prediction'] = self.labels[prediction['class_idx']]
        else:
            saliency, masked_cam = None, image
            prediction = {'prediction': self.prediction_of_preprocesed_image}
        result = self.bot.run([('user', [pil_to_tempfile_path(masked_cam), self.PROMPT.format(object=prediction['prediction'])])])
        result = self.extract_answer(result['content'][0]['text'])
        return result['description'], result['justification'], get_first_number(result['score']), prediction, label, saliency, masked_cam


class OldOriginalCamJudge(ExtractCAM):
    PROMPT = '''Task: Conduct an evaluation of the model's attention mechanism by analyzing its response to the supplied CAM heatmap. This assessment aims to test the model's capacity to effectively interpret and utilize attention when processing visual data.

Image Description:

    1. The input image is a CAM heatmap sourced from a highly reputable dataset widely used in research, known for its quality and reliability. It originates from ImageNet 1K, a standard benchmark in the research community, ensuring it meets all necessary standards for academic and scientific work.
    2. The heatmap uses warm colors (orange, red) to represent areas where the model is focusing most, while cool colors (blue, purple, dark) indicate regions of little to no attention.
    3. The model's focus is on the {object}.
    4. Identify the warm-colored regions and analyze what those regions represent in relation to the object of interest. Additionally, assess the presence of cool-colored regions and their alignment with irrelevant areas or the background.

Evaluation Criteria:

    Focus Accuracy: Analyze which part of the heatmap the warm colors (orange, red) highlight. Is the model's attention accurately placed on the {object}, or is it scattered across other areas?
    Object Recognition: Determine if the model is correctly recognizing the {object}. Is the attention primarily on the correct object, or does the model focus on irrelevant areas?
    Object Coverage: Evaluate how much of the object is being captured by the model's attention. Is the entire object covered, only a small part, or none at all?
    Background and Irrelevant Focus: Check for any significant focus on cool-colored regions. Does this distract the model from the primary object?
    Explanatory Analysis: Provide possible reasons for the model's attention pattern. Consider whether the model is being misled by similarly colored areas, complex backgrounds, or other visual challenges.

Scoring:

Assign a score between 0 and 5 based on the relevance and accuracy of the model's attention:

    0: The model's attention is scattered with no clear target, showing it does not understand the task or the object.
    1: The model consistently directs its attention to something unrelated to {object}, indicating a fundamental misunderstanding of the {object} it is supposed to recognize.
    2: Partial object recognition: The model captures only a small fragment of the {object}, missing most of its critical features. The attention is mostly misdirected, with just minor alignment to the actual object.
    3: The model identifies a limited area of {object}, but its attention still includes some irrelevant parts surrounding it.
    4: The model predominantly focuses on {object}, with only minor distractions or irrelevant attention in the background.
    5: The model accurately captures the entire {object} without any distractions from irrelevant areas or background elements.

Output Format:

    Evaluation: Provide a concise evaluation (5-6 sentences), discussing:
        Where the heatmap focuses (warm colors).
        Whether the attention aligns with the {object}.
        Whether there is any significant focus on irrelevant areas or the background.
        Explain why the model might be focusing on specific regions.

    Score: Assign a score from 0 to 5, justifying your rating in a sentence.

    Your output format must be presented in a dictionary as follows, which is extremely important for the evaluation process to run without any error:
{{
    "evaluation": [your evaluation],
    "justification": [your justification],
    "score": [score]
}}'''
    modifies = ('description', 'justification', 'score', 'prediction', 'label', 'saliency', 'maskedcam')

    def __init__(self, bot, cam_class, model, layer=0, labels=[], slope=25, position=0.4, prediction_of_preprocesed_image=None, **kwargs):
        super().__init__(cam_class, model, layer, slope, position, **kwargs)
        self.bot = bot
        self.labels = labels
        self.prediction_of_preprocesed_image = prediction_of_preprocesed_image

    def process(self, image, label):
        if self.prediction_of_preprocesed_image == None:
            if self.model_type != 'classification' and label in self.labels:
                target_class_idx = self.labels.index(label)
            else:
                target_class_idx = None
            saliency, masked_cam, prediction, _ = super().process(image, target_class_idx)
            prediction['prediction'] = self.labels[prediction['class_idx']]
        else:
            saliency, masked_cam = image, None
            prediction = {'prediction': self.prediction_of_preprocesed_image}
        result = self.bot.run([('user', [pil_to_tempfile_path(saliency), self.PROMPT.format(object=prediction['prediction'])])])
        result = self.extract_answer(result['content'][0]['text'])
        return result['description'], result['justification'], get_first_number(result['score']), prediction, label, saliency, masked_cam

    def extract_answer(self, text):
        return parse_bot_output(text)