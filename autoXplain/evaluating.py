from .xtractcam import ExtractCAM
from .tools import *

PROMPT_1 = '''\
Read the masked CAM image and use this tool to evaluate the model's attention mechanism. This task is designed to assess how well the model can interpret and utilize attention when visual data is partially obscured. You should notice that the model can be very weak and its predition ({object}) might be wrong, which might make its attention irrelevant to the {object}. For that reason, you should check if the image is highly relevant to the {object} or not, if it is not too relevant or hard to see the {object}, you should give a very low score.

A masked Grad-CAM image is an image in which only the regions the model focuses on (as determined by a Grad-CAM heatmap) are visible. All other areas of the original image have been blacked out. This allows us to isolate and visualize what parts of the image the model considers important when making its prediction. The model's prediction given its attention in this image is {object}, you should consider this information to evaluate the image understanding ability of the model.

To ensure the program work flawlessly, your answer must call a function follow exactly this template:
<function>
<parameter description>
- Describe the visible {object} in the masked image.
- How much of the {object} is highlighted (partially, fully, or not at all)?
- Do these visible areas align well with the {object}?
- Is the model focusing on any irrelevant regions or background (limited attention to the {object})?
- Why might the model be attending to those specific regions (e.g., similarity, shape, color, distractions)?
</parameter>
<parameter justification>
Based on the attention pattern observed, provide a clear reason for the score you assign.
</parameter>
<parameter score>
Give a single number, score from 0 to 5:
- 0: Completely scattered or irrelevant attention
- 1: Consistently focuses on the wrong object.
- 2: Minor overlap with the object, mostly incorrect focus.
- 3: Some correct focus on the object, but also includes irrelevant regions.
- 4: Mostly correct focus, with minor background distractions.
- 5: Fully accurate attention on the object, no distractions.
</parameter>
</function>

<IMPORTANT>
- The model considers the masked image as proof of {object}. Debate based on this.
- Function calls MUST follow the specified format, start with <function and end with </function>.
- Required parameters MUST follow the specified format, start with <parameter example_parameter> and end with </parameter>.
- You can only call one function each turn.
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after.
'''

PROMPT_2 = '''\
You are given a masked Grad-CAM image, where only the areas highlighted by the model’s attention remain visible — the rest has been blacked out. These visible areas represent what the model considers important when making a prediction.

The model predicts that the object in the image is a {object}. Your task is to evaluate how accurately the model's attention (as shown in the visible regions) supports this prediction.

To complete the task, use the following function template to return your evaluation:

<function>
<parameter description>
- Describe the visible {object} in the image.
- Is the {object} partially, fully, or not highlighted?
- Do the highlighted regions match the true shape/location of the {object}?
- Are there irrelevant or distracting regions included?
- Why do you think the model is focusing on those areas?
</parameter>
<parameter justification>
Explain your reasoning for the score based on the observed attention pattern.
</parameter>
<parameter score>
Rate the explanation quality from 0 to 5:
- 0: Attention is completely off-target.
- 1: Focused on wrong object.
- 2: Very weak alignment with the correct object.
- 3: Some useful attention but many irrelevant areas.
- 4: Mostly good attention with minor issues.
- 5: Excellent, fully relevant attention.
</parameter>
</function>

<IMPORTANT>
- Only one function call is allowed per turn.
- Use the exact structure as shown above.
- You may include optional reasoning *before* the function call in natural language, but not after.
'''

PROMPT_3 = '''\
You are presented with a masked Grad-CAM image, where only regions highlighted by the model’s attention remain visible — the rest of the image is blacked out. The prediction made by the model is: {object}.

Before making any evaluation, please begin by describing what you see in the masked image. Then, assess whether the visible areas align well with the object "{object}" and how effectively the model focuses its attention.

You must return your evaluation using this exact function format:

<function>
<parameter description>
- First, describe what you see in the masked image — what visible parts of the {object} can be observed?
- How much of the {object} is visible and highlighted (fully, partially, or not at all)?
- Do the highlighted regions correspond accurately to the actual {object}?
- Are there irrelevant parts or distractions in the focus?
- Why might the model be attending to these specific areas?
</parameter>
<parameter justification>
Provide a justification for your score based on the attention patterns and what you observed.
</parameter>
<parameter score>
Assign a score from 0 to 5:
- 0: Attention completely misses the object.
- 1: Focus is entirely on the wrong region.
- 2: Minimal overlap with the actual object.
- 3: Mixed relevance — some object focus, some irrelevant.
- 4: Mostly correct with minor errors.
- 5: Excellent attention — fully covers the object without distractions.
</parameter>
</function>

<IMPORTANT>
- Begin by visually describing the masked image.
- The function format must follow the template exactly.
- Only one function call is allowed per turn.
- Reasoning may be included before the function call, but not after.
'''

class CamJudge(ExtractCAM):
    modifies = ('description', 'justification', 'score', 'prediction', 'label', 'saliency', 'maskedcam')
    PROMPT = PROMPT_1

    def __init__(self, bot, cam_class, model, layer=0, labels=[], slope=25, position=0.4, prediction_of_preprocesed_image=None, **kwargs):
        '''
        prediction_of_preprocesed_image: None means there are no preprocessed image, the method will extract CAM from beginning
        '''
        super().__init__(cam_class, model, layer, slope, position, **kwargs)
        self.bot = bot
        self.labels = labels
        self.prediction_of_preprocesed_image = prediction_of_preprocesed_image
        
    def extract_answer(self, text):
        return extract_function_calls(text)

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
        result = self.extract_answer(result['content'][0]['text'])[-1]
        return result['args']['description'], result['args']['justification'], get_first_number(result['args']['score']), prediction, label, saliency, masked_cam
