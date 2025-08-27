from .xtractcam import *
from .tools import *
from torchcam import *

class AverageDrop(ExtractCAM):
    modifies = ('score', 'prediction', 'saliency')

    def __init__(self, cam_class, model, layer=0, labels=[], **kwargs):
        super().__init__(cam_class, model, layer, **kwargs)
        self.labels = labels
        self.model.eval()
        self.device = next(model.parameters()).device

    def process(self, image):
        # --- Step 1: Load & preprocess image ---
        if isinstance(image, str):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")
        orig_size = img.size

        img_tensor = normalize(
            to_tensor(resize(img, (224, 224))),
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        ).unsqueeze(0).to(self.device)
        img_tensor.requires_grad_()

        # --- Step 2: Setup CAM + Forward ---
        cam_extractor = self.cam_class(self.model, None)
        output = self.model(img_tensor)
        class_idx = output.argmax().item()
        prediction = self.labels[class_idx] if self.labels else str(class_idx)

        # Score on full image
        full_score = torch.softmax(output, dim=1)[0, class_idx].item()

        # --- Step 3: Generate CAM ---
        cam = cam_extractor(class_idx, output)[0].squeeze().cpu().numpy()
        cam_extractor.remove_hooks()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_resized = np.array(Image.fromarray(cam).resize((224, 224), resample=Image.BICUBIC))

        # --- Step 4: Apply CAM as mask ---
        mask = torch.from_numpy(cam_resized).unsqueeze(0).repeat(3, 1, 1).to(self.device)
        masked_img = img_tensor * mask  # highlight important region only

        with torch.no_grad():
            masked_output = self.model(masked_img)
            masked_score = torch.softmax(masked_output, dim=1)[0, class_idx].item()

        # --- Step 5: Compute Average Drop ---
        drop = max(0, full_score - masked_score)
        avg_drop = 100.0 * drop / (full_score + 1e-8)  # avoid div by 0

        # --- Step 6: Visualization ---
        heatmap = Image.fromarray(cam_resized.astype(np.float32), mode='F').resize(orig_size, Image.BICUBIC)
        saliency = overlay_mask(img, heatmap, alpha=0.5)

        return -avg_drop, prediction, saliency
