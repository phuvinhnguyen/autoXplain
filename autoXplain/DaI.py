from .xtractcam import *
from .tools import *
from torchcam import *
from sklearn.metrics import auc

class DaI(ExtractCAM):
    modifies = ('score', 'prediction', 'saliency')

    def __init__(self, cam_class, model, layer=0, labels=[], steps=30, **kwargs):
        super().__init__(cam_class, model, layer, **kwargs)
        self.labels = labels
        self.model.eval()
        self.device = next(model.parameters()).device
        self.steps = steps

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

        # --- Step 3: Generate CAM ---
        cam = cam_extractor(class_idx, output)[0].squeeze().cpu().numpy()
        cam_extractor.remove_hooks()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_resized = np.array(Image.fromarray(cam).resize((224, 224), resample=Image.BICUBIC))

        # --- Step 4: Overlay CAM ---
        heatmap = Image.fromarray(cam_resized.astype(np.float32), mode='F').resize(orig_size, Image.BICUBIC)
        saliency = overlay_mask(img, heatmap, alpha=0.5)

        # --- Step 5: AUC evaluation ---
        flat = cam_resized.flatten()
        idx_sorted = np.argsort(-flat)
        h, w = cam_resized.shape
        total_pixels = h * w
        steps = self.steps
        x = np.linspace(0, 1, steps + 1)

        insertion_scores, deletion_scores = [], []
        inserted = torch.zeros_like(img_tensor)
        deleted = img_tensor.clone()

        for step in range(steps + 1):
            k = int(total_pixels * step / steps)
            mask = torch.zeros_like(torch.from_numpy(cam_resized.flatten()))
            if k > 0:
                mask[idx_sorted[:k]] = 1
            mask = mask.reshape(h, w)
            mask_3ch = mask.unsqueeze(0).repeat(3, 1, 1).to(self.device)

            # Insertion
            inserted_img = img_tensor * mask_3ch + inserted * (1 - mask_3ch)
            with torch.no_grad():
                score = torch.softmax(self.model(inserted_img), dim=1)[0, class_idx].item()
            insertion_scores.append(score)

            # Deletion
            deleted_img = img_tensor * (1 - mask_3ch)
            with torch.no_grad():
                score = torch.softmax(self.model(deleted_img), dim=1)[0, class_idx].item()
            deletion_scores.append(score)

        # --- Step 6: Compute AUC score ---
        del_auc = auc(x, deletion_scores)
        ins_auc = auc(x, insertion_scores)
        final_score = (ins_auc + (1 - del_auc)) / 2  # normalized: higher = better

        return final_score, prediction, saliency
