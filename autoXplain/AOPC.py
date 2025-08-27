from .xtractcam import *
from .tools import *
from torchcam import *

class AOPC(ExtractCAM):
    modifies = ('score', 'prediction', 'saliency')

    def __init__(self, cam_class, model, layer=0, labels=[], steps=30, **kwargs):
        super().__init__(cam_class, model, layer, **kwargs)
        self.labels = labels
        self.model.eval()
        self.device = next(model.parameters()).device
        self.steps = steps

    def process(self, image):
        # Step 1: Load & preprocess
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

        # Step 2: forward pass + CAM
        cam_extractor = self.cam_class(self.model, None)
        output = self.model(img_tensor)
        class_idx = output.argmax().item()
        prediction = self.labels[class_idx] if self.labels else str(class_idx)
        full_score = torch.softmax(output, dim=1)[0, class_idx].item()

        cam = cam_extractor(class_idx, output)[0].squeeze().cpu().numpy()
        cam_extractor.remove_hooks()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam_resized = np.array(Image.fromarray(cam).resize((224, 224), resample=Image.BICUBIC))

        # Step 3: sort pixels by relevance
        flat = cam_resized.flatten()
        idx_sorted = np.argsort(-flat)
        h, w = cam_resized.shape
        total_pixels = h * w

        scores = []
        for k in range(self.steps + 1):
            keep = int(total_pixels * (self.steps - k) / self.steps)
            mask = torch.zeros_like(torch.from_numpy(cam_resized.flatten()))
            if keep > 0:
                mask[idx_sorted[:keep]] = 1
            mask = mask.reshape(h, w)
            mask_3ch = mask.unsqueeze(0).repeat(3, 1, 1).to(self.device)
            xk = img_tensor * mask_3ch

            with torch.no_grad():
                outk = self.model(xk)
                score_k = torch.softmax(outk, dim=1)[0, class_idx].item()
            scores.append(score_k)

        # Step 4: compute AOPC
        # sum over steps of (full_score - score_k)
        diffs = [full_score - sk for sk in scores]
        aopc = float(np.mean(diffs))

        # Visualization
        heatmap = Image.fromarray(cam_resized.astype(np.float32), mode='F').resize(orig_size, Image.BICUBIC)
        saliency = overlay_mask(img, heatmap, alpha=0.5)

        # Optionally, return also the curve
        return aopc, prediction, saliency
