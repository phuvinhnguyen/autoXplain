import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Force headless backend to avoid GTK import errors
os.environ.setdefault("MPLBACKEND", "Agg")
try:
    import matplotlib
    matplotlib.use("Agg")
except Exception:
    # If matplotlib is not present or fails to set, ignore
    pass

import gradio as gr
from PIL import Image


def list_samples(base_dir: str) -> List[str]:
    """Return sorted list of metadata json file stems under base_dir/metadata.

    Each stem corresponds to a single sample and is the filename with extension.
    """
    metadata_dir = Path(base_dir) / "metadata"
    if not metadata_dir.exists():
        return []
    files = sorted([p.name for p in metadata_dir.glob("*.json")])
    return files


def load_metadata(base_dir: str, meta_name: str) -> Dict[str, Any]:
    """Load one metadata json by name (e.g., '0_banana.jpg.json')."""
    path = Path(base_dir) / "metadata" / meta_name
    with open(path, "r") as f:
        return json.load(f)


def get_maskedcam_path(base_dir: str, meta: Dict[str, Any]) -> str:
    """Resolve maskedcam image path robustly.

    Tries in order:
    1) as-is if absolute
    2) relative to base_dir
    3) relative to parent of base_dir
    4) base_dir/maskedcam/basename
    5) infer from image_hash with common extensions
    """
    masked = meta.get("maskedcam")
    if isinstance(masked, str) and masked:
        p = Path(masked)
        candidates = []
        if p.is_absolute():
            candidates.append(p)
        else:
            candidates.append(Path(base_dir) / p)
            candidates.append(Path(base_dir).parent / p)
            candidates.append(Path(base_dir) / "maskedcam" / p.name)
        for c in candidates:
            if c.exists():
                return str(c)
        # Fall through to hash-based inference
    # Fallback: infer from metadata filename if present
    image_hash = meta.get("image_hash")
    if image_hash:
        base = Path(base_dir) / "maskedcam"
        # If image_hash already has extension, try directly
        direct = base / image_hash
        if direct.exists():
            return str(direct)
        for ext in [".jpg", ".jpeg", ".png"]:
            c = base / f"{image_hash}{ext if not str(image_hash).lower().endswith(ext) else ''}"
            if c.exists():
                return str(c)
    # As last resort, return empty
    return ""


def get_label_and_prediction(meta: Dict[str, Any]) -> Tuple[str, str]:
    label = meta.get("label", "")
    pred = meta.get("prediction", "")
    if isinstance(pred, dict):
        pred = pred.get("prediction", "")
    return str(label), str(pred)


def save_human_score(base_dir: str, meta_name: str, meta: Dict[str, Any], score: float) -> str:
    """Save human annotation json under base_dir/human_annot with added human_score.

    The output file mirrors the metadata filename.
    """
    out_dir = Path(base_dir) / "human_annot"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / meta_name
    # Shallow copy and append human_score
    data = dict(meta)
    data["human_score"] = score
    # Preserve model justification alongside human score for reference
    data["human_justification"] = meta.get("justification", "")
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    return str(out_path)


def count_progress(base_dir: str) -> Tuple[int, int]:
    total = len(list_samples(base_dir))
    human_dir = Path(base_dir) / "human_annot"
    done = 0
    if human_dir.exists():
        done = len(list(human_dir.glob("*.json")))
    return done, total


def build_ui():
    with gr.Blocks() as demo:
        state = gr.State({
            "base_dir": "",
            "files": [],
            "index": 0,
        })

        gr.Markdown("**Human Annotation Tool** — select the experiment folder containing `metadata/` and `maskedcam/`.")

        with gr.Row():
            base_dir = gr.Textbox(label="Experiment Folder", value="/home/kat/Desktop/FPTAI/autoXplain/examples/raw_experiment/explanation_output_resnet18_20250905_163859")
            load_btn = gr.Button("Load", variant="primary")
            progress = gr.Label(value="0/0")

        with gr.Row():
            img = gr.Image(label="MaskedCAM", interactive=False)
            with gr.Column():
                fname = gr.Textbox(label="Sample", interactive=False)
                label = gr.Textbox(label="Label", interactive=False)
                pred = gr.Textbox(label="Prediction", interactive=False)
                desc = gr.Textbox(label="Description", lines=4, interactive=False)
                justify_view = gr.Textbox(label="Justification", lines=3, interactive=False)

        with gr.Row():
            gr.Markdown("**Choose score:**")
            b0 = gr.Button("0")
            b1 = gr.Button("1")
            b2 = gr.Button("2")
            b3 = gr.Button("3")
            b4 = gr.Button("4")
            b5 = gr.Button("5")

        with gr.Row():
            prev_btn = gr.Button("Prev")
            next_btn = gr.Button("Next")

        def _load_image(path: str):
            try:
                if path and Path(path).exists():
                    return Image.open(path).convert("RGB")
            except Exception:
                return None
            return None

        def do_load(dir_path: str):
            files = list_samples(dir_path)
            idx = 0
            done, total = count_progress(dir_path)
            st = {"base_dir": dir_path, "files": files, "index": idx}
            if not files:
                return st, None, "", "", "", "", f"{done}/{total}", ""
            meta = load_metadata(dir_path, files[idx])
            image_path = get_maskedcam_path(dir_path, meta)
            image_obj = _load_image(image_path)
            lab, pr = get_label_and_prediction(meta)
            return st, image_obj, files[idx], lab, pr, meta.get("description", ""), f"{done}/{total}", meta.get("justification", "")

        def update_view(st: Dict[str, Any]):
            base = st.get("base_dir", "")
            files = st.get("files", [])
            idx = st.get("index", 0)
            if not files:
                done, total = count_progress(base)
                return None, "", "", "", "", f"{done}/{total}", ""
            idx = max(0, min(idx, len(files) - 1))
            meta = load_metadata(base, files[idx])
            image_path = get_maskedcam_path(base, meta)
            image_obj = _load_image(image_path)
            lab, pr = get_label_and_prediction(meta)
            done, total = count_progress(base)
            return image_obj, files[idx], lab, pr, meta.get("description", ""), f"{done}/{total}", meta.get("justification", "")

        def go_next(st: Dict[str, Any]):
            st = dict(st)
            st["index"] = min(len(st.get("files", [])) - 1, st.get("index", 0) + 1)
            return st, *update_view(st)

        def go_prev(st: Dict[str, Any]):
            st = dict(st)
            st["index"] = max(0, st.get("index", 0) - 1)
            return st, *update_view(st)

        def do_save(st: Dict[str, Any], current_score: float):
            base = st.get("base_dir", "")
            files = st.get("files", [])
            idx = st.get("index", 0)
            if not files:
                return st, *update_view(st)
            meta_name = files[idx]
            meta = load_metadata(base, meta_name)
            save_human_score(base, meta_name, meta, float(current_score))
            # After saving, auto-advance if possible
            if idx < len(files) - 1:
                st = dict(st)
                st["index"] = idx + 1
            return st, *update_view(st)

        load_btn.click(
            do_load,
            [base_dir],
            [state, img, fname, label, pred, desc, progress, justify_view],
        )

        next_btn.click(
            go_next,
            [state],
            [state, img, fname, label, pred, desc, progress, justify_view],
        )

        prev_btn.click(
            go_prev,
            [state],
            [state, img, fname, label, pred, desc, progress, justify_view],
        )

        b0.click(lambda s: do_save(s, 0), [state], [state, img, fname, label, pred, desc, progress, justify_view])
        b1.click(lambda s: do_save(s, 1), [state], [state, img, fname, label, pred, desc, progress, justify_view])
        b2.click(lambda s: do_save(s, 2), [state], [state, img, fname, label, pred, desc, progress, justify_view])
        b3.click(lambda s: do_save(s, 3), [state], [state, img, fname, label, pred, desc, progress, justify_view])
        b4.click(lambda s: do_save(s, 4), [state], [state, img, fname, label, pred, desc, progress, justify_view])
        b5.click(lambda s: do_save(s, 5), [state], [state, img, fname, label, pred, desc, progress, justify_view])

    return demo


if __name__ == "__main__":
    app = build_ui()
    # Share false by default; user can change if needed
    app.launch()


