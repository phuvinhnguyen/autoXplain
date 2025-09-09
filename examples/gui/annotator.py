import os
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple

import gradio as gr


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
    """Resolve maskedcam image path. Prefer explicit path in json; otherwise build from stem."""
    if isinstance(meta.get("maskedcam"), str) and meta["maskedcam"]:
        # If path is relative, resolve from base_dir
        maskedcam_path = Path(meta["maskedcam"])  # may be relative like '.../maskedcam/xxx.jpg'
        if not maskedcam_path.is_absolute():
            maskedcam_path = Path(base_dir) / maskedcam_path
        return str(maskedcam_path)
    # Fallback: infer from metadata filename if present
    image_hash = meta.get("image_hash")
    if image_hash:
        candidate = Path(base_dir) / "maskedcam" / f"{image_hash}"
        if candidate.exists():
            return str(candidate)
        # Try common image extensions
        for ext in [".jpg", ".png", ".jpeg"]:
            c = candidate.with_suffix(ext)
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

        with gr.Row():
            score = gr.Slider(0, 5, value=0, step=1, label="Human Score (0-5)")
            save_btn = gr.Button("Save Score", variant="primary")

        with gr.Row():
            prev_btn = gr.Button("Prev")
            next_btn = gr.Button("Next")

        def do_load(dir_path: str):
            files = list_samples(dir_path)
            idx = 0
            done, total = count_progress(dir_path)
            st = {"base_dir": dir_path, "files": files, "index": idx}
            if not files:
                return st, None, "", "", "", "", f"{done}/{total}"
            meta = load_metadata(dir_path, files[idx])
            image_path = get_maskedcam_path(dir_path, meta)
            lab, pr = get_label_and_prediction(meta)
            return st, image_path, files[idx], lab, pr, meta.get("description", ""), f"{done}/{total}"

        def update_view(st: Dict[str, Any]):
            base = st.get("base_dir", "")
            files = st.get("files", [])
            idx = st.get("index", 0)
            if not files:
                done, total = count_progress(base)
                return None, "", "", "", "", f"{done}/{total}"
            idx = max(0, min(idx, len(files) - 1))
            meta = load_metadata(base, files[idx])
            image_path = get_maskedcam_path(base, meta)
            lab, pr = get_label_and_prediction(meta)
            done, total = count_progress(base)
            return image_path, files[idx], lab, pr, meta.get("description", ""), f"{done}/{total}"

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
            [state, img, fname, label, pred, desc, progress],
        )

        next_btn.click(
            go_next,
            [state],
            [state, img, fname, label, pred, desc, progress],
        )

        prev_btn.click(
            go_prev,
            [state],
            [state, img, fname, label, pred, desc, progress],
        )

        save_btn.click(
            do_save,
            [state, score],
            [state, img, fname, label, pred, desc, progress],
        )

    return demo


if __name__ == "__main__":
    app = build_ui()
    # Share false by default; user can change if needed
    app.launch()


