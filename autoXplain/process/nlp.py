from autoXplain.process.base import *
from typing import List, Dict, Any
import json, os, shutil


@process
class NLPProcess(BaseProcess):
    """Process class for NLP explainers (e.g. NLPAgent).

    Config fields (in the ``datasets`` entry):

    .. code-block:: yaml

        - name: NLPProcess
          ds_name: my_nlp_run
          model:
            name: empty           # NLP models use the empty registry model
          questions:              # optional — overrides explain.kwargs.questions
            - "What is the architecture of this model?"
          data_path: /path/to/task.jsonl   # optional: pre-existing JSONL file
          data:                            # optional: inline list of dicts
            - {text: "Hello world", label: positive}

    The JSONL data (either from ``data_path`` or ``data``) is written to
    ``{linked_host_dir}/task.jsonl`` so the agent can access it in the
    sandbox at ``{sandbox_mount_dir}/task.jsonl``.
    """

    def _process(self, ds_cfg: Dict, exp) -> List[Dict[str, Any]]:
        # Write task data into the explainer's linked_host_dir if provided
        linked_host_dir = getattr(exp, "linked_host_dir", None)
        if linked_host_dir:
            self._write_task_data(ds_cfg, linked_host_dir)

        # Questions can come from dataset config or fall back to explainer defaults
        questions = ds_cfg.get("questions", None)
        inputs: Dict[str, Any] = {}
        if questions:
            inputs["questions"] = questions

        out = exp.explain(inputs)  # returns dict-of-lists

        if not out or not isinstance(list(out.values())[0], list):
            return [out]

        # Convert dict-of-lists → list-of-dicts
        return [dict(zip(out.keys(), vals)) for vals in zip(*out.values())]

    # ------------------------------------------------------------------

    def _write_task_data(self, ds_cfg: Dict, linked_host_dir: str) -> None:
        """Write task.jsonl and a ready-to-run inference script into linked_host_dir."""
        task_path = os.path.join(linked_host_dir, "task.jsonl")

        if "data_path" in ds_cfg:
            src = ds_cfg["data_path"]
            if os.path.abspath(src) != os.path.abspath(task_path):
                shutil.copy2(src, task_path)

        elif "data" in ds_cfg:
            with open(task_path, "w") as f:
                for item in ds_cfg["data"]:
                    f.write(json.dumps(item) + "\n")

        # Write a ready-to-run inference script so agents don't need to guess paths.
        # Always overwrite so the latest version is used.
        script_path = os.path.join(linked_host_dir, "run_inference.py")
        with open(script_path, "w") as f:
            f.write(
                "# Suppress noisy progress bars and warnings\n"
                "import warnings, logging, os\n"
                "warnings.filterwarnings('ignore')\n"
                "logging.disable(logging.CRITICAL)\n"
                "os.environ['TOKENIZERS_PARALLELISM'] = 'false'\n"
                "os.environ['TRANSFORMERS_VERBOSITY'] = 'error'\n"
                "os.environ['TQDM_DISABLE'] = '1'\n"
                "\n"
                "import json\n"
                "from transformers import pipeline\n"
                "pipe = pipeline('text-generation', model='/workspace',"
                " trust_remote_code=True, max_new_tokens=10, device='cpu')\n"
                "with open('/workspace/task.jsonl') as fh:\n"
                "    items = [json.loads(l) for l in fh]\n"
                "correct = 0\n"
                "for item in items:\n"
                "    prompt = ('Classify sentiment as positive, negative, or neutral.\\n'\n"
                "              'Text: ' + item['text'] + '\\nAnswer:')\n"
                "    out = pipe(prompt)[0]['generated_text'][len(prompt):].strip().lower()\n"
                "    pred = ('positive' if 'positive' in out\n"
                "            else ('negative' if 'negative' in out else 'neutral'))\n"
                "    if pred == item['label']: correct += 1\n"
                "    print(item['label'], '->', pred)\n"
                "print('Accuracy:', correct, '/', len(items))\n"
            )
