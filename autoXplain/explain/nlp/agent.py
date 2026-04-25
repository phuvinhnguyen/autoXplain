"""NLPAgent — ReACT-style NLP explainer with main and sub-agent loops.

Architecture
------------
Main agent (``_react``):
    Full action set: ``bash``, ``ask``, ``batch``, ``submit``.
    Orchestrates the explanation by delegating focused investigations to the
    sub-agent via ``ask``, running batch inference via ``batch``, and finally
    submitting the answer.

Sub-agent (``_ask``):
    Restricted to ``bash`` tool use and free-form text generation only.
    Cannot call ``ask`` or ``batch``.  Returns its accumulated reasoning as
    the answer when no more bash commands are needed, or immediately on
    ``submit``.

Model setup
-----------
Pass ``model_source`` as either:

* A **HuggingFace hub ID** (e.g. ``"OvercastLab/Quark-50m-Instruct"``) —
  the model is downloaded into ``linked_host_dir``, replacing any existing
  model files there.
* A **local directory path** — its contents are copied into ``linked_host_dir``.
* ``None`` — the model is assumed to already be present in ``linked_host_dir``.

Available actions (main agent):
    bash   — run any shell command
    ask    — delegate a sub-question to the sub-agent
    batch  — answer all entries in {model_dir}/task.jsonl via the VLM
    submit — submit the final answer and end the loop

Available tools (sub-agent):
    bash   — run any shell command
    text   — generate free-form analysis text (no code block needed)
    submit — optional explicit early termination
"""

from autoXplain.explain.nlp.base import BaseNLPExplainer, nlp
from autoXplain.utils.vlm import VLM_REGISTRY
from typing import Optional, Sequence, Dict, Any, List
import os, shutil, re, json

DEFAULT_TOOL_FOLDER = os.path.join(os.path.dirname(__file__), "bash")

_MAX_OBS_CHARS = 1500  # truncate long bash outputs to keep prompts short


_RESULT_PATTERNS = re.compile(
    r"Accuracy:\s*\d+\s*/\s*\d+",
    re.IGNORECASE,
)


def _looks_like_final_result(obs: str) -> bool:
    """Return True if the observation contains a complete inference result."""
    return bool(_RESULT_PATTERNS.search(obs))


def _truncate_obs(obs: str, limit: int = _MAX_OBS_CHARS) -> str:
    """Truncate observation, keeping the tail (where results usually appear)."""
    if len(obs) <= limit:
        return obs
    # Keep only a small prefix for context, then the full tail
    prefix = 120
    tail = limit - prefix
    return obs[:prefix] + f"\n...[{len(obs) - limit} chars truncated]...\n" + obs[-tail:]

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_MAIN_SYSTEM_PROMPT = """\
You are an expert AI researcher analyzing the machine learning model at: {model_dir}

## Research Question
{question}

## STRICT FORMAT RULES — YOU MUST FOLLOW THESE EXACTLY
Every response MUST end with exactly one code block. No exceptions.
Write your reasoning first (Thought), then a code block:

Run a shell command:
```bash
<shell command here>
```

Delegate a sub-question to a helper:
```action
ask <focused sub-question>
```

Run the model on task.jsonl (answers written back to the file):
```action
batch
```

When you have gathered enough evidence and have a confident answer:
```action
submit <your complete answer here>
```

NEVER write "Action:" in plain text. ALWAYS use the code block format above.
If you have nothing more to explore, use submit.

## Additional CLI Tools (callable from bash)
{tools_block}

## Instructions
1. First, run `ls {model_dir}` and `cat {model_dir}/config.json` to understand the model.
2. If `{model_dir}/run_inference.py` exists, run `python3 {model_dir}/run_inference.py` to evaluate the model on task.jsonl.
3. Actually load and run the model using Python to get real evidence.
4. Then submit your answer with concrete numbers and observations.
"""

_SUBAGENT_SYSTEM_PROMPT = """\
You are a focused AI research assistant. Answer this single question:
{question}

The model is located at: {model_dir}

## STRICT FORMAT — follow exactly
Write reasoning first, then end with one code block per response:

Run a shell command:
```bash
<command>
```

When done:
```action
submit <your answer>
```

NEVER write "Action:" in plain text. ALWAYS use a code block.

## Additional CLI Tools (callable from bash)
{tools_block}
"""

_STEP_TEMPLATE = """\
{thought}
```{action_type}
{action_content}
```
Observation: {observation}

"""


@nlp
class NLPAgent(BaseNLPExplainer):
    """ReACT-style NLP agent that explains a model via interactive sandbox execution.

    Parameters
    ----------
    vlm:
        VLM configuration dict: ``{"name": "VLLMClient", "kwargs": {...}}``.
    linked_host_dir:
        Host directory that is mounted into the sandbox as the workspace.
        Model files and task data live here.
    model_source:
        *HuggingFace hub ID* (e.g. ``"OvercastLab/Quark-50m-Instruct"``) or
        *local directory path*.  The model is downloaded / copied into
        ``linked_host_dir``, replacing existing model files.  ``None`` means
        the model is already present.
    replace_existing:
        When ``True`` (default when ``model_source`` is given), clear
        ``linked_host_dir`` before placing the new model.
    questions:
        Default list of research questions.  Can be overridden per-call via
        ``explain({"questions": [...]})```.
    env_mode:
        Sandbox backend.  ``"auto"`` tries Docker first, then proot.
    max_steps:
        Maximum ReACT steps for the main agent per question.
    max_steps_sub:
        Maximum steps for each sub-agent call.
    """

    def __init__(
        self,
        vlm: dict,
        linked_host_dir: str,
        model_source: Optional[str] = None,
        replace_existing: bool = True,
        tool_folder: str = DEFAULT_TOOL_FOLDER,
        questions: Optional[List[str]] = None,
        env_mode: str = "auto",
        env_cwd: Optional[str] = None,
        sandbox_image: str = "ubuntu:22.04",
        sandbox_mount_dir: str = "/workspace",
        sandbox_network_disabled: bool = True,
        preinstall_packages: Optional[Sequence[str]] = None,
        max_steps: int = 30,
        max_steps_sub: int = 15,
        # Absorbed from build_explainer — ignored
        model=None,
        labels=None,
        model_type=None,
    ):
        super().__init__(
            linked_host_dir=linked_host_dir,
            env_mode=env_mode,
            env_cwd=env_cwd,
            sandbox_image=sandbox_image,
            sandbox_mount_dir=sandbox_mount_dir,
            sandbox_network_disabled=sandbox_network_disabled,
            preinstall_packages=preinstall_packages,
        )
        self.max_steps = max_steps
        self.max_steps_sub = max_steps_sub
        self.default_questions: List[str] = questions or []
        self.vlm = VLM_REGISTRY[vlm["name"]](**(vlm.get("kwargs") or {}))

        # Set up model in linked_host_dir
        if model_source is not None:
            self._setup_model(model_source, replace_existing)

        # Copy bash tools into the workspace so they're accessible in the sandbox
        self._tools_sandbox_path: Optional[str] = None
        if os.path.isdir(tool_folder):
            tools_dest = os.path.join(self.linked_host_dir, ".agent_tools")
            if os.path.exists(tools_dest):
                shutil.rmtree(tools_dest)
            shutil.copytree(tool_folder, tools_dest)
            for fname in os.listdir(tools_dest):
                fpath = os.path.join(tools_dest, fname)
                if os.path.isfile(fpath):
                    os.chmod(fpath, 0o755)
            self._tools_sandbox_path = (
                f"{self.env.sandbox_mount_dir}/.agent_tools"
            )
            # Export PATH in the persistent shell so tools are callable by name
            path_export = f'export PATH="{self._tools_sandbox_path}:$PATH"'
            self.env.exec(path_export)
            # Store for auto-restart
            if hasattr(self.env, "_tools_path_export"):
                self.env._tools_path_export = path_export
        self.tool_folder = tool_folder

    # ------------------------------------------------------------------
    # Model setup
    # ------------------------------------------------------------------

    def _setup_model(self, model_source: str, replace_existing: bool) -> None:
        """Download or copy ``model_source`` into ``linked_host_dir``.

        If ``replace_existing`` is True, existing non-tool files are removed
        before placing the new model.
        """
        dest = self.linked_host_dir
        local_path = os.path.abspath(model_source)
        is_local = os.path.isdir(local_path)

        if replace_existing:
            _clear_model_dir(dest)

        if is_local:
            if local_path == dest:
                return  # already in place
            _copy_dir_contents(local_path, dest)
        else:
            # Treat as HuggingFace hub ID
            try:
                from huggingface_hub import snapshot_download
            except ImportError:
                raise ImportError(
                    "huggingface_hub is required to download models.\n"
                    "  pip install huggingface-hub"
                )
            snapshot_download(repo_id=model_source, local_dir=dest)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def explain(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Answer each research question via an independent ReACT main-agent loop.

        Args:
            inputs: dict with optional key ``"questions"`` (List[str]).
                    Falls back to ``self.default_questions`` when absent.

        Returns:
            Dict of parallel lists, one entry per question:
                ``question``, ``answer``, ``trajectory``, ``steps``.
        """
        questions: List[str] = inputs.get("questions") or self.default_questions
        if not questions:
            return {"questions": [], "answers": [], "trajectories": [], "steps": []}

        results = []
        for question in questions:
            result = self._react(question)
            result["question"] = question
            results.append(result)

        return {k: [r[k] for r in results] for k in results[0].keys()}

    # ------------------------------------------------------------------
    # Main agent — full action set: bash / ask / batch / submit
    # ------------------------------------------------------------------

    def _react(self, question: str) -> Dict[str, Any]:
        """Full ReACT loop for one question.  Supports bash, ask, batch, submit."""
        model_dir = self.env.sandbox_mount_dir
        trajectory: List[Dict[str, Any]] = []
        answer: Optional[str] = None

        system_prompt = _MAIN_SYSTEM_PROMPT.format(
            model_dir=model_dir,
            question=question,
            tools_block=self._get_tools_block(),
        )

        for step_i in range(self.max_steps):
            prompt = self._build_react_prompt(system_prompt, trajectory)
            response = self._vlm_query(prompt)

            thought = self._extract_thought(response)
            call = self._extract_call(response)

            action_type = call["type"]
            action_content = call["content"]

            if action_type == "action":
                first_token = action_content.split(None, 1)[0].lower()

                if first_token == "submit":
                    answer = action_content[len("submit"):].strip()
                    trajectory.append({
                        "thought": thought,
                        "action_type": action_type,
                        "action_content": action_content,
                        "observation": "[submitted]",
                    })
                    break

                elif first_token == "ask":
                    sub_question = action_content[len("ask"):].strip()
                    sub_result = self._ask(sub_question)
                    observation = sub_result.get("answer", "(no answer)")

                elif first_token == "batch":
                    observation = self._batch()

                else:
                    observation = (
                        f"Unknown action '{first_token}'. "
                        "Available actions: ask, batch, submit."
                    )

            elif action_type == "bash":
                exec_result = self.env.exec(self._ensure_bash(action_content))
                observation = exec_result.output.strip() or "(no output)"
                observation = _truncate_obs(observation)
                # Auto-submit if the output looks like a complete inference result
                if _looks_like_final_result(observation):
                    trajectory.append({
                        "thought": thought,
                        "action_type": action_type,
                        "action_content": action_content,
                        "observation": observation,
                    })
                    answer = observation
                    break

            elif action_type == "none":
                # Agent generated text without a code block.
                # Only treat as implicit submit after at least MIN_STEPS steps
                # to prevent premature termination; otherwise prompt to format.
                MIN_STEPS_BEFORE_IMPLICIT = 6
                if step_i >= MIN_STEPS_BEFORE_IMPLICIT:
                    answer = thought
                    trajectory.append({
                        "thought": thought,
                        "action_type": "none",
                        "action_content": "",
                        "observation": "[implicit submit: text answer accepted]",
                    })
                    break
                else:
                    observation = (
                        "FORMAT ERROR: Your response must end with a code block. "
                        "Use ```bash <cmd>``` to run a command, or "
                        "```action\\nsubmit <answer>``` to submit. "
                        "Do NOT write 'Action:' in plain text."
                    )

            else:
                observation = "(unrecognized action type)"

            trajectory.append({
                "thought": thought,
                "action_type": action_type,
                "action_content": action_content,
                "observation": observation,
            })

        if answer is None:
            last_obs = trajectory[-1]["observation"] if trajectory else "(none)"
            answer = f"(max steps reached) Last observation: {last_obs}"

        return {
            "answer": answer,
            "trajectory": trajectory,
            "steps": len(trajectory),
        }

    # ------------------------------------------------------------------
    # Sub-agent — bash + text generation only (no ask / batch)
    # ------------------------------------------------------------------

    def _ask(self, question: str) -> Dict[str, Any]:
        """Sub-agent loop: bash tools + free-form text generation.

        The sub-agent cannot call ``ask`` or ``batch``.  It terminates when:
        - It issues a ``submit`` action (explicit early exit), OR
        - It generates a response with no code block (final text answer), OR
        - ``max_steps_sub`` is reached (last thought is used as answer).
        """
        model_dir = self.env.sandbox_mount_dir
        trajectory: List[Dict[str, Any]] = []
        answer: Optional[str] = None

        system_prompt = _SUBAGENT_SYSTEM_PROMPT.format(
            model_dir=model_dir,
            question=question,
            tools_block=self._get_tools_block(),
        )

        for sub_i in range(self.max_steps_sub):
            prompt = self._build_react_prompt(system_prompt, trajectory)
            response = self._vlm_query(prompt)

            thought = self._extract_thought(response)
            call = self._extract_call(response)

            action_type = call["type"]
            action_content = call["content"]

            if action_type == "bash":
                exec_result = self.env.exec(self._ensure_bash(action_content))
                observation = exec_result.output.strip() or "(no output)"
                observation = _truncate_obs(observation)
                trajectory.append({
                    "thought": thought,
                    "action_type": action_type,
                    "action_content": action_content,
                    "observation": observation,
                })

            elif action_type == "action":
                first_token = action_content.split(None, 1)[0].lower()
                if first_token == "submit":
                    answer = action_content[len("submit"):].strip()
                    trajectory.append({
                        "thought": thought,
                        "action_type": action_type,
                        "action_content": action_content,
                        "observation": "[submitted]",
                    })
                    break
                else:
                    observation = (
                        f"Sub-agent only supports bash and submit. "
                        f"'{first_token}' is not available here."
                    )
                    trajectory.append({
                        "thought": thought,
                        "action_type": action_type,
                        "action_content": action_content,
                        "observation": observation,
                    })

            else:
                # No code block — accept as final answer only after minimum steps
                if sub_i >= 2:
                    answer = thought or response.strip()
                    trajectory.append({
                        "thought": thought,
                        "action_type": "none",
                        "action_content": "",
                        "observation": "[text answer accepted]",
                    })
                    break
                else:
                    observation = (
                        "FORMAT ERROR: Use ```bash <cmd>``` or "
                        "```action\\nsubmit <answer>```. Never plain 'Action:' text."
                    )
                    trajectory.append({
                        "thought": thought,
                        "action_type": "none",
                        "action_content": "",
                        "observation": observation,
                    })

        if answer is None:
            answer = (
                trajectory[-1]["thought"]
                if trajectory
                else "(sub-agent produced no answer)"
            )

        return {
            "answer": answer,
            "trajectory": trajectory,
            "steps": len(trajectory),
        }

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_react_prompt(
        self, system_prompt: str, trajectory: List[Dict[str, Any]]
    ) -> str:
        """Concatenate system prompt, recent trajectory steps, and continuation cue.

        Only the most recent ``_MAX_TRAJ_STEPS`` steps are included to avoid
        exceeding the model's context length limit.
        """
        _MAX_TRAJ_STEPS = 6
        recent = trajectory[-_MAX_TRAJ_STEPS:] if len(trajectory) > _MAX_TRAJ_STEPS else trajectory
        prompt = system_prompt
        if len(trajectory) > _MAX_TRAJ_STEPS:
            prompt += f"[... {len(trajectory) - _MAX_TRAJ_STEPS} earlier steps omitted for brevity ...]\n\n"
        for step in recent:
            if step["action_type"] == "none":
                prompt += f"{step['thought']}\nObservation: {step['observation']}\n\n"
            else:
                prompt += _STEP_TEMPLATE.format(
                    thought=step["thought"],
                    action_type=step["action_type"],
                    action_content=step["action_content"],
                    observation=step["observation"],
                )
        prompt += "Thought:"
        return prompt

    # ------------------------------------------------------------------
    # Bash sanitiser
    # ------------------------------------------------------------------

    _PYTHON_FIRST_LINE_TOKENS = (
        "import ", "from ", "def ", "class ", "print(",
        "for ", "with ", "if ", "while ", "#!", "try:", "raise ",
    )

    def _ensure_bash(self, command: str) -> str:
        """Detect Python code in a bash block and write it to a script file.

        Handles three patterns:
        1. Raw Python (first code line starts with a Python token).
        2. Comment header then Python code (``# comment\\nimport ...``).
        3. ``python3 -c "<multi-line code>"`` — extract the code, fix the quoting.
        """
        stripped = command.strip()

        # --- Pattern 3: python3 -c "..." with embedded newlines ---
        m = re.search(r'python3?\s+-c\s+["\'](.+)["\']', stripped, re.DOTALL)
        if m and "\n" in m.group(1):
            return self._write_python_script(m.group(1))

        # --- Patterns 1 & 2: first *non-comment* code line is Python ---
        lines = [l.strip() for l in stripped.split("\n") if l.strip()]
        for line in lines:
            if line.startswith("#") and not line.startswith("#!"):
                continue  # skip plain comments; keep looking
            looks_like_python = any(
                line.startswith(tok) for tok in self._PYTHON_FIRST_LINE_TOKENS
            )
            if looks_like_python:
                return self._write_python_script(stripped)
            break  # first non-comment line is not Python → keep as bash

        return command

    def _write_python_script(self, code: str) -> str:
        """Write ``code`` to a numbered script file and return the exec command."""
        script_idx = getattr(self, "_script_counter", 0) + 1
        self._script_counter = script_idx
        script_name = f"_agent_{script_idx:03d}.py"
        script_host = os.path.join(self.linked_host_dir, script_name)
        script_sandbox = f"{self.env.sandbox_mount_dir}/{script_name}"
        with open(script_host, "w") as f:
            f.write(code)
        return f"python3 {script_sandbox}"

    # ------------------------------------------------------------------
    # VLM helper
    # ------------------------------------------------------------------

    def _vlm_query(self, text: str) -> str:
        """Issue a single text-only query to the VLM."""
        responses = self.vlm.query_batch([{"text": text}])
        return responses[0]

    # ------------------------------------------------------------------
    # Tool helpers
    # ------------------------------------------------------------------

    def _get_tools(self) -> List[str]:
        """Return description strings for every tool in the tool folder."""
        descriptions: List[str] = []
        if not os.path.isdir(self.tool_folder):
            return descriptions
        for tool in sorted(os.listdir(self.tool_folder)):
            tool_path = os.path.join(self.tool_folder, tool)
            if os.path.isfile(tool_path):
                result = self.env.exec(f"{tool} --desc")
                desc = result.output.strip()
                if desc:
                    descriptions.append(f"  {tool}: {desc}")
        return descriptions

    def _get_tools_block(self) -> str:
        descriptions = self._get_tools()
        if not descriptions:
            return "(none)"
        return "\n".join(descriptions)

    # ------------------------------------------------------------------
    # Batch helper
    # ------------------------------------------------------------------

    def _batch(self) -> str:
        """Read task.jsonl from the model dir, answer with VLM, write back."""
        task_path = os.path.join(self.linked_host_dir, "task.jsonl")
        if not os.path.exists(task_path):
            return "task.jsonl not found in model directory."
        with open(task_path) as f:
            questions = [json.loads(line) for line in f if line.strip()]
        results = self.vlm.query_batch(questions)
        for q, r in zip(questions, results):
            q["answer"] = r
        with open(task_path, "w") as f:
            for q in questions:
                f.write(json.dumps(q) + "\n")
        return f"Answered {len(questions)} entries; results saved to task.jsonl."

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    def _extract_thought(self, text: str) -> str:
        """Return the reasoning text that precedes the first code block."""
        text = re.sub(r"^\s*Thought:\s*", "", text, count=1)
        match = re.search(r"```", text)
        if match:
            return text[: match.start()].strip()
        return text.strip()

    def _extract_call(self, text: str) -> Dict[str, str]:
        """Extract the first ```lang ... ``` block from the response.

        Returns:
            ``{"type": "bash"|"action"|"none", "content": "..."}``
        """
        match = re.search(r"```(.*?)```", text, re.DOTALL)
        if match:
            inner = match.group(1).strip()
            if inner.startswith("bash"):
                return {"type": "bash", "content": inner[len("bash"):].strip()}
            if inner.startswith("action"):
                return {"type": "action", "content": inner[len("action"):].strip()}
            # Unknown fence type — skip first line (fence lang tag), use rest
            _, _, rest = inner.partition("\n")
            return {"type": "bash", "content": rest.strip() or inner}
        # No code block found
        return {"type": "none", "content": ""}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _clear_model_dir(directory: str) -> None:
    """Remove model files from ``directory``, leaving tool dirs (.agent_tools) intact."""
    for name in os.listdir(directory):
        if name.startswith("."):
            continue  # skip hidden dirs like .agent_tools
        path = os.path.join(directory, name)
        if os.path.isfile(path) or os.path.islink(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)


def _copy_dir_contents(src: str, dst: str) -> None:
    """Copy all contents of ``src`` into ``dst``."""
    for name in os.listdir(src):
        s = os.path.join(src, name)
        d = os.path.join(dst, name)
        if os.path.isfile(s):
            shutil.copy2(s, d)
        elif os.path.isdir(s):
            if os.path.exists(d):
                shutil.rmtree(d)
            shutil.copytree(s, d)
