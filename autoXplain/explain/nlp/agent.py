'''# Agent Actions:
## Bash
include tools and unix commands
```bash
<bash command>
```

## Actions
- submit: submit the explanation
- batch: answer all json questions from a jsonl fil, return answers in that file
    + tagging (medium-risk)
    + free-form (high-risk)
    + classification (low-risk)
    + normalizing (low-risk)
- ask: ask a question about the current working directory -> call another agent to work on this question
```action
<action name> <input>
```

'''

from autoXplain.explain.nlp.base import BaseNLPExplainer, nlp
from autoXplain.utils.vlm import VLM_REGISTRY
from typing import Optional, Sequence, Dict, Any, List
import os, shutil, re, json

DEFAULT_TOOL_FOLDER = os.path.join(os.path.dirname(__file__), "bash")

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

# Top-level explain: one question at a time, full action set
_EXPLAIN_SYSTEM_PROMPT = """\
You are an expert AI researcher analyzing the machine learning model located at:
{model_dir}

Your current task is to answer the following research question:
{question}

## Available Tools

**Bash** — run any shell command (non-interactive, no pagers):
```bash
<command>
```

## Additional CLI Tools (callable from bash)
{tools_block}

## Available Actions

**Ask** — delegate a focused sub-question to a helper agent and receive its
answer as an observation:
```action
ask <sub-question>
```

**Batch** — have the VLM answer all entries in {model_dir}/task.jsonl
(each line: {{"text": "...", "image_path": "..."}}); answers are written back
to the same file:
```action
batch
```

**Submit** — when you have a confident, evidence-based answer, submit it:
```action
submit <your answer>
```

## Instructions
1. Start by exploring {model_dir} to understand what is available.
2. Write your reasoning as plain text (Thought), then issue exactly one action.
3. Use each observation to guide the next step.
4. Submit as soon as you can answer the research question with confidence.
"""

# Sub-agent: bash + submit only
_ASK_SYSTEM_PROMPT = """\
You are an expert AI researcher. Your goal is to answer a single focused
research question about the machine learning model located at: {model_dir}

## Available Tools

**Bash** — run any shell command (non-interactive, no pagers):
```bash
<command>
```

## Additional CLI Tools (callable from bash)
{tools_block}

## Available Actions

**Submit** — when you have gathered sufficient evidence, submit your final answer:
```action
submit <answer>
```

## Instructions
1. Inspect {model_dir} as needed to gather evidence.
2. Write your reasoning first (Thought), then one action per step.
3. Submit a concise, evidence-based answer as soon as you can.

## Research Question
{question}
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
    def __init__(
        self,
        model,
        vlm: dict,
        linked_host_dir: str,
        tool_folder: str = DEFAULT_TOOL_FOLDER,
        env_mode: str = "interactive",
        env_cwd: Optional[str] = None,
        sandbox_image: str = "ubuntu:22.04",
        sandbox_mount_dir: str = "/workspace",
        sandbox_network_disabled: bool = True,
        preinstall_packages: Optional[Sequence[str]] = None,
        max_steps: int = 99,
    ):
        super().__init__(model, linked_host_dir, env_mode, env_cwd, sandbox_image, sandbox_mount_dir, sandbox_network_disabled, preinstall_packages)
        if os.path.exists(tool_folder):
            tool_path = shutil.copy(tool_folder, self.env.sandbox_mount_dir)
            self.env.exec(f"export PATH={tool_path}:$PATH")  # add tools to PATH
        self.tool_folder = tool_folder
        self.vlm = VLM_REGISTRY[vlm["name"]](**(vlm.get("kwargs") or {}))
        self.max_steps = max_steps

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def explain(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Answer each research question in sequence via independent ReACT loops.

        For each question the agent runs its own ReACT trajectory (bash / ask / batch / submit).  ``submit`` ends that question's loop and its text becomes the answer.  The next question then starts a fresh trajectory.

        Args:
            inputs: dict with at least:
                - ``questions`` (List[str]): research questions to answer.

        Returns:
            Dict of lists (parallel arrays), one entry per question:
                - ``question``   – original question text
                - ``answer``     – final submitted answer
                - ``trajectory`` – ReACT steps for this question
                                   [{thought, action_type, action_content,
                                     observation}, ...]
                - ``steps``      – number of ReACT steps taken
        """
        questions: List[str] = inputs.get("questions", [])
        results = []

        for question in questions:
            result = self._react(question)
            result["question"] = question
            results.append(result)

        if not results:
            return {"questions": [], "answers": [], "trajectories": [], "steps": []}

        # Convert list-of-dicts → dict-of-lists (same convention as VLMJudge)
        return {k: [r[k] for r in results] for k in results[0].keys()}

    # ------------------------------------------------------------------
    # Top-level ReACT loop (full action set: bash / ask / batch / submit)
    # ------------------------------------------------------------------

    def _react(self, question: str) -> Dict[str, Any]:
        """Full ReACT loop for one question.  Supports bash, ask, batch, submit.

        Returns:
            dict with keys: answer (str), trajectory (list), steps (int).
        """
        model_dir = self.env.sandbox_mount_dir
        trajectory: List[Dict[str, Any]] = []
        answer: Optional[str] = None

        system_prompt = _EXPLAIN_SYSTEM_PROMPT.format(
            model_dir=model_dir,
            question=question,
            tools_block=self._get_tools_block(),
        )

        for _ in range(self.max_steps):
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
                        "Use bash, ask, batch, or submit."
                    )

            elif action_type == "bash":
                exec_result = self.env.exec(action_content)
                observation = exec_result.output.strip() or "(no output)"

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
            answer = (
                f"(max steps reached without submit) Last observation: {last_obs}"
            )

        return {
            "answer": answer,
            "trajectory": trajectory,
            "steps": len(trajectory),
        }

    # ------------------------------------------------------------------
    # Sub-agent ReACT loop (bash + submit only)
    # ------------------------------------------------------------------

    def _ask(self, question: str) -> Dict[str, Any]:
        """Focused sub-agent for a single sub-question (bash + submit only).

        Returns:
            dict with keys: answer (str), trajectory (list), steps (int).
        """
        model_dir = self.env.sandbox_mount_dir
        trajectory: List[Dict[str, Any]] = []
        answer: Optional[str] = None

        system_prompt = _ASK_SYSTEM_PROMPT.format(
            model_dir=model_dir,
            question=question,
            tools_block=self._get_tools_block(),
        )

        for _ in range(self.max_steps):
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
                else:
                    observation = (
                        f"Sub-agent only supports bash and submit. "
                        f"Unknown action: '{first_token}'."
                    )

            elif action_type == "bash":
                exec_result = self.env.exec(action_content)
                observation = exec_result.output.strip() or "(no output)"

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
            answer = (
                f"(max steps reached without submit) Last observation: {last_obs}"
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
        """Concatenate system prompt, prior (Thought/Action/Observation) steps,
        and the continuation cue."""
        prompt = system_prompt
        for step in trajectory:
            prompt += _STEP_TEMPLATE.format(
                thought=step["thought"],
                action_type=step["action_type"],
                action_content=step["action_content"],
                observation=step["observation"],
            )
        prompt += "Thought:"
        return prompt

    # ------------------------------------------------------------------
    # VLM helper
    # ------------------------------------------------------------------

    def _vlm_query(self, text: str) -> str:
        """Issue a single text-only query to the VLM and return the response."""
        responses = self.vlm.query_batch([{"text": text}])
        return responses[0]

    # ------------------------------------------------------------------
    # Tool helpers
    # ------------------------------------------------------------------

    def _get_tools(self) -> List[str]:
        """Return raw description strings for every tool in the tool folder."""
        descriptions = []
        if not os.path.isdir(self.tool_folder):
            return descriptions
        for tool in sorted(os.listdir(self.tool_folder)):
            if os.path.isfile(os.path.join(self.tool_folder, tool)):
                result = self.env.exec(f"{tool} --desc")
                desc = result.stdout.strip()
                if desc:
                    descriptions.append(f"  {tool}: {desc}")
        return descriptions

    def _get_tools_block(self) -> str:
        """Return a formatted prompt section listing all extra bash tools.

        Returns an empty string when no tools are available so the placeholder
        in the prompt template disappears cleanly.
        """
        descriptions = self._get_tools()
        if not descriptions:
            return ""
        lines = "\n".join(descriptions)
        return (
            "## Additional CLI Tools (callable from bash)\n"
            f"{lines}\n\n"
        )

    def _batch(self) -> str:
        '''Read file task.jsonl from self.env.sandbox_mount_dir and answer each question in the file
        Json format:
        {
            "text": str,
            "image_path": str # Optional, should not be used if the question is text-only
        }
        '''
        with open(os.path.join(self.env.sandbox_mount_dir, "task.jsonl"), "r") as f:
            questions = [json.loads(line) for line in f]
            results = self.vlm.query_batch(questions)

        for question, result in zip(questions, results):
            question["answer"] = result

        with open(os.path.join(self.env.sandbox_mount_dir, "task.jsonl"), "w") as f:
            for question in questions:
                f.write(json.dumps(question) + "\n")

        return f"Answers of {len(questions)} questions have been saved to task.jsonl as fields 'answer'."

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
        '''Extract the call from the text.

        Uses re.search so the code block may be preceded by a Thought section.
        Returns a dictionary with the type of call and the content of the call.
        If no call is found, returns a dictionary with the type "bash" and the
        content "echo 'No call found'".
        '''
        match = re.search(r"```(.*?)```", text, re.DOTALL)
        if match:
            inner = match.group(1).strip()
            if inner.startswith("bash"):
                return {
                    "type": "bash",
                    "content": inner[len("bash"):].strip(),
                }
            elif inner.startswith("action"):
                return {
                    "type": "action",
                    "content": inner[len("action"):].strip(),
                }
        return {"type": "bash", "content": "echo 'No call found'"}
