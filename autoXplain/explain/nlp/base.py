import hashlib
import os
import subprocess
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

from autoXplain.explain.base import BaseExplainer

NLP_REGISTRY = {}


def nlp(cls):
    NLP_REGISTRY[cls.__name__] = cls
    return cls


class BaseNLPExplainer(BaseExplainer):
    """Base for NLP explainers.

    NLP explainers follow the same contract as other explain methods:
    implement `explain(inputs)` and return a dictionary.
    Typical inputs are text-centric payloads such as:
        {'text': str, ...}
    """

    def __init__(
        self,
        model,
        linked_host_dir: str,
        env_mode: str = "interactive",
        env_cwd: Optional[str] = None,
        sandbox_image: str = "ubuntu:22.04",
        sandbox_mount_dir: str = "/workspace",
        sandbox_network_disabled: bool = True,
        preinstall_packages: Optional[Sequence[str]] = None,
    ):
        super().__init__(model)
        self.env = create_env(
            mode=env_mode,
            linked_host_dir=linked_host_dir,
            cwd=env_cwd,
            sandbox_image=sandbox_image,
            sandbox_mount_dir=sandbox_mount_dir,
            sandbox_network_disabled=sandbox_network_disabled,
            preinstall_packages=preinstall_packages,
        )


@dataclass
class EnvExecResult:
    stdout: str
    stderr: str
    returncode: int

    @property
    def output(self) -> str:
        if self.stderr:
            return f"{self.stdout}\n{self.stderr}" if self.stdout else self.stderr
        return self.stdout


class BaseEnv(ABC):
    """Execution environment abstraction for NLP explainers."""

    @abstractmethod
    def exec(self, command: str, timeout_s: Optional[int] = None) -> EnvExecResult:
        ...

    def __del__(self):
        return None


class DockerIsolatedEnv(BaseEnv):
    """
    Run each command in a fresh Docker container.
    - `cd`/exports from one command do NOT affect the next command.
    - Only `linked_host_dir` is mounted into the sandbox container.
    """

    def __init__(
        self,
        linked_host_dir: str,
        image: str,
        mount_dir: str,
        cwd: Optional[str] = None,
        network_disabled: bool = True,
    ):
        self.linked_host_dir = _ensure_abs_dir(linked_host_dir)
        self.image = image
        self.mount_dir = mount_dir
        self.cwd = cwd
        self.network_disabled = network_disabled
        _ensure_docker_available()

    def exec(self, command: str, timeout_s: Optional[int] = None) -> EnvExecResult:
        workdir = self.cwd or self.mount_dir
        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{self.linked_host_dir}:{self.mount_dir}",
            "-w",
            workdir,
        ]
        if self.network_disabled:
            cmd.extend(["--network", "none"])
        cmd.extend([self.image, "sh", "-lc", command])
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return EnvExecResult(
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )


class DockerInteractiveEnv(BaseEnv):
    """
    Run commands in one persistent Docker container + persistent shell session.
    - state is shared across commands (e.g. `cd`, `export`).
    - sandbox still only sees mounted `linked_host_dir`.
    """

    def __init__(
        self,
        linked_host_dir: str,
        image: str,
        mount_dir: str,
        cwd: Optional[str] = None,
        network_disabled: bool = True,
    ):
        self.linked_host_dir = _ensure_abs_dir(linked_host_dir)
        self.image = image
        self.mount_dir = mount_dir
        self.cwd = cwd or mount_dir
        self.network_disabled = network_disabled
        _ensure_docker_available()

        run_cmd = [
            "docker",
            "run",
            "-d",
            "--rm",
            "-v",
            f"{self.linked_host_dir}:{self.mount_dir}",
            "-w",
            self.cwd,
        ]
        if self.network_disabled:
            run_cmd.extend(["--network", "none"])
        run_cmd.extend([self.image, "sh", "-lc", "sleep infinity"])
        proc = subprocess.run(run_cmd, capture_output=True, text=True, check=True)
        self.container_id = proc.stdout.strip()

        self._proc = subprocess.Popen(
            ["docker", "exec", "-i", self.container_id, "sh"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    def exec(self, command: str, timeout_s: Optional[int] = None) -> EnvExecResult:
        if self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("Interactive shell pipes are unavailable")
        marker = f"__AUTOXPLAIN_ENV_DONE_{uuid.uuid4().hex}__"
        wrapped = f"{command}\necho {marker}:$?\n"
        self._proc.stdin.write(wrapped)
        self._proc.stdin.flush()

        stdout_chunks = []
        while True:
            line = self._proc.stdout.readline()
            if line == "":
                break
            if line.startswith(marker + ":"):
                returncode = int(line.strip().split(":")[-1])
                break
            stdout_chunks.append(line)
        else:
            returncode = 1

        return EnvExecResult(
            stdout="".join(stdout_chunks),
            stderr="",
            returncode=returncode,
        )

    def __del__(self):
        if hasattr(self, "_proc") and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if getattr(self, "container_id", None):
            subprocess.run(
                ["docker", "rm", "-f", self.container_id],
                capture_output=True,
                text=True,
            )
            self.container_id = None


def _ensure_abs_dir(path: str) -> str:
    if not path:
        raise ValueError("linked_host_dir is required")
    abs_path = os.path.abspath(path)
    if not os.path.isdir(abs_path):
        raise ValueError(f"linked_host_dir does not exist: {abs_path}")
    return abs_path


def _ensure_docker_available() -> None:
    check = subprocess.run(
        ["docker", "version", "--format", "{{.Server.Version}}"],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        raise RuntimeError(
            "Docker is required for BaseNLPExplainer sandbox env. "
            "Please install/start Docker."
        )


def create_env(
    mode: str = "interactive",
    linked_host_dir: Optional[str] = None,
    cwd: Optional[str] = None,
    sandbox_image: str = "ubuntu:22.04",
    sandbox_mount_dir: str = "/workspace",
    sandbox_network_disabled: bool = True,
    preinstall_packages: Optional[Sequence[str]] = None,
) -> BaseEnv:
    """
    Factory for true sandboxed environment (Docker-based).
    - mode='isolated': new container per command
    - mode='interactive': persistent container + shared shell state
    """
    if linked_host_dir is None:
        raise ValueError("linked_host_dir must be provided for sandbox env")
    effective_packages = list(preinstall_packages or _default_nlp_packages())
    effective_image = _prepare_sandbox_image(sandbox_image, effective_packages)

    if mode == "isolated":
        return DockerIsolatedEnv(
            linked_host_dir=linked_host_dir,
            image=effective_image,
            mount_dir=sandbox_mount_dir,
            cwd=cwd,
            network_disabled=sandbox_network_disabled,
        )
    if mode == "interactive":
        return DockerInteractiveEnv(
            linked_host_dir=linked_host_dir,
            image=effective_image,
            mount_dir=sandbox_mount_dir,
            cwd=cwd,
            network_disabled=sandbox_network_disabled,
        )
    raise ValueError(f"Unknown env mode: {mode}")


class _DeprecatedLocalEnv(BaseEnv):
    """Reserved to avoid accidental local execution in NLP explainers."""

    def exec(self, command: str, timeout_s: Optional[int] = None) -> EnvExecResult:
        raise RuntimeError("Local env is disabled. Use Docker sandbox env modes.")

    def __del__(self):
        return None


def _default_nlp_packages() -> Sequence[str]:
    return (
        "torch",
        "transformers",
        "nltk",
        "numpy",
        "scipy",
        "sentencepiece",
        "accelerate",
    )


def _prepare_sandbox_image(base_image: str, packages: Sequence[str]) -> str:
    """
    Build (or reuse) a derived docker image with Python and requested NLP packages.
    This keeps runtime `env.exec` fast and reproducible.
    """
    safe_pkgs = [p.strip() for p in packages if p and p.strip()]
    if not safe_pkgs:
        _docker_pull_if_needed(base_image)
        return base_image

    _docker_pull_if_needed(base_image)
    tag = _stable_image_tag(base_image, safe_pkgs)
    exists = subprocess.run(
        ["docker", "image", "inspect", tag],
        capture_output=True,
        text=True,
    )
    if exists.returncode == 0:
        return tag

    install_cmd = (
        "apt-get update && "
        "DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip && "
        "python3 -m pip install --no-cache-dir --upgrade pip && "
        f"python3 -m pip install --no-cache-dir {' '.join(safe_pkgs)}"
    )
    build_cmd = [
        "docker",
        "build",
        "-t",
        tag,
        "-",
    ]
    dockerfile = (
        f"FROM {base_image}\n"
        "RUN " + install_cmd + "\n"
        'CMD ["sh"]\n'
    )
    proc = subprocess.run(
        build_cmd,
        input=dockerfile,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to build sandbox image {tag}:\n{proc.stderr}")
    return tag


def _docker_pull_if_needed(image: str) -> None:
    check = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        text=True,
    )
    if check.returncode == 0:
        return
    pull = subprocess.run(["docker", "pull", image], capture_output=True, text=True)
    if pull.returncode != 0:
        raise RuntimeError(f"Failed to pull docker image {image}:\n{pull.stderr}")


def _stable_image_tag(base_image: str, packages: Sequence[str]) -> str:
    cleaned = base_image.replace("/", "_").replace(":", "_")
    digest_src = "|".join(packages)
    digest = hashlib.sha256(digest_src.encode("utf-8")).hexdigest()[:12]
    return f"autoxplain-nlp-sandbox-{cleaned}-{digest}"

