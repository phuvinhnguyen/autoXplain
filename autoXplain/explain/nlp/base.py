import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from autoXplain.explain.base import BaseExplainer

NLP_REGISTRY = {}


def nlp(cls):
    NLP_REGISTRY[cls.__name__] = cls
    return cls


class BaseNLPExplainer(BaseExplainer):
    """Base for NLP explainers.

    Unlike image/saliency explainers, NLP explainers do not operate on a
    torch.nn.Module directly.  The model lives inside the sandbox environment
    as a directory of weights + config files.  The ``model`` argument accepted
    by ``build_explainer`` is silently ignored; use ``model_source`` in
    concrete subclasses to specify the model.
    """

    target = "summary"

    def __init__(
        self,
        linked_host_dir: str,
        env_mode: str = "auto",
        env_cwd: Optional[str] = None,
        sandbox_image: str = "ubuntu:22.04",
        sandbox_mount_dir: str = "/workspace",
        sandbox_network_disabled: bool = True,
        preinstall_packages: Optional[Sequence[str]] = None,
        # Absorbed from build_explainer — not used by NLP explainers
        model=None,
        labels=None,
        model_type=None,
    ):
        # NLP explainers don't use a torch model — bypass BaseExplainer.__init__
        self.model = None
        self.device = None
        self.linked_host_dir = os.path.abspath(linked_host_dir)
        os.makedirs(self.linked_host_dir, exist_ok=True)
        self.env = create_env(
            mode=env_mode,
            linked_host_dir=self.linked_host_dir,
            cwd=env_cwd,
            sandbox_image=sandbox_image,
            sandbox_mount_dir=sandbox_mount_dir,
            sandbox_network_disabled=sandbox_network_disabled,
            preinstall_packages=preinstall_packages,
        )

    @abstractmethod
    def explain(self, inputs: Dict) -> Dict:
        ...

    def run(self, inputs: Dict) -> Dict:
        return self.explain(inputs)

    def __call__(self, inputs: Dict) -> Dict:
        return self.explain(inputs)


# ---------------------------------------------------------------------------
# Execution result
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Environment base
# ---------------------------------------------------------------------------

class BaseEnv(ABC):
    """Execution environment abstraction for NLP explainers."""

    sandbox_mount_dir: str = "/workspace"

    @abstractmethod
    def exec(self, command: str, timeout_s: Optional[int] = None) -> EnvExecResult:
        ...

    def __del__(self):
        return None


# ---------------------------------------------------------------------------
# Docker environments
# ---------------------------------------------------------------------------

class DockerIsolatedEnv(BaseEnv):
    """
    Run each command in a fresh Docker container.
    - State (cd/exports) from one command does NOT carry to the next.
    - Only ``linked_host_dir`` is mounted into the sandbox.
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
        self.sandbox_mount_dir = mount_dir
        self.cwd = cwd
        self.network_disabled = network_disabled
        _ensure_docker_available()

    def exec(self, command: str, timeout_s: Optional[int] = None) -> EnvExecResult:
        workdir = self.cwd or self.mount_dir
        cmd = [
            "docker", "run", "--rm",
            "-v", f"{self.linked_host_dir}:{self.mount_dir}",
            "-w", workdir,
        ]
        if self.network_disabled:
            cmd.extend(["--network", "none"])
        cmd.extend([self.image, "sh", "-lc", command])
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        return EnvExecResult(
            stdout=proc.stdout,
            stderr=proc.stderr,
            returncode=proc.returncode,
        )


class DockerInteractiveEnv(BaseEnv):
    """
    Run commands in one persistent Docker container with a shared shell session.
    - Shell state (cd, exports) is shared across calls.
    - Only ``linked_host_dir`` is visible to the sandbox.
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
        self.sandbox_mount_dir = mount_dir
        self.cwd = cwd or mount_dir
        self.network_disabled = network_disabled
        _ensure_docker_available()

        run_cmd = [
            "docker", "run", "-d", "--rm",
            "-v", f"{self.linked_host_dir}:{self.mount_dir}",
            "-w", self.cwd,
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
            encoding="utf-8",
            errors="replace",
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
        returncode = 1
        while True:
            line = self._proc.stdout.readline()
            if line == "":
                break
            if line.startswith(marker + ":"):
                returncode = int(line.strip().split(":")[-1])
                break
            stdout_chunks.append(line)

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


# ---------------------------------------------------------------------------
# Proot environment (no Docker / no root required)
# ---------------------------------------------------------------------------

class ProotInteractiveEnv(BaseEnv):
    """
    Run commands via proot with a persistent shell session.

    proot provides filesystem isolation without requiring root or Docker:
    the agent only sees ``linked_host_dir`` (mounted at ``mount_dir``),
    plus the system libraries required to run Python.

    Note: proot does NOT isolate network access.  If the agent must not
    reach the internet, pair this with firewall rules or a network namespace
    created by ``unshare --net`` (requires user-namespaces to be enabled).

    Install proot:
        apt install proot          # Debian/Ubuntu
        conda install -c conda-forge proot
    """

    # Host directories always bound into the sandbox (read access follows host perms)
    _SYSTEM_BIND_DIRS = [
        "/usr", "/bin", "/sbin",
        "/lib", "/lib64", "/lib32",
        "/proc", "/dev", "/sys", "/tmp",
        "/etc/alternatives", "/etc/ssl",
        "/etc/ld.so.cache",
        "/etc/passwd", "/etc/group",
        "/etc/nsswitch.conf", "/etc/resolv.conf",
        "/etc/hostname",
        # NVIDIA GPU access (needed for CUDA in the sandbox)
        "/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools",
        "/dev/nvidia0", "/dev/nvidia1", "/dev/nvidia2", "/dev/nvidia3",
        "/proc/driver/nvidia",
    ]

    def __init__(
        self,
        linked_host_dir: str,
        mount_dir: str = "/workspace",
        cwd: Optional[str] = None,
        network_disabled: bool = True,  # proot can't enforce; kept for API compat
        rootfs_dir: Optional[str] = None,
    ):
        self.linked_host_dir = _ensure_abs_dir(linked_host_dir)
        self.mount_dir = mount_dir
        self.sandbox_mount_dir = mount_dir
        self._cwd = cwd or mount_dir
        self.network_disabled = network_disabled

        _ensure_proot_available()

        if rootfs_dir is None:
            self._tmpdir = tempfile.mkdtemp(prefix="autoxplain_proot_")
        else:
            self._tmpdir = None
            os.makedirs(rootfs_dir, exist_ok=True)
        self.rootfs_dir = rootfs_dir or self._tmpdir
        _init_proot_rootfs_skeleton(self.rootfs_dir)

        cmd = ["proot", "-r", self.rootfs_dir]

        for d in self._SYSTEM_BIND_DIRS:
            if os.path.exists(d):
                cmd += ["-b", f"{d}:{d}"]

        # Bind the active Python environment (conda, venv, etc.) if not under /usr
        for p in _get_python_bind_paths():
            cmd += ["-b", f"{p}:{p}"]

        # Bind HuggingFace model cache (read-only access; avoids re-downloading)
        hf_cache = os.path.expanduser("~/.cache/huggingface")
        if os.path.isdir(hf_cache):
            cmd += ["-b", f"{hf_cache}:{hf_cache}"]

        cmd += ["-b", f"{self.linked_host_dir}:{self.mount_dir}"]
        cmd += ["-w", self._cwd]
        cmd += ["/bin/sh"]

        env = os.environ.copy()
        env["PROOT_NO_SECCOMP"] = "1"   # avoid seccomp issues on newer kernels

        if network_disabled:
            # Attempt to run inside a network namespace (requires unprivileged
            # user namespaces: sysctl kernel.unprivileged_userns_clone=1)
            unshare = shutil.which("unshare")
            if unshare:
                try:
                    test = subprocess.run(
                        [unshare, "--net", "--map-root-user", "true"],
                        capture_output=True, timeout=3,
                    )
                    if test.returncode == 0:
                        cmd = [unshare, "--net", "--map-root-user"] + cmd
                except Exception:
                    pass

        self._cmd = cmd
        self._env = env
        self._tools_path_export: Optional[str] = None  # set externally by NLPAgent

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",   # replace undecodable bytes instead of crashing
            bufsize=1,
            env=env,
        )

        # Verify the shell started correctly
        probe = self.exec("echo __PROOT_ALIVE__", timeout_s=10)
        if "__PROOT_ALIVE__" not in probe.stdout:
            self._proc.terminate()
            raise RuntimeError(
                f"proot shell failed to respond after startup.\n"
                f"Output: {probe.output!r}\n"
                f"Command was: {' '.join(cmd[:6])} ..."
            )

    def _restart_shell(self) -> None:
        """Restart the proot shell process (e.g. after a crash)."""
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()
        self._proc = subprocess.Popen(
            self._cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=self._env,
        )
        # Re-export PATH for tools
        if self._tools_path_export:
            self.exec(self._tools_path_export)

    def exec(self, command: str, timeout_s: Optional[int] = None) -> EnvExecResult:
        if self._proc is None or self._proc.stdin is None or self._proc.stdout is None:
            raise RuntimeError("proot shell pipes unavailable")

        # Auto-restart if the shell crashed between calls
        if self._proc.poll() is not None:
            self._restart_shell()

        marker = f"__AUTOXPLAIN_ENV_DONE_{uuid.uuid4().hex}__"
        wrapped = f"{command}\necho {marker}:$?\n"
        try:
            self._proc.stdin.write(wrapped)
            self._proc.stdin.flush()
        except BrokenPipeError:
            # Shell died — return error as observation (agent can retry differently)
            return EnvExecResult(
                stdout="",
                stderr="Shell crashed (BrokenPipe). Try a simpler command or write a script file.",
                returncode=1,
            )

        import select as _select, time as _time
        effective_timeout = timeout_s or 180
        stdout_chunks: List[str] = []
        returncode = 1
        deadline = _time.time() + effective_timeout
        shell_died = False

        while True:
            remaining = deadline - _time.time()
            if remaining <= 0:
                return EnvExecResult(
                    stdout="".join(stdout_chunks),
                    stderr=f"(command timed out after {effective_timeout}s)",
                    returncode=124,
                )
            ready, _, _ = _select.select([self._proc.stdout], [], [], min(remaining, 5.0))
            if not ready:
                if self._proc.poll() is not None:
                    shell_died = True
                    break
                continue
            line = self._proc.stdout.readline()
            if line == "":
                shell_died = True
                break
            if line.startswith(marker + ":"):
                try:
                    returncode = int(line.strip().split(":")[-1])
                except ValueError:
                    returncode = 0
                break
            stdout_chunks.append(line)

        collected = "".join(stdout_chunks)
        if shell_died:
            return EnvExecResult(
                stdout=collected,
                stderr="(shell exited unexpectedly during command — it will be restarted on next call)",
                returncode=1,
            )
        return EnvExecResult(
            stdout=collected,
            stderr="",
            returncode=returncode,
        )

    def __del__(self):
        if hasattr(self, "_proc") and self._proc and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._tmpdir and os.path.exists(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Deprecated local env (kept to surface a clear error message)
# ---------------------------------------------------------------------------

class _DeprecatedLocalEnv(BaseEnv):
    """Reserved to avoid accidental local execution in NLP explainers."""

    def exec(self, command: str, timeout_s: Optional[int] = None) -> EnvExecResult:
        raise RuntimeError("Local env is disabled. Use a sandbox env (docker or proot).")

    def __del__(self):
        return None


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_env(
    mode: str = "auto",
    linked_host_dir: Optional[str] = None,
    cwd: Optional[str] = None,
    sandbox_image: str = "ubuntu:22.04",
    sandbox_mount_dir: str = "/workspace",
    sandbox_network_disabled: bool = True,
    preinstall_packages: Optional[Sequence[str]] = None,
) -> BaseEnv:
    """Create a sandboxed execution environment.

    mode:
        ``"auto"``               — try Docker first, fall back to proot
        ``"interactive"``        — Docker interactive (persistent shell)
        ``"isolated"``           — Docker isolated (fresh container per command)
        ``"docker_interactive"`` — alias for ``"interactive"``
        ``"docker_isolated"``    — alias for ``"isolated"``
        ``"proot"``              — proot-based persistent shell (no Docker/root)
    """
    if linked_host_dir is None:
        raise ValueError("linked_host_dir must be provided for sandbox env")

    if mode == "auto":
        if _is_docker_available():
            mode = "interactive"
        elif _is_proot_available():
            mode = "proot"
        else:
            raise RuntimeError(
                "No sandbox backend found.\n"
                "  Option 1 – install Docker: https://docs.docker.com/get-docker/\n"
                "  Option 2 – install proot:  apt install proot\n"
                "             or: conda install -c conda-forge proot"
            )

    if mode in ("interactive", "docker_interactive"):
        effective_packages = list(preinstall_packages or _default_nlp_packages())
        effective_image = _prepare_sandbox_image(sandbox_image, effective_packages)
        return DockerInteractiveEnv(
            linked_host_dir=linked_host_dir,
            image=effective_image,
            mount_dir=sandbox_mount_dir,
            cwd=cwd,
            network_disabled=sandbox_network_disabled,
        )

    if mode in ("isolated", "docker_isolated"):
        effective_packages = list(preinstall_packages or _default_nlp_packages())
        effective_image = _prepare_sandbox_image(sandbox_image, effective_packages)
        return DockerIsolatedEnv(
            linked_host_dir=linked_host_dir,
            image=effective_image,
            mount_dir=sandbox_mount_dir,
            cwd=cwd,
            network_disabled=sandbox_network_disabled,
        )

    if mode == "proot":
        return ProotInteractiveEnv(
            linked_host_dir=linked_host_dir,
            mount_dir=sandbox_mount_dir,
            cwd=cwd,
            network_disabled=sandbox_network_disabled,
        )

    raise ValueError(
        f"Unknown env mode: {mode!r}. "
        "Choose from: auto, interactive, isolated, docker_interactive, docker_isolated, proot"
    )


# ---------------------------------------------------------------------------
# Proot helpers
# ---------------------------------------------------------------------------

def _ensure_proot_available() -> None:
    if shutil.which("proot") is None:
        raise RuntimeError(
            "proot is not installed or not on PATH.\n"
            "  Ubuntu/Debian: sudo apt install proot\n"
            "  Conda:         conda install -c conda-forge proot"
        )


def _is_proot_available() -> bool:
    return shutil.which("proot") is not None


def _init_proot_rootfs_skeleton(rootfs_dir: str) -> None:
    """Create the minimal directory skeleton required by proot."""
    for d in ["bin", "etc", "lib", "lib64", "usr", "proc", "dev",
              "sys", "tmp", "run", "sbin", "workspace", "root"]:
        os.makedirs(os.path.join(rootfs_dir, d), exist_ok=True)

    # Minimal /etc/passwd so shells don't complain
    passwd_path = os.path.join(rootfs_dir, "etc", "passwd")
    if not os.path.exists(passwd_path):
        uid = os.getuid()
        gid = os.getgid()
        with open(passwd_path, "w") as f:
            f.write("root:x:0:0:root:/root:/bin/sh\n")
            f.write(f"user:x:{uid}:{gid}:user:/workspace:/bin/sh\n")

    group_path = os.path.join(rootfs_dir, "etc", "group")
    if not os.path.exists(group_path):
        gid = os.getgid()
        with open(group_path, "w") as f:
            f.write("root:x:0:\n")
            f.write(f"user:x:{gid}:\n")


def _get_python_bind_paths() -> List[str]:
    """Return host directories that must be bound into proot for Python to work.

    Covers conda envs, venvs, and any Python prefix outside ``/usr``.
    """
    system_prefixes = ("/usr", "/bin", "/sbin", "/lib", "/lib64")
    paths: List[str] = []

    prefix = os.path.realpath(sys.prefix)
    if not any(prefix.startswith(p) for p in system_prefixes):
        paths.append(prefix)

    exec_dir = os.path.realpath(os.path.dirname(sys.executable))
    if (exec_dir not in paths
            and not any(exec_dir.startswith(p) for p in system_prefixes)
            and not exec_dir.startswith(prefix)):
        paths.append(exec_dir)

    return paths


# ---------------------------------------------------------------------------
# Docker helpers
# ---------------------------------------------------------------------------

def _ensure_abs_dir(path: str) -> str:
    if not path:
        raise ValueError("linked_host_dir is required")
    abs_path = os.path.abspath(path)
    os.makedirs(abs_path, exist_ok=True)
    return abs_path


def _ensure_docker_available() -> None:
    check = subprocess.run(
        ["docker", "version", "--format", "{{.Server.Version}}"],
        capture_output=True,
        text=True,
    )
    if check.returncode != 0:
        raise RuntimeError(
            "Docker is required but not available. "
            "Install/start Docker or use env_mode='proot'."
        )


def _is_docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


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
    """Build (or reuse) a derived Docker image with Python + requested NLP packages."""
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
    dockerfile = (
        f"FROM {base_image}\n"
        "RUN " + install_cmd + "\n"
        'CMD ["sh"]\n'
    )
    proc = subprocess.run(
        ["docker", "build", "-t", tag, "-"],
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
    digest = hashlib.sha256("|".join(packages).encode()).hexdigest()[:12]
    return f"autoxplain-nlp-sandbox-{cleaned}-{digest}"
