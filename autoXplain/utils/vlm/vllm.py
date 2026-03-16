import os
import sys
import asyncio
import aiohttp
import base64
import subprocess
import time
import requests
from typing import List, Dict, Any
from autoXplain.utils.vlm.base import VLMBase, vlm

DEFAULT_VLLM_PORT = 8000
DEFAULT_HOST = "127.0.0.1"


@vlm
class VLLMClient(VLMBase):
    def __init__(
        self,
        model_name: str = 'Qwen/Qwen2.5-VL-2B-Instruct',
        port: int = None,
        concurrency: int = 8,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 32768,
        start_server: bool = True,
    ):
        self.model_name = model_name
        self.port = port or int(os.getenv('VLLM_PORT', DEFAULT_VLLM_PORT))
        self.host = os.getenv('VLLM_HOST', DEFAULT_HOST)
        self.base_url = f"http://{self.host}:{self.port}"
        self.concurrency = concurrency

        if not self._health_check():
            if start_server:
                self._start_server(gpu_memory_utilization, max_model_len)
            else:
                raise RuntimeError(
                    f"vLLM server not reachable at {self.base_url}. "
                    "Start it manually (e.g. vllm serve <model> --port {}) or set start_server: true in config."
                    .format(self.port)
                )
        assert self._verify_model(), f"Model {model_name} not loaded on server"

    def _health_check(self):
        try:
            return requests.get(f"{self.base_url}/health", timeout=2).status_code == 200
        except Exception:
            return False

    def _verify_model(self):
        try:
            r = requests.get(f"{self.base_url}/v1/models", timeout=5).json()
            return any(m["id"] == self.model_name for m in r.get("data", []))
        except Exception:
            return False

    def _start_server(self, gpu_mem, max_len):
        # Use vllm from the same env as current Python (e.g. conda reagraph)
        vllm_bin = os.path.join(os.path.dirname(sys.executable), "vllm")
        if not os.path.isfile(vllm_bin):
            vllm_bin = "vllm"  # fallback to PATH
        cmd = [
            vllm_bin, "serve", self.model_name,
            "--host", self.host, "--port", str(self.port),
            "--dtype", "bfloat16", "--trust-remote-code",
            "--gpu-memory-utilization", str(gpu_mem),
            "--max-model-len", str(max_len),
        ]
        print(f"Starting vLLM server: {' '.join(cmd)}")
        subprocess.Popen(cmd, start_new_session=True, env=os.environ.copy())
        for _ in range(180):
            if self._health_check() and self._verify_model():
                print("vLLM server ready")
                return
            time.sleep(1)
        raise RuntimeError("vLLM server startup timeout")

    def _encode_image(self, path: str) -> str:
        if path.startswith(('http://', 'https://')):
            return path
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(path)[1].lower().lstrip('.')
        mime = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg',
                'png': 'image/png', 'webp': 'image/webp'}.get(ext, 'image/jpeg')
        return f"data:{mime};base64,{b64}"

    async def _call_api(self, session, sem, image_path: str, text: str, **kwargs) -> str:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": self._encode_image(image_path)}},
                {"type": "text", "text": text},
            ]}],
            "max_tokens": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.1),
            "repetition_penalty": kwargs.get("repetition_penalty", 1.15),
        }
        async with sem:
            async with session.post(f"{self.base_url}/v1/chat/completions", json=payload) as r:
                r.raise_for_status()
                data = await r.json()
        return data["choices"][0]["message"]["content"]

    def query_batch(self, queries: List[Dict[str, Any]], **kwargs) -> List[str]:
        async def _run():
            sem = asyncio.Semaphore(self.concurrency)
            timeout = aiohttp.ClientTimeout(total=3600)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                tasks = [
                    self._call_api(session, sem, q["image_path"], q["text"], **kwargs)
                    for q in queries
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)
            return [str(r) if isinstance(r, Exception) else r for r in results]

        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(_run())
        except RuntimeError:
            return asyncio.run(_run())
