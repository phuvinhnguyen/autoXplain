import base64
import os
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from autoXplain.utils.vlm.base import VLMBase, vlm


@vlm
class OpenAIClient(VLMBase):
    def __init__(
        self,
        model_name: str = "gpt-4o",
        api_key: str = None,
        base_url: str = None,
        concurrency: int = 8,
    ):
        self.model_name = model_name
        self.concurrency = concurrency
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
        )

    def _encode_image(self, path: str) -> str:
        if path.startswith(('http://', 'https://')):
            return path
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(path)[1].lower().lstrip('.')
        mime = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png', 'webp': 'image/webp'}.get(ext, 'image/jpeg')
        return f"data:{mime};base64,{b64}"

    def _call_api(self, image_path: str, text: str, **kwargs) -> str:
        url = self._encode_image(image_path)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": text},
            ]}],
            max_tokens=kwargs.get("max_tokens", 512),
            temperature=kwargs.get("temperature", 0.1),
        )
        return response.choices[0].message.content

    def query_batch(self, queries: List[Dict[str, Any]], **kwargs) -> List[str]:
        results = [None] * len(queries)
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {
                pool.submit(self._call_api, q["image_path"], q["text"], **kwargs): i
                for i, q in enumerate(queries)
            }
            for future in as_completed(futures):
                i = futures[future]
                try:
                    results[i] = future.result()
                except Exception as e:
                    results[i] = str(e)
        return results
