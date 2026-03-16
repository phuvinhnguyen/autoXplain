import base64
import os
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import google.generativeai as genai
from autoXplain.utils.vlm.base import VLMBase, vlm


@vlm
class GeminiClient(VLMBase):
    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        api_key: str = None,
        concurrency: int = 8,
    ):
        self.model_name = model_name
        self.concurrency = concurrency
        genai.configure(api_key=api_key or os.getenv("GEMINI_API_KEY"))
        self.model = genai.GenerativeModel(model_name)

    def _load_image(self, path: str):
        if path.startswith(('http://', 'https://')):
            import httpx
            data = httpx.get(path).content
            ext = path.split('?')[0].split('.')[-1].lower()
        else:
            with open(path, "rb") as f:
                data = f.read()
            ext = Path(path).suffix.lower().lstrip('.')
        mime = {'jpg': 'image/jpeg', 'jpeg': 'image/jpeg', 'png': 'image/png', 'webp': 'image/webp'}.get(ext, 'image/jpeg')
        return {"mime_type": mime, "data": base64.b64encode(data).decode()}

    def _call_api(self, image_path: str, text: str, **kwargs) -> str:
        image = self._load_image(image_path)
        response = self.model.generate_content(
            [{"mime_type": image["mime_type"], "data": base64.b64decode(image["data"])}, text],
            generation_config=genai.GenerationConfig(
                max_output_tokens=kwargs.get("max_tokens", 512),
                temperature=kwargs.get("temperature", 0.1),
            )
        )
        return response.text

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
