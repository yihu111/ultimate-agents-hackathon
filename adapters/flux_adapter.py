import os
import time
import base64
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import requests
import asyncio


class FluxAdapter:
    def __init__(
        self,
        *,
        model: str,
        use_raw_mode: bool,
        api_key: Optional[str] = None,
        base_url: str = "https://api.bfl.ai",
        aspect_ratio: Optional[str] = "16:9",
        width: int = 1024,
        height: int = 1024,
        safety_tolerance: int = 6,
        prompt_upsampling: bool = False,
        poll_timeout: int = 180,
        connect_timeout: int = 10,
        read_timeout: int = 120,
        max_post_retries: int = 3,
    ):
        self.api_key = api_key or os.getenv("BFL_API_KEY")
        if not self.api_key:
            raise ValueError("BFL_API_KEY not set")

        self.base_url = base_url.rstrip("/")
        self.model = model
        self.use_raw_mode = use_raw_mode
        self.aspect_ratio = aspect_ratio
        self.width = width
        self.height = height
        self.safety_tolerance = safety_tolerance
        self.prompt_upsampling = prompt_upsampling
        self.poll_timeout = poll_timeout
        self.connect_timeout = connect_timeout
        self.read_timeout = read_timeout
        self.max_post_retries = max_post_retries

        self._session = requests.Session()
        self._session.headers.update({
            "accept": "application/json",
            "x-key": self.api_key,
            "Content-Type": "application/json",
        })

    async def generate(self, prompt_text: str, *, input_image: Optional[str] = None, guidance_scale: Optional[float] = None) -> Tuple[str, Dict]:
        return await asyncio.to_thread(self._generate_sync, prompt_text, input_image, guidance_scale)

    # ---------------- internal (sync) ----------------

    def _generate_sync(self, prompt_text: str, input_image: Optional[str], guidance_scale: Optional[float]) -> Tuple[str, Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "prompt": prompt_text,
            "safety_tolerance": self.safety_tolerance,
            "prompt_upsampling": self.prompt_upsampling,
            "raw": self.use_raw_mode,
        }

        if guidance_scale is not None:
            payload["guidance_scale"] = guidance_scale

        # image+text support: use 'input_image' for Kontext models
        if input_image:
            payload["input_image"] = self._to_data_url_if_needed(input_image)

        if self.aspect_ratio:
            payload["aspect_ratio"] = self.aspect_ratio
        else:
            payload["width"] = self.width
            payload["height"] = self.height

        endpoint = f"{self.base_url}/v1/{self.model}"
        resp = self._post_with_retries(endpoint, payload)
        data = resp.json()
        request_id = data["id"]
        polling_url = data.get("polling_url", f"{self.base_url}/v1/get_result")

        result = self._poll_for_result(polling_url, request_id, self.poll_timeout)
        sample = result.get("result", {}).get("sample")
        if not sample:
            raise RuntimeError(f"Missing sample in result: {result}")

        meta = {
            "request_id": request_id,
            "model": self.model,
            "result": result.get("result", {}),
        }
        return sample, meta

    def _post_with_retries(self, url: str, json_payload: Dict[str, Any]) -> requests.Response:
        last_exc = None
        for attempt in range(self.max_post_retries):
            try:
                resp = self._session.post(url, json=json_payload, timeout=(self.connect_timeout, self.read_timeout))
                resp.raise_for_status()
                return resp
            except requests.exceptions.ReadTimeout as e:
                last_exc = e
            except requests.RequestException as e:
                last_exc = e
            time.sleep(1.5 * (attempt + 1))
        assert last_exc is not None
        raise last_exc

    def _poll_for_result(self, polling_url: str, request_id: str, max_wait: int) -> Dict[str, Any]:
        start = time.time()
        while time.time() - start < max_wait:
            time.sleep(0.5)
            try:
                r = self._session.get(polling_url, params={"id": request_id}, timeout=5)
                r.raise_for_status()
                result = r.json()
            except requests.exceptions.Timeout:
                continue
            except requests.RequestException:
                continue

            status = result.get("status")
            if status == "Ready":
                return result
            if status in ("Error", "Failed"):
                raise RuntimeError(f"Generation failed: {result}")
        raise TimeoutError(f"Request {request_id} timed out after {max_wait}s")

    def _to_data_url_if_needed(self, path_or_url: str) -> str:
        if path_or_url.startswith(("data:", "http://", "https://")):
            return path_or_url
        p = Path(path_or_url)
        if not p.is_file():
            return path_or_url
        ext = p.suffix.lower().lstrip(".") or "png"
        mime = "image/png" if ext == "png" else f"image/{ext}"
        data = base64.b64encode(p.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{data}"


