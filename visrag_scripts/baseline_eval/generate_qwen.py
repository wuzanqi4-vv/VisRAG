"""Qwen-VL multi-modal generation module with checkpoint resume and rate-limiting.

Calls OpenAI-compatible DashScope API for Qwen-VL models, supports passing multiple
top-5 retrieved document images encoded as base64 or local paths, with retry and
exponential backoff.
"""

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import requests


def encode_image_to_base64(image_path: str) -> str:
    """Reads a local image and converts it to a base64 data URI."""
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    ext = Path(image_path).suffix.lower().replace(".", "")
    mime_type = "jpeg" if ext in ["jpg", "jpeg"] else ext
    return f"data:image/{mime_type};base64,{encoded}"


class QwenGenerator:
    """Wrapper for calling Qwen-VL models with multi-image inputs."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY", "")
        self.base_url = (
            base_url
            or os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
        ).rstrip("/")
        self.model_name = model_name or os.getenv("QWEN_VL_MODEL", "qwen-vl-max")

        if not self.api_key:
            print("[Warning] DASHSCOPE_API_KEY is not set. API calls will fail unless provided.")

    def generate_answer(
        self,
        question: str,
        image_paths: List[str],
        max_retries: int = 5,
        retry_delay: float = 3.0,
    ) -> Dict[str, Any]:
        """Submits multi-image prompt and question to Qwen-VL model.

        Args:
            question: The benchmark question to answer.
            image_paths: List of top-k retrieved image paths.
            max_retries: Max number of API retries on transient errors.
            retry_delay: Delay between retries in seconds.

        Returns:
            Dictionary containing prediction, raw response, token counts, and latency.
        """
        # Build multi-modal message content
        content_items = []
        for idx, img_p in enumerate(image_paths):
            data_uri = encode_image_to_base64(img_p)
            content_items.append({"type": "text", "text": f"[Page {idx + 1} Image]:"})
            content_items.append({
                "type": "image_url",
                "image_url": {"url": data_uri},
            })

        prompt_text = (
            f"Based on the {len(image_paths)} document page image(s) provided above, "
            f"answer the following question accurately and concisely.\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        content_items.append({"type": "text", "text": prompt_text})

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": content_items,
                }
            ],
            "temperature": 0.0,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        url = f"{self.base_url}/chat/completions"

        for attempt in range(1, max_retries + 1):
            start_time = time.time()
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=60)
                latency = time.time() - start_time

                if resp.status_code == 200:
                    res_json = resp.json()
                    choice = res_json.get("choices", [{}])[0]
                    prediction = choice.get("message", {}).get("content", "").strip()
                    usage = res_json.get("usage", {})
                    return {
                        "prediction": prediction,
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                        "latency_sec": latency,
                        "status": "success",
                    }
                else:
                    print(f"[API Attempt {attempt}/{max_retries}] Status {resp.status_code}: {resp.text}")
                    if attempt < max_retries:
                        time.sleep(retry_delay * attempt)
            except Exception as e:
                print(f"[API Attempt {attempt}/{max_retries}] Request exception: {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)

        return {
            "prediction": "",
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_sec": 0.0,
            "status": "failed",
            "error": "Exceeded max retries",
        }
