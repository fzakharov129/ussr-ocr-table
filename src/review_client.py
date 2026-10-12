"""Shared Responses API client configured for Yandex or Alibaba."""
import os
from openai import DefaultHttpxClient, OpenAI
from .validations import ReviewChecks


class ReviewClient:
    def __init__(self, config: dict):
        ReviewChecks.config(config)
        self.config = dict(config)
        self.provider = config.get("provider", "yandex")
        api_key = os.getenv(config["api_key_env"], "")
        base_url = os.getenv(config["base_url_env"], "")
        folder = os.getenv(config.get("folder_id_env", ""), "")
        ReviewChecks.credentials(api_key, base_url, folder, provider=self.provider)
        self.model = (f"gpt://{folder}/{config['model']}"
                      if self.provider == "yandex" else config["model"])
        self._client = OpenAI(
            api_key=api_key, base_url=base_url,
            project=folder if self.provider == "yandex" else None,
            timeout=config["timeout"], max_retries=config["max_retries"],
            http_client=DefaultHttpxClient(trust_env=False, timeout=config["timeout"]),
        )

    @property
    def base_url(self):
        return self._client.base_url

    @property
    def responses(self):
        """Compatibility with existing terminal diagnostics."""
        return self._client.responses

    def recognize(self, image_url: str, prompt: str):
        return self._request([{"role": "user", "content": [
            {"type": "input_image", "image_url": image_url},
        ]}], prompt)

    def review_text(self, text: str, prompt: str):
        return self._request(text, prompt)

    def _request(self, input_data, prompt):
        options = dict(
            model=self.model, temperature=self.config["temperature"],
            max_output_tokens=self.config["max_output_tokens"],
            instructions=prompt, input=input_data,
        )
        if self.provider == "alibaba":
            options["extra_body"] = {"enable_thinking": self.config.get("enable_thinking", False)}
        with ReviewChecks.api_call():
            return self._client.responses.create(**options)

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
