import os
import re
import json
import httpx
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from app.models import OllamaConfig
from app.ollama import OllamaClient, OllamaError, OllamaResponseError


class BaseLLMProvider(ABC):
    """Abstract interface for LLM completion providers."""

    @abstractmethod
    async def generate_reply(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.65,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        max_tokens: int = 160,
        format_json: bool = True,
    ) -> str:
        """Generates completion text from system and user prompt."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @property
    @abstractmethod
    def is_cloud(self) -> bool:
        pass


class OllamaProvider(BaseLLMProvider):
    """Local Ollama LLM provider (default, privacy-first)."""

    def __init__(self, model: str = "qwen2.5:7b", base_url: str = "http://localhost:11434", timeout: float = 60.0):
        self._model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def is_cloud(self) -> bool:
        return False

    async def generate_reply(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.65,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        max_tokens: int = 160,
        format_json: bool = True,
    ) -> str:
        url = f"{self.base_url}/api/chat"
        options: Dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "repeat_penalty": repeat_penalty,
        }
        if max_tokens:
            options["num_predict"] = max_tokens

        payload: Dict[str, Any] = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": options,
        }
        if format_json:
            payload["format"] = "json"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload)
            if resp.status_code == 404:
                # If model is not found, fallback to checking if another local model exists
                raise OllamaResponseError(
                    f"Model '{self._model}' not found in local Ollama. Run 'ollama pull {self._model}' or switch model in settings."
                )
            if resp.status_code != 200:
                raise OllamaResponseError(f"Ollama error {resp.status_code}: {resp.text}")

            data = resp.json()
            msg = data.get("message", {})
            return msg.get("content", "").strip()


def mask_sensitive_pii(text: str) -> str:
    """Masks phone numbers, email addresses, and URLs before dispatch to external cloud APIs."""
    # Mask phone numbers (+91..., 10 digits, etc.)
    masked = re.sub(r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE]", text)
    # Mask emails
    masked = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]", masked)
    # Mask urls
    masked = re.sub(r"https?://\S+|www\.\S+", "[LINK]", masked)
    return masked


class CloudProvider(BaseLLMProvider):
    """
    Optional Cloud API Provider (Gemini / Groq / OpenAI compatible).
    API key is read strictly from environment variables (e.g. GEMINI_API_KEY, GROQ_API_KEY, OPENAI_API_KEY).
    Masks PII before sending.
    """

    def __init__(
        self,
        provider_type: str = "gemini",
        model: Optional[str] = None,
        api_key_env: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.provider_type = provider_type.lower()
        self.timeout = timeout

        if self.provider_type == "gemini":
            self.api_key_env = api_key_env or "GEMINI_API_KEY"
            self._model = model or "gemini-1.5-flash"
        elif self.provider_type == "groq":
            self.api_key_env = api_key_env or "GROQ_API_KEY"
            self._model = model or "llama-3.3-70b-versatile"
        elif self.provider_type == "openai":
            self.api_key_env = api_key_env or "OPENAI_API_KEY"
            self._model = model or "gpt-4o-mini"
        else:
            self.api_key_env = api_key_env or "CLOUD_API_KEY"
            self._model = model or "default"

    @property
    def model_name(self) -> str:
        return f"{self.provider_type}:{self._model}"

    @property
    def is_cloud(self) -> bool:
        return True

    def _get_api_key(self) -> str:
        key = os.environ.get(self.api_key_env, "").strip()
        if not key and self.provider_type == "gemini":
            from app.config import config
            key = getattr(config, "gemini_api_key", "").strip()
        if not key:
            raise ValueError(
                f"Missing API key for cloud provider '{self.provider_type}'. "
                f"Please set the '{self.api_key_env}' environment variable or paste it in app/config.py."
            )
        return key

    async def generate_reply(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.65,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        max_tokens: int = 160,
        format_json: bool = True,
    ) -> str:
        api_key = self._get_api_key()

        # Mask PII prior to external transmission
        clean_sys = mask_sensitive_pii(system_prompt)
        clean_user = mask_sensitive_pii(user_prompt)

        if self.provider_type == "gemini":
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model}:generateContent?key={api_key}"
            payload = {
                "systemInstruction": {"parts": [{"text": clean_sys}]},
                "contents": [{"parts": [{"text": clean_user}]}],
                "generationConfig": {
                    "temperature": temperature,
                    "topP": top_p,
                    "maxOutputTokens": max_tokens,
                    "responseMimeType": "application/json" if format_json else "text/plain",
                },
            }
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(url, json=payload)
                if resp.status_code != 200:
                    raise OllamaResponseError(f"Gemini API returned error {resp.status_code}: {resp.text}")
                data = resp.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        return parts[0].get("text", "").strip()
                return ""

        elif self.provider_type in ("groq", "openai"):
            base_url = "https://api.groq.com/openai/v1/chat/completions" if self.provider_type == "groq" else "https://api.openai.com/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": clean_sys},
                    {"role": "user", "content": clean_user},
                ],
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
            }
            if format_json:
                payload["response_format"] = {"type": "json_object"}

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(base_url, headers=headers, json=payload)
                if resp.status_code != 200:
                    raise OllamaResponseError(f"{self.provider_type.upper()} API returned error {resp.status_code}: {resp.text}")
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "").strip()
                return ""

        raise ValueError(f"Unsupported cloud provider: {self.provider_type}")


def get_llm_provider(
    provider_name: str = "ollama",
    model_name: Optional[str] = None,
    base_url: str = "http://localhost:11434",
) -> BaseLLMProvider:
    """Factory creating the appropriate LLM provider."""
    p_norm = (provider_name or "ollama").strip().lower()
    if p_norm in ("gemini", "groq", "openai"):
        return CloudProvider(provider_type=p_norm, model=model_name)
    target_model = model_name or "qwen2.5:7b"
    return OllamaProvider(model=target_model, base_url=base_url)
