import json
import logging
import httpx
from typing import List, Optional, Dict, Any
from app.models import OllamaStatus, OllamaConfig

logger = logging.getLogger("app.ollama")


class OllamaError(Exception):
    """Base exception for all Ollama-related failures."""
    pass


class OllamaConnectionError(OllamaError):
    """Raised when the Ollama HTTP server cannot be reached."""
    pass


class OllamaModelNotFoundError(OllamaError):
    """Raised when the requested model is not downloaded or found locally."""
    pass


class OllamaTimeoutError(OllamaError):
    """Raised when Ollama request exceeds configured timeout."""
    pass


class OllamaResponseError(OllamaError):
    """Raised when Ollama returns an unexpected status code or response format."""
    pass


class OllamaClient:
    """
    Local-only HTTP client for Ollama API.
    Guarantees no external/cloud telemetry or fallback.
    """

    def __init__(self, config: Optional[OllamaConfig] = None):
        self.config = config or OllamaConfig()

    @property
    def base_url(self) -> str:
        return self.config.base_url.rstrip("/")

    async def check_health(self) -> OllamaStatus:
        """
        Pings Ollama server and lists available local models.
        Never raises exceptions; returns an OllamaStatus object.
        """
        try:
            models = await self.list_models()
            return OllamaStatus(
                is_connected=True,
                available_models=models,
                error_message=None,
            )
        except OllamaConnectionError as e:
            return OllamaStatus(
                is_connected=False,
                available_models=[],
                error_message=str(e),
            )
        except Exception as e:
            return OllamaStatus(
                is_connected=False,
                available_models=[],
                error_message=f"Error connecting to Ollama: {str(e)}",
            )

    async def list_models(self) -> List[str]:
        """
        Retrieves installed model names from GET /api/tags.
        """
        url = f"{self.base_url}/api/tags"
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                resp = await client.get(url)

                if resp.status_code != 200:
                    raise OllamaResponseError(
                        f"Failed to fetch models from Ollama (status {resp.status_code}): {resp.text}"
                    )

                data = resp.json()
                raw_models = data.get("models", [])
                model_names = [m.get("name") for m in raw_models if isinstance(m, dict) and m.get("name")]
                return sorted(model_names)

        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama at '{self.base_url}'. "
                "Ensure Ollama is running locally ('ollama serve' or start Ollama.app)."
            ) from e
        except httpx.TimeoutException as e:
            raise OllamaTimeoutError(
                f"Timed out contacting Ollama at '{self.base_url}' after {self.config.timeout_seconds}s."
            ) from e
        except httpx.HTTPError as e:
            raise OllamaResponseError(f"HTTP error communicating with Ollama: {str(e)}") from e

    async def generate_chat(
        self,
        system_prompt: str,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.65,
        max_tokens: Optional[int] = 160,
        format_json: bool = False,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
    ) -> str:
        """
        Sends chat completion request to POST /api/chat.
        Strictly local: never redirects to any third-party or cloud provider.
        """
        target_model = model or self.config.model
        url = f"{self.base_url}/api/chat"

        options: Dict[str, Any] = {
            "temperature": temperature,
            "repeat_penalty": repeat_penalty,
            "top_p": top_p,
        }
        if max_tokens:
            options["num_predict"] = max_tokens

        payload: Dict[str, Any] = {
            "model": target_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": options,
            "keep_alive": "15m",
        }
        if format_json:
            payload["format"] = "json"

        # Console and Logger diagnostics
        logger.info("[OLLAMA REQUEST] POST %s | Model: %s", url, target_model)
        print(f"\n[OLLAMA REQUEST] POST {url} | Model: {target_model} | Keep-Alive: 15m")
        print(f"[OLLAMA PAYLOAD] {json.dumps(payload, indent=2)}\n")

        try:
            async with httpx.AsyncClient(timeout=self.config.timeout_seconds) as client:
                resp = await client.post(url, json=payload)

                print(f"[OLLAMA RESPONSE {resp.status_code}] {resp.text[:500]}\n")
                logger.info("[OLLAMA RESPONSE %s] %s", resp.status_code, resp.text[:500])

                if resp.status_code == 404:
                    err_msg = resp.text
                    raise OllamaModelNotFoundError(
                        f"Model '{target_model}' was not found in your local Ollama library. "
                        f"Run 'ollama pull {target_model}' in your terminal to install it, "
                        "or pick another model in settings."
                    )

                if resp.status_code != 200:
                    raise OllamaResponseError(
                        f"Ollama returned error {resp.status_code}: {resp.text}"
                    )

                data = resp.json()
                message_content = data.get("message", {}).get("content", "")
                if not message_content:
                    raise OllamaResponseError("Ollama returned an empty response.")

                return message_content

        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            err = OllamaConnectionError(
                f"Cannot connect to Ollama at '{self.base_url}'. "
                "Ensure Ollama is running locally ('ollama serve')."
            )
            print(f"[OLLAMA CONNECTION ERROR] {err}")
            raise err from e
        except httpx.TimeoutException as e:
            err = OllamaTimeoutError(
                f"Ollama took longer than {self.config.timeout_seconds}s to respond for model '{target_model}'."
            )
            print(f"[OLLAMA TIMEOUT ERROR] {err}")
            raise err from e
        except OllamaError as e:
            print(f"[OLLAMA ERROR] {e}")
            raise
        except Exception as e:
            err = OllamaResponseError(f"Unexpected error calling Ollama: {str(e)}")
            print(f"[OLLAMA UNEXPECTED ERROR] {err}")
            raise err from e
