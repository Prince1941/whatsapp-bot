import pytest
import httpx
from unittest.mock import patch, MagicMock, AsyncMock
from app.models import OllamaConfig
from app.ollama import (
    OllamaClient,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaResponseError,
)


@pytest.mark.asyncio
async def test_ollama_healthcheck_success():
    client = OllamaClient(OllamaConfig())

    mock_resp = httpx.Response(
        status_code=200,
        json={"models": [{"name": "llama3.2:latest"}, {"name": "mistral:latest"}]},
        request=httpx.Request("GET", "http://localhost:11434/api/tags"),
    )

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.return_value = mock_resp
        status = await client.check_health()
        assert status.is_connected is True
        assert "llama3.2:latest" in status.available_models
        assert "mistral:latest" in status.available_models
        assert status.error_message is None


@pytest.mark.asyncio
async def test_ollama_healthcheck_offline():
    client = OllamaClient(OllamaConfig())

    with patch.object(httpx.AsyncClient, "get", new_callable=AsyncMock) as mock_get:
        mock_get.side_effect = httpx.ConnectError("Connection refused")
        status = await client.check_health()
        assert status.is_connected is False
        assert status.available_models == []
        assert "Cannot connect to Ollama" in status.error_message


@pytest.mark.asyncio
async def test_generate_chat_success():
    client = OllamaClient(OllamaConfig(model="llama3.2"))

    mock_resp = httpx.Response(
        status_code=200,
        json={"message": {"role": "assistant", "content": "=== DRAFT 1 ===\nhey sounds good!"}},
        request=httpx.Request("POST", "http://localhost:11434/api/chat"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        result = await client.generate_chat("system prompt", "user prompt")
        assert "hey sounds good!" in result


@pytest.mark.asyncio
async def test_generate_chat_model_missing():
    client = OllamaClient(OllamaConfig(model="nonexistent-model"))

    mock_resp = httpx.Response(
        status_code=404,
        text="model 'nonexistent-model' not found, try pulling it first",
        request=httpx.Request("POST", "http://localhost:11434/api/chat"),
    )

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        with pytest.raises(OllamaModelNotFoundError) as exc_info:
            await client.generate_chat("system prompt", "user prompt")
        assert "was not found in your local Ollama library" in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_chat_timeout():
    client = OllamaClient(OllamaConfig(timeout_seconds=5.0))

    with patch.object(httpx.AsyncClient, "post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.TimeoutException("Read timeout")
        with pytest.raises(OllamaTimeoutError) as exc_info:
            await client.generate_chat("system prompt", "user prompt")
        assert "took longer than" in str(exc_info.value)
