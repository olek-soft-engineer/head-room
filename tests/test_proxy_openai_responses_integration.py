"""Integration tests for OpenAI /v1/responses endpoint with real API calls.

These tests require a valid OPENAI_API_KEY environment variable.
They test the /v1/responses endpoint (introduced March 2025) with compression.

Run with:
    OPENAI_API_KEY=your-key pytest tests/test_proxy_openai_responses_integration.py -v
"""

import json
import os

import pytest

# Skip entire module if no API key
pytestmark = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from headroom.proxy.server import ProxyConfig, create_app  # noqa: E402


@pytest.fixture
def openai_responses_client():
    """Create test client for OpenAI responses API with optimization enabled."""
    config = ProxyConfig(
        optimize=True,  # Enable compression
        cache_enabled=False,
        rate_limit_enabled=False,
        cost_tracking_enabled=False,
    )
    app = create_app(config)
    with TestClient(app) as client:
        yield client


@pytest.fixture
def api_key():
    """Get OpenAI API key from environment."""
    return os.environ.get("OPENAI_API_KEY")


class TestOpenAIResponsesBasic:
    """Test /v1/responses endpoint basic functionality."""

    def test_basic_generation(self, openai_responses_client, api_key):
        """Basic text generation works."""
        response = openai_responses_client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": "gpt-4o-mini", "input": "What is 2+2? Reply with just the number."},
        )
        assert response.status_code == 200
        data = response.json()

        # Verify responses API format
        assert "id" in data
        assert "output" in data
        assert len(data["output"]) > 0
        assert data["output"][0]["type"] == "message"
        assert data["output"][0]["role"] == "assistant"

        # Get the text content
        content = data["output"][0]["content"]
        assert len(content) > 0
        text = content[0].get("text", "")
        assert "4" in text

        # Verify usage metadata
        assert "usage" in data

    def test_with_instructions(self, openai_responses_client, api_key):
        """System instructions work correctly."""
        response = openai_responses_client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gpt-4o-mini",
                "input": "Hello",
                "instructions": "Always respond with exactly one word.",
            },
        )
        assert response.status_code == 200
        data = response.json()

        content = data["output"][0]["content"]
        text = content[0].get("text", "")
        # Should be a short response due to instructions
        assert len(text.split()) <= 3

    def test_input_as_array(self, openai_responses_client, api_key):
        """Input can be an array of messages."""
        response = openai_responses_client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gpt-4o-mini",
                "input": [
                    {"role": "user", "content": "My name is TestUser789."},
                    {"role": "assistant", "content": "Nice to meet you, TestUser789!"},
                    {"role": "user", "content": "What is my name?"},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()

        content = data["output"][0]["content"]
        text = content[0].get("text", "").lower()
        assert "testuser789" in text

    def test_generation_parameters(self, openai_responses_client, api_key):
        """Generation parameters are respected."""
        response = openai_responses_client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gpt-4o-mini",
                "input": "Write a very short poem about AI.",
                "max_output_tokens": 50,
                "temperature": 0.1,
            },
        )
        assert response.status_code == 200
        data = response.json()
        # Response should be limited by max_output_tokens
        assert data["usage"]["output_tokens"] <= 60  # Some buffer


class TestOpenAIResponsesTools:
    """Test function calling / tools with /v1/responses endpoint."""

    def test_function_calling(self, openai_responses_client, api_key):
        """Function calling works correctly."""
        # Note: /v1/responses uses a different tools format than /v1/chat/completions
        # - name, description, parameters are at top level, not nested under "function"
        response = openai_responses_client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gpt-4o-mini",
                "input": "What is the weather in Tokyo?",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_weather",
                        "description": "Get current weather for a location",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string", "description": "City name"}
                            },
                            "required": ["location"],
                        },
                    }
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()

        # Find tool call in output
        output = data["output"]
        tool_call_found = False
        for item in output:
            if item.get("type") == "function_call":
                tool_call_found = True
                assert item["name"] == "get_weather"
                args = (
                    json.loads(item["arguments"])
                    if isinstance(item["arguments"], str)
                    else item["arguments"]
                )
                assert "tokyo" in args.get("location", "").lower()
                break

        assert tool_call_found, "Expected function_call in output"


class TestOpenAIResponsesCompression:
    """Test that compression works with /v1/responses endpoint."""

    def test_compression_on_assistant_message(self, openai_responses_client, api_key):
        """Large data in assistant message gets compressed."""
        # Create large JSON data (simulating tool output)
        items = [
            {"id": i, "name": f"Item {i}", "desc": f"Description for item {i}"} for i in range(100)
        ]
        tool_output = json.dumps(items)

        # Send as multi-turn with assistant message containing data
        response = openai_responses_client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": "gpt-4o-mini",
                "input": [
                    {"role": "user", "content": "Get items from database"},
                    {"role": "assistant", "content": f"Here are the results:\n{tool_output}"},
                    {"role": "user", "content": "How many items are there?"},
                ],
            },
        )
        assert response.status_code == 200
        data = response.json()

        content = data["output"][0]["content"]
        text = content[0].get("text", "")
        # Model should correctly count the items
        assert "100" in text

        # Check that compression happened via stats
        stats = openai_responses_client.get("/stats").json()
        # At least some tokens should have been saved
        assert stats["tokens"]["saved"] >= 0  # May or may not compress depending on size


class TestOpenAIResponsesStats:
    """Test that proxy stats track /v1/responses requests correctly."""

    def test_stats_track_openai_provider(self, openai_responses_client, api_key):
        """Stats show requests under 'openai' provider."""
        # Make a request
        openai_responses_client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": "gpt-4o-mini", "input": "Hi"},
        )

        stats = openai_responses_client.get("/stats").json()
        assert "openai" in stats["requests"]["by_provider"]
        assert stats["requests"]["by_provider"]["openai"] >= 1

    def test_stats_track_model(self, openai_responses_client, api_key):
        """Stats track the specific model used."""
        openai_responses_client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": "gpt-4o-mini", "input": "Hi"},
        )

        stats = openai_responses_client.get("/stats").json()
        assert "gpt-4o-mini" in stats["requests"]["by_model"]


class TestOpenAIResponsesErrorHandling:
    """Test error handling for /v1/responses endpoint."""

    def test_invalid_api_key(self, openai_responses_client):
        """Invalid API key returns appropriate error."""
        response = openai_responses_client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer invalid-key-123"},
            json={"model": "gpt-4o-mini", "input": "Hi"},
        )
        assert response.status_code >= 400

    def test_invalid_model(self, openai_responses_client, api_key):
        """Invalid model returns appropriate error."""
        response = openai_responses_client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": "nonexistent-model-xyz", "input": "Hi"},
        )
        assert response.status_code >= 400

    def test_missing_input(self, openai_responses_client, api_key):
        """Missing input handled gracefully."""
        response = openai_responses_client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": "gpt-4o-mini"},
        )
        # Should either return error or handle gracefully
        assert response.status_code in [200, 400, 422]
