from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from headroom.proxy.server import HeadroomProxy, ProxyConfig, create_app


def test_backend_api_responses_alias_delegates_to_openai_handler(monkeypatch):
    async def fake_handle(self, request):  # type: ignore[no-untyped-def]
        return JSONResponse({"ok": True, "path": request.url.path})

    monkeypatch.setattr(HeadroomProxy, "handle_openai_responses", fake_handle)

    with TestClient(create_app(ProxyConfig())) as client:
        response = client.post("/backend-api/responses", json={"model": "gpt-5.3-codex"})

    assert response.status_code == 200
    assert response.json() == {"ok": True, "path": "/backend-api/responses"}
