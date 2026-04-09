from fastapi import WebSocket
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from headroom.proxy.server import HeadroomProxy, ProxyConfig, create_app


def test_codex_responses_aliases_delegate_to_openai_handler(monkeypatch):
    async def fake_handle(self, request):  # type: ignore[no-untyped-def]
        return JSONResponse({"ok": True, "path": request.url.path})

    monkeypatch.setattr(HeadroomProxy, "handle_openai_responses", fake_handle)

    with TestClient(create_app(ProxyConfig())) as client:
        for path in ("/backend-api/responses", "/backend-api/codex/responses"):
            response = client.post(path, json={"model": "gpt-5.3-codex"})
            assert response.status_code == 200
            assert response.json() == {"ok": True, "path": path}


def test_codex_responses_websocket_aliases_delegate_to_openai_handler(monkeypatch):
    seen_paths: list[str] = []

    async def fake_handle_ws(self, websocket: WebSocket):  # type: ignore[no-untyped-def]
        seen_paths.append(websocket.url.path)
        await websocket.accept()
        await websocket.send_json({"ok": True, "path": websocket.url.path})
        await websocket.close()

    monkeypatch.setattr(HeadroomProxy, "handle_openai_responses_ws", fake_handle_ws)

    with TestClient(create_app(ProxyConfig())) as client:
        for path in ("/backend-api/responses", "/backend-api/codex/responses"):
            with client.websocket_connect(path) as websocket:
                assert websocket.receive_json() == {"ok": True, "path": path}

    assert seen_paths == ["/backend-api/responses", "/backend-api/codex/responses"]
