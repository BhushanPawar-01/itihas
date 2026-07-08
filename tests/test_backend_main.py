import sys
import types

from fastapi.testclient import TestClient


def _install_graph_stub(monkeypatch):
    graph_module = types.ModuleType("src.agents.graph")

    async def run_query(*args, **kwargs):
        return {}

    def run_query_streaming(*args, **kwargs):
        return []

    graph_module.run_query = run_query
    graph_module.run_query_streaming = run_query_streaming
    monkeypatch.setitem(sys.modules, "src.agents.graph", graph_module)


def test_root_serves_index_with_no_store_headers(tmp_path, monkeypatch):
    _install_graph_stub(monkeypatch)

    import backend.main as backend_main

    index = tmp_path / "index.html"
    index.write_text("<html>itihas</html>", encoding="utf-8")
    monkeypatch.setattr(backend_main, "FRONTEND_DIST", tmp_path)

    client = TestClient(backend_main.app)
    response = client.get("/")

    assert response.status_code == 200
    assert response.text == "<html>itihas</html>"
    assert "no-store" in response.headers.get("cache-control", "")


def test_query_alias_is_available_without_405(tmp_path, monkeypatch):
    _install_graph_stub(monkeypatch)

    import backend.main as backend_main

    index = tmp_path / "index.html"
    index.write_text("<html>itihas</html>", encoding="utf-8")
    monkeypatch.setattr(backend_main, "FRONTEND_DIST", tmp_path)

    client = TestClient(backend_main.app)
    response = client.post("/query", json={})

    assert response.status_code != 405
