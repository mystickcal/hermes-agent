from gateway.native_task_host import (
    NativeTaskHostProvider, get_native_task_host_provider, register_native_task_host_provider,
    restore_native_task_host_provider, snapshot_native_task_host_provider,
)


class Provider(NativeTaskHostProvider):
    def invoke(self, operation, context, payload):
        return {"operation": operation, "context": context, "payload": payload}


def test_provider_is_scoped_and_restoreable(tmp_path):
    first, second = Provider(), Provider()
    scope = tmp_path / "profile"
    register_native_task_host_provider(first, profile_home=scope)
    previous = snapshot_native_task_host_provider(scope)
    register_native_task_host_provider(second, profile_home=scope)
    assert get_native_task_host_provider(scope) is second
    assert restore_native_task_host_provider(previous, profile_home=scope)
    assert get_native_task_host_provider(scope) is first
    restore_native_task_host_provider(None, profile_home=scope)
    assert get_native_task_host_provider(scope) is None


def test_native_context_and_forwarded_dashboard_use_same_identity(tmp_path, monkeypatch):
    import asyncio
    import json
    import sqlite3
    from types import SimpleNamespace
    import httpx
    from aiohttp.test_utils import make_mocked_request
    from gateway.config import PlatformConfig
    from gateway.native_task_host import native_task_host_context
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.platforms.api_server_native_tasks import _handle_context
    from hermes_cli.dashboard_task_context import resolve_dashboard_task_context

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    key = "native-test-bearer-credential-123456"
    monkeypatch.setenv("API_SERVER_KEY", key)
    sqlite3.connect(tmp_path / "state.db").close()
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    request = make_mocked_request("GET", "/v1/native-task/context", headers={"Authorization": "Bearer " + key})
    response = asyncio.run(_handle_context(adapter, request))
    assert response.status == 200
    payload = json.loads(response.body)
    assert payload["context"] == native_task_host_context()
    assert vars(resolve_dashboard_task_context(request)) == payload["context"]
    seen = []
    def handle(forwarded):
        seen.append(forwarded)
        return httpx.Response(200, json=payload)
    client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kw: client(transport=httpx.MockTransport(handle), **kw))
    wrapper = SimpleNamespace(headers={"authorization": "Bearer " + key})
    actual = resolve_dashboard_task_context(wrapper, profile=payload["context"]["profile_name"])
    assert vars(actual) == payload["context"]
    assert seen[0].url.path == "/v1/native-task/context"
    assert seen[0].headers["Authorization"] == "Bearer " + key


def test_native_context_refuses_missing_wrong_and_unconfigured_bearer(tmp_path, monkeypatch):
    import asyncio
    import sqlite3
    from aiohttp.test_utils import make_mocked_request
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from gateway.platforms.api_server_native_tasks import _handle_context
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sqlite3.connect(tmp_path / "state.db").close()
    monkeypatch.setenv("API_SERVER_KEY", "native-test-bearer-credential-123456")
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    for headers in ({}, {"Authorization": "Bearer wrong"}):
        request = make_mocked_request("GET", "/v1/native-task/context", headers=headers)
        assert asyncio.run(_handle_context(adapter, request)).status == 401
    monkeypatch.delenv("API_SERVER_KEY")
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    request = make_mocked_request("GET", "/v1/native-task/context")
    assert asyncio.run(_handle_context(adapter, request)).status == 401


def test_native_http_projection_preserves_authenticated_missing_target():
    import json
    from gateway.platforms.api_server_native_tasks import _project
    response = _project("history.snapshot", {"ok": False, "error": "target_missing", "status": 404})
    assert response.status == 404
    assert json.loads(response.body) == {"error": "target_missing", "retryable": False}
