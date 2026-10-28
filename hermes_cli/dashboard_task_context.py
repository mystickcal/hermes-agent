"""Resolve task identity from the authenticated native API process."""
from __future__ import annotations
from types import SimpleNamespace

def resolve_dashboard_task_context(request, profile=None):
    from gateway.native_task_host import native_task_host_context
    marker = request.get("hermes.native_task_bearer_authenticated") if hasattr(request, "get") else False
    if marker is True:
        value = native_task_host_context()
    else:
        # Forward the credential to the native listener; it is the auth authority.
        # Request wrappers from isolated dashboard plugins need not be mappings.
        values = [v for k, v in request.headers.items() if k.lower() == "authorization"]
        auth = values[0] if len(values) == 1 else ""
        if (not isinstance(auth, str) or not auth.startswith("Bearer ")
                or not auth[7:].strip() or len(auth) > 8192 or "\r" in auth or "\n" in auth):
            raise PermissionError("Native task bearer access required")
        import httpx
        from gateway.config import load_gateway_config, Platform
        from gateway.platforms.api_server import listen_address
        config = load_gateway_config()
        platform = config.platforms.get(Platform.API_SERVER)
        _, port = listen_address(platform.extra if platform else {})
        with httpx.Client(timeout=3, follow_redirects=False, trust_env=False) as client:
            response = client.get(f"http://127.0.0.1:{port}/v1/native-task/context",
                                  headers={"Authorization": auth})
        if response.status_code != 200 or len(response.content) > 8192:
            raise PermissionError("Native task identity unavailable")
        value = response.json().get("context")
        local = native_task_host_context()
        keys = ("principal_id", "principal_kind", "profile_name", "profile_home", "store_id")
        if (not isinstance(value, dict) or any(value.get(k) != local[k] for k in keys)
                or not isinstance(value.get("host_epoch"), str) or not value["host_epoch"]):
            raise PermissionError("Native task identity mismatch")
    if profile is not None and profile != value["profile_name"]:
        raise PermissionError("Requested profile is not the served profile")
    return SimpleNamespace(**value)
