"""Authenticated native task-host HTTP contracts and capability projection."""
from __future__ import annotations
import asyncio
import json
from functools import partial
from aiohttp import web
from gateway.native_task_host import get_native_task_host_provider, native_task_host_context

def _error(code, status=403, *, retryable=False):
    return web.json_response({"error": code, "retryable": retryable}, status=status)

def _authorize(self, request):
    error = self._check_auth(request)
    if error is not None:
        return None, error
    from hermes_cli.auth import has_usable_secret
    if not has_usable_secret(self._expected_api_key(), min_length=16):
        return None, _error("unauthorized", 401)
    request["hermes.native_task_bearer_authenticated"] = True
    try:
        return native_task_host_context(), None
    except (OSError, ValueError):
        return None, _error("unavailable", 503, retryable=True)

def _context(request):
    from hermes_cli.dashboard_task_context import resolve_dashboard_task_context
    return vars(resolve_dashboard_task_context(request))

def _project(operation, result):
    if not isinstance(result, dict):
        return _error("unavailable", 502, retryable=True)
    if result.get("ok") is not True:
        status = result.get("status", 403)
        if status not in {400, 403, 404, 409, 429, 503}:
            status = 403
        code = "unauthorized" if status == 403 else str(result.get("error", "invalid_input"))
        return _error(code, status, retryable=status in {429, 503})
    value = {k: v for k, v in result.items() if k != "ok"}
    if operation == "history.capabilities":
        value = value.get("capabilities", {})
    return web.json_response(value)

async def _proof_context(self, request, provider, context, body):
    locator = await asyncio.to_thread(provider.invoke, "discord.locate", context, {"proof": body["proof"]})
    if not isinstance(locator, dict) or locator.get("ok") is not True:
        return None
    from plugins.platforms.discord.task_voice_context import current_discord_room
    adapter = self._get_platform_callback_adapter(request, "discord")
    room = current_discord_room(adapter, locator)
    return {**context, "room": room} if room else None

async def _call(self, request, operation, fields=None):
    context, error = _authorize(self, request)
    if error is not None:
        return error
    provider = get_native_task_host_provider(context["profile_home"])
    if provider is None:
        return _error("unavailable", 503, retryable=True)
    body = {}
    if request.method != "GET":
        body, error = await self._read_json_body(request)
        if error is not None:
            return error
    if (not isinstance(body, dict) or len(json.dumps(body).encode()) > 163840
            or (fields is not None and (set(body) != fields
                or any(not isinstance(v, str) or not v or len(v) > 256 for v in body.values())))):
        return _error("invalid_input", 400)
    try:
        if operation.startswith("discord.") and operation != "discord.revoke":
            context = await _proof_context(self, request, provider, context, body)
            if context is None:
                return _error("unauthorized", 403)
        result = await asyncio.to_thread(provider.invoke, operation, context, body)
    except Exception:
        return _error("unavailable", 502, retryable=True)
    return _project(operation, result)

async def _handle_context(self, request):
    context, error = _authorize(self, request)
    if error is not None:
        return error
    return web.json_response({"object": "hermes.native_task.context", "context": context})

def _http_routes(self):
    routes = [("GET", "/v1/native-task/context", partial(_handle_context, self))]
    for action in ("redeem", "verify", "rebind", "revoke"):
        fields = {"proof", "session_id", "binding_id"}
        if action == "rebind":
            fields |= {"next_session_id", "next_binding_id"}
        routes.append(("POST", f"/v1/task-context/discord/{action}",
                       partial(_call, self, operation="discord." + action, fields=fields)))
    routes.append(("GET", "/v1/passive-history/capabilities",
                   partial(_call, self, operation="history.capabilities")))
    for action in ("attach", "snapshot", "commit", "reconcile", "detach"):
        routes.append(("POST", f"/v1/passive-history/{action}",
                       partial(_call, self, operation="history." + action)))
    return routes

async def native_task_features(self):
    try:
        context = native_task_host_context()
        provider = get_native_task_host_provider(context["profile_home"])
        if provider is None:
            return {}
        result = await asyncio.to_thread(provider.invoke, "capabilities", context, {})
        ops = set(result.get("operations", ())) if result.get("ok") is True else set()
        from hermes_state import SessionDB
        history = {"history." + v for v in ("attach", "snapshot", "commit", "reconcile", "detach")}
        healthy = history <= ops and callable(getattr(SessionDB, "append_messages_with_receipt", None))
        return {
            "passive_history": {"version": 1, "supported": healthy},
            "linked_child_dispatch": {
                "version": 1, "supported": healthy and {"child.prepare", "child.accepted", "child.authorize"} <= ops,
                "separate_child_goal": True, "origin_sources": ["fresh", "passive_receipt"],
            },
        }
    except (OSError, ValueError, TypeError, AttributeError):
        return {}

async def _handle_capabilities(self, request: "web.Request") -> "web.Response":
    """GET /v1/capabilities — the stable, machine-readable API surface for external UIs."""
    from gateway.platforms.api_server import (
        RunIdempotencyStore, _api_runs, _STATIC_FEATURE_FLAGS,
        _BROWSER_CONTROL_PROTOCOL_VERSION, BROWSER_CONTROL_CAPABILITIES,
        BROWSER_CONTROL_ARTIFACT_CAPABILITIES, BROWSER_CONTROL_DEVELOPER_CAPABILITIES,
        DEFAULT_MAX_ARTIFACT_BYTES, DEFAULT_ARTIFACT_TTL_SECONDS,
        DEFAULT_ALLOWED_MIME_TYPES, _CAPABILITY_ENDPOINTS,
    )
    return web.json_response({
        "object": "hermes.api_server.capabilities", "platform": "hermes-agent",
        "model": self._model_name,
        "auth": {"type": "bearer", "required": bool(self._api_key)},
        "runtime": {
            "mode": "server_agent", "tool_execution": "server", "split_runtime": False,
            "description": (
                "The API server creates a server-side Hermes AIAgent; "
                "tools execute on the API-server host unless a future "
                "explicit split-runtime mode is enabled.")},
        "features": {
            "chat_completions": True, "chat_completions_streaming": True,
            "responses_api": True, "responses_streaming": True, "run_submission": True,
            "runs_idempotency": _api_runs._idempotency_capabilities(self, store_type=RunIdempotencyStore),
            **await native_task_features(self),
            **_STATIC_FEATURE_FLAGS,
            "cors": bool(self._cors_origins),
            # Always advertised for feature-detection; enabled follows config.
            "browser_extension_control": {
                "enabled": self._browser_control_enabled(),
                "protocol_version": _BROWSER_CONTROL_PROTOCOL_VERSION,
                "capabilities": sorted(BROWSER_CONTROL_CAPABILITIES),
                "artifact_capabilities": sorted(BROWSER_CONTROL_ARTIFACT_CAPABILITIES),
                "developer_capabilities": sorted(BROWSER_CONTROL_DEVELOPER_CAPABILITIES),
                "developer_mode": self._browser_control_developer_mode(),
                "artifact_transport": {
                    "upload": {"method": "POST", "path": "/v1/artifacts/upload"},
                    "download": {
                        "method": "GET", "path": "/v1/artifacts/download/{artifact_id}"},
                    "max_bytes": DEFAULT_MAX_ARTIFACT_BYTES,
                    "ttl_seconds": DEFAULT_ARTIFACT_TTL_SECONDS,
                    "allowed_mime_types": sorted(DEFAULT_ALLOWED_MIME_TYPES)},
                "real_browser_actions": True,
                "transports": {
                    "local_vps": "websocket-subprotocol-ticket",
                    "cloud": "authenticated-gateway-rpc"}}},
        "endpoints": {name: {"method": m, "path": p} for name, (m, p) in _CAPABILITY_ENDPOINTS},
    })
