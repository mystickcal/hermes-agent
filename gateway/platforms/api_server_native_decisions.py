"""Bounded, tool-free decisions over a confirmed subscription model route."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid

from aiohttp import web

MODEL = "gpt-6.1-terra"
PROVIDER = "openai-codex"
MODEL_OPTIONS = {"reasoning": {"enabled": True, "effort": "high"}}
_DECISION_KEYS = {"name", "arguments", "message"}


def _error(code, status):
    return web.json_response({"error": {"message": "Native task decision was refused.",
                                       "type": "native_task_error", "code": code}}, status=status)


def _bounded_text(value, maximum):
    return isinstance(value, str) and len(value.encode("utf-8")) <= maximum


def _schema_names(schema):
    """Accept the concrete decision schema, never external refs or executable schema hooks."""
    if (not isinstance(schema, dict)
            or set(schema) != {"type", "additionalProperties", "required", "properties"}
            or schema.get("type") != "object"
            or schema.get("additionalProperties") is not False
            or not isinstance(schema.get("required"), list)
            or set(schema["required"]) != _DECISION_KEYS):
        raise ValueError("Invalid decision schema")
    properties = schema.get("properties")
    if not isinstance(properties, dict) or set(properties) != _DECISION_KEYS:
        raise ValueError("Invalid decision properties")
    name = properties["name"]
    if (not isinstance(name, dict) or set(name) != {"type", "enum"}
            or name["type"] != "string" or not isinstance(name["enum"], list)
            or not 1 <= len(name["enum"]) <= 64
            or not all(_bounded_text(value, 128) for value in name["enum"])
            or "" not in name["enum"]
            or properties["arguments"] != {"type": "object"}
            or properties["message"] != {"type": "string"}):
        raise ValueError("Invalid decision choices")
    return frozenset(name["enum"])


def _request_body(body):
    if not isinstance(body, dict) or set(body) != {"instructions", "input", "json_schema"}:
        raise ValueError("Invalid decision body")
    if len(json.dumps(body, ensure_ascii=False).encode("utf-8")) > 163840:
        raise ValueError("Decision request is too large")
    instructions, items, schema = body["instructions"], body["input"], body["json_schema"]
    if (not _bounded_text(instructions, 8192) or not isinstance(items, list)
            or len(items) != 1 or len(json.dumps(schema).encode("utf-8")) > 16384):
        raise ValueError("Invalid decision bounds")
    item = items[0]
    if (not isinstance(item, dict) or set(item) != {"type", "text"}
            or item["type"] != "text" or not _bounded_text(item["text"], 131072)):
        raise ValueError("Invalid decision input")
    return instructions, item["text"], schema, _schema_names(schema)


def _validated_result(result, names, session_id):
    if not isinstance(result, dict) or result.get("failed"):
        raise ValueError("Decision run failed")
    runtime = result.get("runtime")
    if (not isinstance(runtime, dict) or runtime.get("provider") != PROVIDER
            or runtime.get("model") != MODEL or runtime.get("model_lock") != "confirmed"
            or result.get("session_id") != session_id):
        raise ValueError("Decision runtime was not confirmed")
    output = result.get("final_response")
    if not _bounded_text(output, 16384):
        raise ValueError("Decision output is too large")
    decision = json.loads(output)
    if (not isinstance(decision, dict) or set(decision) != _DECISION_KEYS
            or not isinstance(decision["name"], str) or decision["name"] not in names
            or not isinstance(decision["arguments"], dict)
            or not _bounded_text(decision["message"], 4000)):
        raise ValueError("Decision output does not match the schema")
    json.dumps(decision, allow_nan=False)
    return decision, runtime


def _validated_agent(agent_ref):
    if not agent_ref:
        raise ValueError("Decision agent evidence is missing")
    agent = agent_ref[0]
    if (getattr(agent, "tools", None) != []
            or getattr(agent, "reasoning_config", None) != {"enabled": True, "effort": "high"}
            or getattr(agent, "_fallback_chain", None) != []):
        raise ValueError("Decision agent did not serve the restricted runtime")


def _runtime_request():
    return {"require_model_lock": True,
            "requested": {"provider": PROVIDER, "model": MODEL},
            "model_options": {"reasoning": {"enabled": True, "effort": "high"}},
            "route_source": "native_task_decision"}


async def handle(self, request):
    from gateway.platforms.api_server_runs import _native_run_provider

    try:
        # Native task decisions share the authenticated provider boundary with child work.
        # This also rejects the legacy default-listener no-key branch used by some tests.
        _native_run_provider(self, request)
    except PermissionError:
        return _error("native_task_denied", 403)
    body, err = await self._read_json_body(request)
    if err is not None:
        return err
    try:
        instructions, text, schema, names = _request_body(body)
    except (TypeError, ValueError):
        return _error("invalid_native_decision", 400)
    db = await self._ensure_session_db_async()
    if db is None:
        return _error("native_task_store_unavailable", 503)
    session_id = "native_decision_" + uuid.uuid4().hex
    await asyncio.to_thread(db.create_session, session_id, "api_server", model=MODEL)
    # One native model turn, with no tools. Schema is an output constraint in the
    # initial prompt and validated below; no unsupported API response-format is invented.
    policy = (instructions + "\nReturn exactly one JSON object matching this schema. "
              "No markdown or extra text. Do not execute actions. Keep message brief.\n"
              + json.dumps(schema, ensure_ascii=False, separators=(",", ":")))
    agent_ref = [None]
    end_reason = "native_task_decision_failed"
    try:
        result, _ = await self._run_agent(
            user_message=text, conversation_history=[], ephemeral_system_prompt=policy,
            session_id=session_id, agent_ref=agent_ref,
            requested_model=MODEL, requested_provider=PROVIDER,
            requested_runtime={"provider": PROVIDER, "model": MODEL},
            route_source="native_task_decision", model_options=MODEL_OPTIONS,
            confirmed_runtime_lock=True, enabled_toolsets_override=[], max_iterations_override=1,
            notification_category="diagnostic")
        _validated_agent(agent_ref)
        decision, runtime = _validated_result(result, names, session_id)
        persisted = await asyncio.to_thread(self._persist_session_runtime_lock, session_id, _runtime_request())
        if not persisted:
            raise ValueError("Decision runtime receipt was not persisted")
        end_reason = "native_task_decision"
    except Exception:
        return _error("native_decision_unconfirmed", 502)
    finally:
        try:
            await asyncio.to_thread(db.end_session, session_id, end_reason)
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Native decision session finalization failed: %s", type(exc).__name__)
    return web.json_response({"decision": decision, "runtime": runtime, "session_id": session_id})


def routes(self):
    return [("POST", "/v1/native-task/decision", self._handle_native_task_decision)]
