"""Decision endpoint uses native auth, runtime receipts, and a real isolated SessionDB."""
import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from gateway.platforms.api_server import APIServerAdapter, _request_reasoning_config
from gateway.platforms import api_server_native_decisions as decisions
from gateway.platforms import api_server_runs as runs
from gateway.native_task_host import NativeTaskHostProvider, register_native_task_host_provider, restore_native_task_host_provider
from hermes_constants import get_hermes_home
from hermes_state import SessionDB

TOKEN = "native-decision-test-key-0123456789"


class Request:
    def __init__(self, body, token=TOKEN):
        self.body = copy.deepcopy(body)
        self.headers = {"Authorization": "Bearer " + token}
        self.remote = "127.0.0.1"
        self.method = "POST"
        self.path = "/v1/native-task/decision"
        self.path_qs = self.path
        self.transport = None
        self.query = {}
        self.match_info = {}
        self.app = {}

    async def json(self):
        return self.body


class Provider(NativeTaskHostProvider):
    def invoke(self, operation, context, payload):
        raise AssertionError("Decision classification must not dispatch a provider action")


def request_body(text="Captured operator request"):
    return {"instructions": "Route the captured operator request.",
            "input": [{"type": "text", "text": text}],
            "json_schema": {"type": "object", "additionalProperties": False,
                "required": ["name", "arguments", "message"],
                "properties": {"name": {"type": "string", "enum": ["", "delegate_task"]},
                               "arguments": {"type": "object"}, "message": {"type": "string"}}}}


@pytest.fixture
def environment(monkeypatch):
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    db = SessionDB(db_path=home / "state.db")
    owner = object.__new__(APIServerAdapter)
    owner._api_key = TOKEN
    owner._ensure_session_db = lambda: db
    async def ensure():
        return db
    owner._ensure_session_db_async = ensure
    owner.calls = []
    owner.problem = ""
    owner.output = {"name": "delegate_task", "arguments": {"workload": "normal"}, "message": "Working."}
    async def run_agent(**kwargs):
        owner.calls.append(kwargs)
        assert kwargs["requested_provider"] == "openai-codex"
        assert kwargs["requested_model"] == "gpt-6.1-terra"
        assert kwargs["enabled_toolsets_override"] == []
        assert kwargs["max_iterations_override"] == 1
        assert kwargs["confirmed_runtime_lock"] is True
        assert _request_reasoning_config(kwargs["model_options"]) == {"enabled": True, "effort": "high"}
        agent = SimpleNamespace(
            provider="openai-codex", model="gpt-6.1-terra", session_id=kwargs["session_id"],
            reasoning_config={"enabled": True, "effort": "high"}, tools=[], _fallback_chain=[],
            _hermes_api_runtime={"provider": "openai-codex", "model": "gpt-6.1-terra"})
        changes = {
            "model": ("model", "other"),
            "provider": ("provider", "openai"),
            "reasoning": ("reasoning_config", {"enabled": True, "effort": "low"}),
            "tools": ("tools", [{"function": {"name": "terminal"}}]),
            "fallback": ("_fallback_chain", [{"provider": "openai", "model": "other"}]),
        }
        if owner.problem in changes:
            attribute, value = changes[owner.problem]
            setattr(agent, attribute, value)
        kwargs["agent_ref"][0] = agent
        output = owner.output if isinstance(owner.output, str) else json.dumps(owner.output)
        return owner._finish_turn_result(
            agent, {"final_response": output}, kwargs["session_id"], route=None,
            requested_runtime=kwargs["requested_runtime"], route_source=kwargs["route_source"],
            confirmed_runtime_lock=True)
    owner._run_agent = run_agent
    register_native_task_host_provider(Provider(), profile_home=home)
    yield owner, db
    restore_native_task_host_provider(None, profile_home=home)
    db.close()


def test_native_decision_accepts_real_catalog_size_and_persists_confirmed_high_lock(environment):
    owner, db = environment
    response = asyncio.run(decisions.handle(owner, Request(request_body("x" * 35000))))
    assert response.status == 200
    result = json.loads(response.body)
    assert result["decision"] == owner.output
    assert result["runtime"]["provider"] == "openai-codex"
    assert result["runtime"]["model"] == "gpt-6.1-terra"
    assert result["runtime"]["model_lock"] == "confirmed"
    session = db.get_session(result["session_id"])
    lock = json.loads(session["model_config"])["browser_model_lock"]
    assert lock["confirmed"] is True
    assert lock["model_options"] == {"reasoning": {"enabled": True, "effort": "high"}}
    assert session["end_reason"] == "native_task_decision"
    assert len(owner.calls) == 1


@pytest.mark.parametrize("problem", ["model", "provider", "reasoning", "tools", "fallback"])
def test_native_decision_denies_unserved_or_less_restricted_agent(environment, problem):
    owner, db = environment
    owner.problem = problem
    response = asyncio.run(decisions.handle(owner, Request(request_body())))
    assert response.status == 502 and len(owner.calls) == 1
    sid = owner.calls[0]["session_id"]
    config = json.loads(db.get_session(sid)["model_config"] or "{}")
    assert "browser_model_lock" not in config
    assert db.get_session(sid)["end_reason"] == "native_task_decision_failed"


@pytest.mark.parametrize("output", ["```json\n{}\n```", {"name": "forged_tool", "arguments": {}, "message": ""},
    {"name": "", "arguments": [], "message": ""}, {"name": "", "arguments": {}, "message": "", "extra": True}])
def test_native_decision_validates_actual_json_against_the_bound_tool_schema(environment, output):
    owner, db = environment
    owner.output = output
    assert asyncio.run(decisions.handle(owner, Request(request_body()))).status == 502
    assert len(owner.calls) == 1
    assert db.get_session(owner.calls[0]["session_id"])["end_reason"] == "native_task_decision_failed"


def test_native_decision_rejects_auth_and_body_overrides_before_model_work(environment):
    owner, _ = environment
    assert asyncio.run(decisions.handle(owner, Request(request_body(), token="wrong"))).status == 403
    owner._api_key = ""
    assert asyncio.run(decisions.handle(owner, Request(request_body()))).status == 403
    owner._api_key = TOKEN
    body = request_body()
    body["provider"] = "openai"
    assert asyncio.run(decisions.handle(owner, Request(body))).status == 400
    body = request_body()
    body["json_schema"]["$ref"] = "https://invalid.example/schema"
    assert asyncio.run(decisions.handle(owner, Request(body))).status == 400
    assert asyncio.run(decisions.handle(owner, Request(request_body("x" * 140000)))).status == 400
    assert not owner.calls


def test_linked_child_receipt_uses_native_session_runtime_lock(environment):
    owner, db = environment
    db.create_session("child", "api_server", model="gpt-6.1-terra")
    run = SimpleNamespace(session_id="child", agent_kwargs={
        "requested_provider": "openai-codex", "requested_model": "gpt-6.1-terra",
        "model_options": {"reasoning": {"enabled": True, "effort": "high"}}})
    runs._persist_confirmed_run_runtime(owner, run)
    lock = json.loads(db.get_session("child")["model_config"])["browser_model_lock"]
    assert lock["confirmed"] is True and lock["provider"] == "openai-codex"
    assert lock["model_options"]["reasoning"]["effort"] == "high"


def test_provider_fault_ends_only_the_new_decision_session(environment):
    owner, db = environment
    db.create_session("unrelated", "api_server")
    async def fail(**kwargs):
        owner.calls.append(kwargs)
        raise RuntimeError("provider failed")
    owner._run_agent = fail
    assert asyncio.run(decisions.handle(owner, Request(request_body()))).status == 502
    sid = owner.calls[0]["session_id"]
    assert db.get_session(sid)["end_reason"] == "native_task_decision_failed"
    assert not db.get_session("unrelated")["end_reason"]


def test_untrusted_exception_names_values_and_frames_are_not_retained(environment, caplog):
    import logging
    owner, _ = environment
    marker = "private_credential_marker_1234"
    exception_type = type(marker, (Exception,), {})
    async def fail(**kwargs):
        owner.calls.append(kwargs)
        raise exception_type(marker + "_value") from ValueError(marker + "_cause")
    owner._run_agent = fail
    caplog.set_level(logging.ERROR)
    assert asyncio.run(decisions.handle(owner, Request(request_body()))).status == 502
    assert marker not in caplog.text
    records = [record for record in caplog.records if record.name == decisions.__name__]
    assert records and all(record.exc_info[2] is None for record in records)
    assert all(type(record.exc_info[1]) is RuntimeError for record in records)
