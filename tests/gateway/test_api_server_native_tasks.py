"""Native linked-child admission exercises real run storage and scheduling contracts."""
import asyncio
from contextvars import ContextVar
import copy
import json
from types import SimpleNamespace

import pytest

from gateway.platforms import api_server_runs as runs
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore


class Request:
    def __init__(self, body, key="one", token="test-native-key-0123456789"):
        self.body = body
        self.headers = {"Idempotency-Key": key, "Authorization": "Bearer " + token}
        self.match_info = {}
        self.query = {}

    async def json(self):
        return copy.deepcopy(self.body)


class Provider:
    def __init__(self):
        self.calls = []
        self.allowed = True
        self.wrong_route = False
        self.refuse_acceptance = False
        self.prior_accepted = False

    def invoke(self, operation, context, payload):
        self.calls.append((operation, context, payload))
        if operation == "child.prepare":
            body = payload["body"]
            workload = body["child"].get("workload", "normal")
            return {"ok": True, "plan": {
                "reservation_id": "reservation", "parent_session_id": body["session_id"],
                "child_session_id": "child", "goal": body["child"]["goal"],
                "context": body["child"]["context"], "workload": workload,
                "provider": "openai-codex", "model": "other" if self.wrong_route else
                    {"normal": "gpt-6.1-terra", "difficult": "gpt-6.1-sol"}[workload],
                "reasoning": {"enabled": True, "effort": "high"},
                **({"state": "accepted", "run_id": "old_run"} if self.prior_accepted else {}),
            }, "replayed": False}
        if operation == "child.accepted":
            if self.refuse_acceptance:
                raise RuntimeError("receipt transport unavailable")
            return {"ok": True, "status": "accepted", "run_id": payload["run_id"]}
        return {"ok": True, "authorized": self.allowed, "run_id": payload["run_id"]}


class Owner:
    def __init__(self, tmp_path):
        self._run_idempotency_store = RunIdempotencyStore(str(tmp_path / "runs.db"))
        self._run_statuses = {}
        self._run_idempotency_ids = set()
        self._run_owners = {}
        self._run_streams = {}
        self._run_streams_created = {}
        self._run_approval_sessions = {}
        self._active_run_tasks = {}
        self._background_tasks = set()
        self._run_owner_pid = 0
        self._run_owner_started = 0

    def _expected_api_key(self):
        return "test-native-key-0123456789"

    def _check_auth(self, request):
        return None

    def _run_idempotency_scope(self, request):
        return "test-authenticated-profile"

    def _concurrency_limited_response(self):
        return None

    def _durable_run_status(self, request, run_id):
        return runs._durable_run_status(self, request, run_id)

    def _set_run_status(self, run_id, status, **fields):
        return runs._set_run_status(self, run_id, status, **fields)

    def _activate_admitted_request(self):
        pass

    def _parse_session_key_header(self, request):
        return None, None

    async def _normalize_room_dispatch(self, request, body):
        return body, None


@pytest.fixture
def environment(tmp_path, monkeypatch):
    import gateway.native_task_host as host
    provider = Provider()
    context = {"principal_id": "api-server", "principal_kind": "native_bearer",
               "profile_home": str(tmp_path), "profile_name": "default",
               "store_id": "test-store", "host_epoch": "test-epoch"}
    monkeypatch.setattr(host, "native_task_host_context", lambda: context)
    monkeypatch.setattr(host, "get_native_task_host_provider", lambda home: provider)
    api = SimpleNamespace(
        _api_request_profile=ContextVar("test_profile", default="default"),
        _api_request_browser_control_principal=ContextVar("test_principal", default=None),
        _api_request_browser_control_transport_family=ContextVar("test_family", default=None),
        _openai_error=lambda message, **kw: {"error": {"message": message, **kw}},
    )
    launches = []
    async def execute(owner, launch, **kw):
        launches.append(launch)
    monkeypatch.setattr(runs, "_execute_run", execute)
    owner = Owner(tmp_path)
    yield owner, provider, api, launches
    owner._run_idempotency_store.close()


def body(workload="normal"):
    return {"input": "Original exact user words", "session_id": "parent",
            "origin": {"event_id": "event", "origin_turn_id": "origin"},
            "child": {"goal": "Inspect server", "context": "Relevant data", "workload": workload}}


@pytest.mark.parametrize("workload,model", [("normal", "gpt-6.1-terra"), ("difficult", "gpt-6.1-sol")])
def test_child_runs_use_separate_session_locked_route_and_acceptance_before_execution(environment, workload, model):
    owner, provider, api, launches = environment
    async def run():
        response = await runs._handle_runs(owner, Request(body(workload)), _api_server=api)
        assert response.status == 202
        accepted = json.loads(response.body)
        assert provider.calls[-1][0] == "child.accepted"
        await asyncio.gather(*owner._active_run_tasks.values())
        assert len(launches) == 1
        launch = launches[0]
        assert launch.session_id == "child" and launch.user_message == "Inspect server"
        assert "Original exact user words" not in launch.user_message
        assert launch.agent_kwargs["requested_model"] == model
        assert launch.agent_kwargs["requested_provider"] == "openai-codex"
        assert launch.agent_kwargs["confirmed_runtime_lock"] is True
        assert launch.agent_kwargs["model_options"] == {"reasoning": {"enabled": True, "effort": "high"}}
        again = await runs._handle_runs(owner, Request(body(workload)), _api_server=api)
        assert json.loads(again.body)["run_id"] == accepted["run_id"]
        assert json.loads(again.body)["replayed"] is True
        assert len(launches) == 1
        changed = body(workload)
        changed["child"]["goal"] = "Different work"
        conflict = await runs._handle_runs(owner, Request(changed), _api_server=api)
        assert conflict.status == 409 and len(launches) == 1
    asyncio.run(run())


def test_child_acceptance_loss_and_invalid_route_never_execute(environment):
    owner, provider, api, launches = environment
    async def run():
        provider.wrong_route = True
        assert (await runs._handle_runs(owner, Request(body()), _api_server=api)).status == 409
        provider.wrong_route = False
        provider.refuse_acceptance = True
        assert (await runs._handle_runs(owner, Request(body()), _api_server=api)).status == 409
        replay = await runs._handle_runs(owner, Request(body()), _api_server=api)
        assert replay.status == 202 and json.loads(replay.body)["status"] == "failed"
        assert not launches and not owner._active_run_tasks
    asyncio.run(run())


def test_linked_control_requires_native_provider_owner_and_strong_bearer(environment):
    owner, provider, api, launches = environment
    async def run():
        denied = await runs._handle_runs(owner, Request(body(), token="wrong"), _api_server=api)
        assert denied.status == 403 and not provider.calls
        response = await runs._handle_runs(owner, Request(body()), _api_server=api)
        run_id = json.loads(response.body)["run_id"]
        await asyncio.gather(*owner._active_run_tasks.values())
        request = Request(body())
        assert runs._request_owns_run(owner, request, run_id)
        provider.allowed = False
        assert not runs._request_owns_run(owner, request, run_id)
        assert len(launches) == 1
    asyncio.run(run())


def test_confirmed_runtime_marker_requires_actual_served_pair():
    run = SimpleNamespace(agent_kwargs={"confirmed_runtime_lock": True,
        "requested_provider": "openai-codex", "requested_model": "gpt-6.1-terra"})
    assert runs._confirmed_run_runtime(run, {"provider": "openai-codex", "model": "gpt-6.1-terra"}) == "confirmed"
    with pytest.raises(RuntimeError, match="runtime was not served"):
        runs._confirmed_run_runtime(run, {"provider": "openai", "model": "gpt-6.1-terra"})


def test_accepted_provider_reservation_without_native_receipt_does_not_relaunch(environment):
    owner, provider, api, launches = environment
    provider.prior_accepted = True
    response = asyncio.run(runs._handle_runs(owner, Request(body()), _api_server=api))
    assert response.status == 409
    assert not launches and not owner._active_run_tasks and not owner._run_statuses


def test_concurrent_prepare_replays_a_matching_accepted_native_receipt(environment):
    import threading
    owner, provider, api, launches = environment
    entered, release = threading.Event(), threading.Event()
    lock = threading.Lock()
    original = provider.invoke
    calls, admitted = 0, {}
    def invoke(operation, context, payload):
        nonlocal calls
        if operation == "child.prepare":
            with lock:
                calls += 1
                first = calls == 1
            if first:
                entered.set()
                assert release.wait(5)
                result = original(operation, context, payload)
                result["plan"].update(state="accepted", run_id=admitted["run_id"])
                return result
        return original(operation, context, payload)
    provider.invoke = invoke
    async def run():
        first = asyncio.create_task(runs._handle_runs(owner, Request(body()), _api_server=api))
        try:
            assert await asyncio.to_thread(entered.wait, 3)
            second = await runs._handle_runs(owner, Request(body()), _api_server=api)
            assert second.status == 202
            admitted.update(json.loads(second.body))
        finally:
            release.set()
        replay = await asyncio.wait_for(first, 3)
        assert replay.status == 202
        assert json.loads(replay.body)["run_id"] == admitted["run_id"]
        assert json.loads(replay.body)["replayed"] is True
        await asyncio.gather(*owner._active_run_tasks.values())
        assert len(launches) == 1
    asyncio.run(run())


@pytest.mark.parametrize("payload,expected,error", [
    ({"input": "Original words"}, ("Original words", "Original words"), None),
    ({"input": [{"role": "user", "content": "Original words"}]},
     ([{"role": "user", "content": "Original words"}], "Original words"), None),
    ({}, None, "Missing 'input' field"),
    ({"input": [{"role": "user", "content": ""}]}, None, "No user message found in input"),
])
def test_ordinary_run_input_preserves_native_text_and_missing_input_contract(payload, expected, error):
    assert runs._run_input_parts(payload) == (expected, error)
