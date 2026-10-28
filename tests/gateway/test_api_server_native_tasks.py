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


# Ordinary locked requests share this module with Talk's native child producer.
# Exercise the real HTTP parsers, SessionDB, native idempotency and worker seam.
@pytest.fixture
def ordinary_locked_runtime(monkeypatch):
    from unittest.mock import AsyncMock, Mock
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter

    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={
        "key": "ordinary-locked-test-key-0123456789"}))
    db = adapter._ensure_session_db()
    assert db is not None and adapter._run_idempotency_store.durable
    state = SimpleNamespace(launches=[], mailbox_launches=[],
        started=asyncio.Event(), release=asyncio.Event(), last_request=None)
    state.release.set()
    state.resolve = Mock(return_value={"provider": "openai-codex"})
    monkeypatch.setattr(adapter, "_resolve_provider_runtime", state.resolve)
    state.mailbox = AsyncMock(return_value=None)
    monkeypatch.setattr(adapter, "_admit_to_live_bot_chat", state.mailbox)
    state.activate = Mock(wraps=adapter._activate_admitted_request)
    monkeypatch.setattr(adapter, "_activate_admitted_request", state.activate)

    async def execute(owner, launch, **kwargs):
        state.launches.append(launch)
        state.started.set()
        await state.release.wait()
        owner._set_run_status(launch.run_id, "completed", output="test transport complete")
        runs._retire_live_run(owner, launch.run_id)

    async def mailbox_execute(owner, launch, *delivery, **kwargs):
        state.mailbox_launches.append((launch, delivery))
        owner._set_run_status(launch.run_id, "completed", output="test mailbox complete")
        runs._retire_live_run(owner, launch.run_id)

    monkeypatch.setattr(runs, "_execute_run", execute)
    monkeypatch.setattr(runs, "_execute_run_via_live_owner", mailbox_execute)
    yield adapter, db, state
    adapter._run_idempotency_store.close()
    db.close()


def _ordinary_locked_body(session_id=None):
    value = {"input": "Inspect the requested native task", "provider": "openai-codex",
             "model": "gpt-6.1-terra", "require_model_lock": True,
             "model_options": {"reasoning": {"enabled": True, "effort": "high"}}}
    if session_id is not None:
        value["session_id"] = session_id
    return value


def _ordinary_runs_app(adapter, state):
    from aiohttp import web

    @web.middleware
    async def observe(request, handler):
        state.last_request = request
        return await handler(request)

    app = web.Application(middlewares=[observe])
    app.router.add_post("/v1/runs", adapter._handle_runs)
    return app


def _ordinary_headers(key):
    return {"Authorization": "Bearer ordinary-locked-test-key-0123456789", "Idempotency-Key": key}


@pytest.mark.asyncio
@pytest.mark.parametrize("fault,expected", [
    ("route_provider", 400), ("route_key", 400), ("route_url", 400),
    ("provider_unavailable", 409), ("provider_mismatch", 409),
])
async def test_ordinary_locked_refusal_precedes_native_reservation(
    ordinary_locked_runtime, fault, expected,
):
    from aiohttp.test_utils import TestClient, TestServer
    adapter, _, state = ordinary_locked_runtime
    payload = _ordinary_locked_body()
    if fault.startswith("route_"):
        payload["model"] = "locked-alias"
        route = {"model": "gpt-6.1-terra", "provider": "openai-codex"}
        if fault == "route_provider":
            route["provider"] = "openai"
        elif fault == "route_key":
            route["api_key"] = "dummy-route-credential"
        else:
            route["base_url"] = "https://invalid.example/v1"
        adapter._model_routes["locked-alias"] = route
    elif fault == "provider_unavailable":
        state.resolve.side_effect = RuntimeError("fixture provider unavailable")
    else:
        state.resolve.return_value = {"provider": "openai"}
    async with TestClient(TestServer(_ordinary_runs_app(adapter, state))) as client:
        response = await client.post("/v1/runs", json=payload, headers=_ordinary_headers("refused"))
        assert response.status == expected
        await response.json()
    scope = adapter._run_idempotency_scope(state.last_request)
    assert adapter._run_idempotency_store.lookup(scope, "refused", "unused") == ("missing", None)
    assert not state.launches and not state.mailbox_launches
    assert not adapter._run_owners and not adapter._run_statuses and not adapter._run_streams
    assert not adapter._run_approval_sessions and not adapter._active_run_tasks
    assert state.activate.call_count == 0 and adapter.active_agent_work_count() == 0
    assert state.mailbox.await_count == 0
    assert state.resolve.call_count == (0 if fault.startswith("route_") else 1)


@pytest.mark.asyncio
async def test_ordinary_locked_accepted_replay_bypasses_failed_provider_and_full_capacity(
    ordinary_locked_runtime,
):
    from aiohttp.test_utils import TestClient, TestServer
    adapter, _, state = ordinary_locked_runtime
    adapter._max_concurrent_runs = 1
    state.release.clear()
    payload = _ordinary_locked_body()
    async with TestClient(TestServer(_ordinary_runs_app(adapter, state))) as client:
        try:
            first = await client.post("/v1/runs", json=payload, headers=_ordinary_headers("accepted"))
            assert first.status == 202
            original = await first.json()
            await asyncio.wait_for(state.started.wait(), 3)
            assert adapter._concurrency_limited_response().status == 429
            state.resolve.side_effect = RuntimeError("fixture provider is now unavailable")
            retry = await client.post("/v1/runs", json=payload, headers=_ordinary_headers("accepted"))
            assert retry.status == 202
            replay = await retry.json()
            assert replay["run_id"] == original["run_id"] and replay["replayed"] is True
            changed = {**payload, "input": "Different request"}
            conflict = await client.post("/v1/runs", json=changed, headers=_ordinary_headers("accepted"))
            assert conflict.status == 409
            assert len(state.launches) == 1 and state.resolve.call_count == 1
        finally:
            state.release.set()
            await asyncio.gather(*list(adapter._active_run_tasks.values()))


@pytest.mark.asyncio
async def test_ordinary_locked_canonical_compression_lineage_refuses_without_mailbox_delivery(
    ordinary_locked_runtime,
):
    from aiohttp.test_utils import TestClient, TestServer
    from hermes_state import SessionDB
    from hermes_constants import get_hermes_home
    adapter, db, state = ordinary_locked_runtime
    db.create_session("canonical", "hermes_browser")
    db.set_session_title("canonical", SessionDB.CANONICAL_BOT_CHAT_TITLE)
    db.end_session("canonical", "compression")
    db.create_session("canonical-tip", "hermes_browser", parent_session_id="canonical")
    assert {"canonical", "canonical-tip"}.issubset(set(db.get_compression_lineage("canonical-tip")))
    payload = _ordinary_locked_body("canonical-tip")
    async with TestClient(TestServer(_ordinary_runs_app(adapter, state))) as client:
        locked = await client.post("/v1/runs", json=payload, headers=_ordinary_headers("locked-owner"))
        assert locked.status == 409
        await locked.json()
        scope = adapter._run_idempotency_scope(state.last_request)
        assert adapter._run_idempotency_store.lookup(scope, "locked-owner", "unused") == ("missing", None)
        assert not adapter._run_owners and not adapter._run_statuses and not adapter._run_streams
        assert not state.launches and state.mailbox.await_count == 0
        assert state.activate.call_count == 0 and adapter.active_agent_work_count() == 0
        # An unlocked request still follows the original, owner-controlled mailbox route.
        state.mailbox.return_value = (get_hermes_home(), {"delivery_id": "fixture-delivery", "status": "queued"})
        unlocked = {**payload, "require_model_lock": False}
        accepted = await client.post("/v1/runs", json=unlocked, headers=_ordinary_headers("unlocked-owner"))
        assert accepted.status == 202
        await accepted.json()
        await asyncio.gather(*list(adapter._active_run_tasks.values()))
        assert state.mailbox.await_count == 1 and len(state.mailbox_launches) == 1
        assert not state.launches


class _ServedFixtureAgent:
    def __init__(self, session_id, mismatch):
        self.session_id = session_id
        self.provider = "openai" if mismatch == "provider" else "openai-codex"
        self.model = "other-model" if mismatch == "model" else "gpt-6.1-terra"
        self._hermes_api_runtime = {"provider": "openai-codex", "model": "gpt-6.1-terra"}
        self.memory_manager = None

    def run_conversation(self, **kwargs):
        return {"final_response": "fixture model completion"}


@pytest.mark.asyncio
@pytest.mark.parametrize("producer", ["ordinary", "native_child"])
@pytest.mark.parametrize("mismatch", [None, "provider", "model"])
async def test_ordinary_and_native_terminal_receipts_validate_the_actual_sync_worker(
    ordinary_locked_runtime, producer, mismatch,
):
    from aiohttp.test_utils import TestClient, TestServer
    from gateway.platforms import api_server as api_module
    import time
    adapter, db, state = ordinary_locked_runtime
    db.create_session("receipt-parent", "api_server")
    db.create_session("receipt-child", "api_server", parent_session_id="receipt-parent")
    if producer == "ordinary":
        async with TestClient(TestServer(_ordinary_runs_app(adapter, state))) as client:
            response = await client.post("/v1/runs", json=_ordinary_locked_body("receipt-child"),
                                         headers=_ordinary_headers("receipt"))
            assert response.status == 202
            await response.json()
            await asyncio.gather(*list(adapter._active_run_tasks.values()))
    else:
        plan = {"child_session_id": "receipt-child", "parent_session_id": "receipt-parent",
                "model": "gpt-6.1-terra", "goal": "Run the derived task", "context": "Fixture data"}
        adapter._set_run_status("run-native-receipt", "queued", session_id="receipt-child")
        runs._schedule_linked_child(adapter, None, plan, "run-native-receipt", time.time(), _api_server=api_module)
        await asyncio.gather(*list(adapter._active_run_tasks.values()))
    assert len(state.launches) == 1
    launch = state.launches[0]
    assert launch.confirmed_runtime_lock is (producer == "ordinary")
    assert launch.agent_kwargs["confirmed_runtime_lock"] is True
    assert launch.agent_kwargs["model_options"] == {"reasoning": {"enabled": True, "effort": "high"}}
    agent = _ServedFixtureAgent(launch.session_id, mismatch)

    async def finish():
        _, _, served = await asyncio.to_thread(
            runs._run_agent_sync, adapter, launch, agent, lambda data: None, _api_server=api_module)
        return await runs._final_run_runtime(adapter, launch, served)

    if mismatch:
        with pytest.raises(RuntimeError, match="runtime.*(mismatch|not served)"):
            await finish()
        assert "browser_model_lock" not in json.loads(db.get_session(launch.session_id)["model_config"] or "{}")
    else:
        runtime = await finish()
        assert runtime["provider"] == "openai-codex" and runtime["model"] == "gpt-6.1-terra"
        assert runtime["model_lock"] == "confirmed"
        if producer == "ordinary":
            assert runtime["requested"] == {"provider": "openai-codex", "model": "gpt-6.1-terra"}
        else:
            lock = json.loads(db.get_session(launch.session_id)["model_config"])["browser_model_lock"]
            assert lock["confirmed"] is True and lock["model_options"]["reasoning"]["effort"] == "high"
