import importlib.util
import stat
import sys
import types
from pathlib import Path

import pytest


CANDIDATE = Path(__file__).parents[1] / "agent" / "tool_executor.py"
REPLAY_CANDIDATE = Path(__file__).parents[1] / "agent" / "continuity" / "replay_ledger.py"


def _load_candidate():
    spec = importlib.util.spec_from_file_location("governed_candidate_tool_executor", CANDIDATE)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.__file__ is not None
    assert Path(module.__file__).resolve() == CANDIDATE.resolve()
    return module


def _load_replay_candidate():
    spec = importlib.util.spec_from_file_location("governed_candidate_replay_ledger", REPLAY_CANDIDATE)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.__file__ is not None
    assert Path(module.__file__).resolve() == REPLAY_CANDIDATE.resolve()
    return module


def _agent():
    return types.SimpleNamespace(
        _tool_guardrails=types.SimpleNamespace(
            before_call=lambda _name, _args: types.SimpleNamespace(allows_execution=True)
        ),
        _turns_since_memory=1,
        _iters_since_skill=1,
        _current_turn_id="turn-1",
        session_id="session-1",
    )


def test_authorized_dispatch_crosses_replay_boundary_after_policy(monkeypatch):
    module = _load_candidate()
    dispatches = []
    executions = []

    class Ledger:
        def dispatch(self, call_id, name, args, leaf):
            dispatches.append((call_id, name, dict(args)))
            return leaf(args), "completed"

    monkeypatch.setattr(module, "ledger_for_agent", lambda agent, task_id: Ledger())
    monkeypatch.setattr(module, "_pre_tool_block", lambda agent, ref: (None, ref.args))
    monkeypatch.setattr(module, "_begin_tool_execution", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_run_with_activity_heartbeat", lambda agent, name, leaf: leaf())
    monkeypatch.setattr("agent.terminal_approval_batch.prepare_current_terminal", lambda ref: None)

    ref = module._ToolCallRef("terminal", {"command": "true"}, "task-1", "call-1", [])
    state = module._ManagedToolResult(
        result=None, args=ref.args, blocked=False, dispatched=False, middleware_trace=[]
    )
    result = module._dispatch_authorized_once(
        _agent(), state, ref,
        execute=lambda args: executions.append(dict(args)) or "ok",
        scope_block=None, display_index=None, begin_execution=None, authorization_gate=None,
    )

    assert result == "ok"
    assert dispatches == [("call-1", "terminal", {"command": "true"})]
    assert executions == [{"command": "true"}]


def test_blocked_dispatch_never_claims_replay_identity(monkeypatch):
    module = _load_candidate()
    monkeypatch.setattr(
        module,
        "ledger_for_agent",
        lambda *_args, **_kwargs: pytest.fail("blocked calls must not reach replay custody"),
    )
    monkeypatch.setattr(module, "_blocked_tool_result", lambda *args, **kwargs: "blocked")
    ref = module._ToolCallRef("terminal", {}, "task-1", "call-1", [])
    state = module._ManagedToolResult(
        result=None, args=ref.args, blocked=False, dispatched=False, middleware_trace=[]
    )

    assert module._dispatch_authorized_once(
        _agent(), state, ref, execute=lambda args: pytest.fail("blocked call executed"),
        scope_block="no", display_index=None, begin_execution=None, authorization_gate=None,
    ) == "blocked"


def test_replay_ledger_rejects_symlink_parent_and_preserves_modes(tmp_path):
    replay = _load_replay_candidate()
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    link = tmp_path / "link"
    link.symlink_to(private, target_is_directory=True)
    with pytest.raises(RuntimeError, match="not a directory"):
        replay.ReplayLedger(link / "ledger.jsonl", session_id="s", turn_id="t")

    ledger = replay.ReplayLedger(private / "ledger.jsonl", session_id="s", turn_id="t")
    assert ledger.dispatch("c", "tool", {"value": 1}, lambda args: args)[1] == "completed"
    assert stat.S_IMODE((private / "ledger.jsonl").stat().st_mode) == 0o600
    assert stat.S_IMODE((private / "ledger.jsonl.lock").stat().st_mode) == 0o600