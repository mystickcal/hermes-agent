from __future__ import annotations

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    value = SessionDB(tmp_path / "state.db")
    value.create_session("voice-session", "test")
    return value


def test_passive_receipt_is_atomic_and_idempotent(db):
    first = db.append_messages_with_receipt(
        "voice-session", [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
        producer="passive.ingress.v1", event_id="event-1", origin_turn_id="turn-1",
    )
    replay = db.append_messages_with_receipt(
        "voice-session", [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
        producer="passive.ingress.v1", event_id="event-1", origin_turn_id="turn-1",
    )
    assert first["message_ids"] and first["revision"] == first["message_ids"][-1]
    assert not first["replayed"] and replay["replayed"]
    assert replay["message_ids"] == first["message_ids"]
    assert [message["content"] for message in db.get_messages("voice-session")] == ["hello", "hi"]
    assert db.get_message_receipt("voice-session", producer="passive.ingress.v1", event_id="event-1") == replay


def test_passive_receipt_rejects_changed_replay_without_new_messages(db):
    db.append_messages_with_receipt(
        "voice-session", [{"role": "user", "content": "hello"}],
        producer="passive.ingress.v1", event_id="event-1", origin_turn_id="turn-1",
    )
    with pytest.raises(ValueError, match="conflicts"):
        db.append_messages_with_receipt(
            "voice-session", [{"role": "user", "content": "changed"}],
            producer="passive.ingress.v1", event_id="event-1", origin_turn_id="turn-1",
        )
    assert [message["content"] for message in db.get_messages("voice-session")] == ["hello"]


def test_passive_receipt_rolls_back_messages_when_receipt_transaction_fails(db, monkeypatch):
    original = db._bump_session_counters
    def fail(*args, **kwargs):
        raise RuntimeError("force transaction rollback")
    monkeypatch.setattr(db, "_bump_session_counters", fail)
    with pytest.raises(RuntimeError, match="rollback"):
        db.append_messages_with_receipt(
            "voice-session", [{"role": "user", "content": "hello"}],
            producer="passive.ingress.v1", event_id="event-rollback", origin_turn_id="turn-1",
        )
    monkeypatch.setattr(db, "_bump_session_counters", original)
    assert db.get_messages("voice-session") == []
    assert db.get_message_receipt(
        "voice-session", producer="passive.ingress.v1", event_id="event-rollback"
    ) is None


def test_passive_receipt_refuses_untrusted_row_shape_and_missing_session(db):
    with pytest.raises(ValueError, match="only role and content"):
        db.append_messages_with_receipt(
            "voice-session", [{"role": "user", "content": "hello", "tool_calls": []}],
            producer="passive.ingress.v1", event_id="event-1", origin_turn_id="turn-1",
        )
    with pytest.raises(ValueError, match="does not exist"):
        db.append_messages_with_receipt(
            "missing", [{"role": "user", "content": "hello"}],
            producer="passive.ingress.v1", event_id="event-1", origin_turn_id="turn-1",
        )
