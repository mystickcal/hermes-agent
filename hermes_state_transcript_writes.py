"""Shared, transaction-safe transcript write primitives for SessionDB."""
from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional

from hermes_state_common import _COMPRESSION_LOCK_ROW_SQL, _ENDED_ROW_SQL, _ended_by_compression

_DELETE_COMPRESSION_LOCK_SQL = "DELETE FROM compression_locks WHERE session_id = ? AND holder = ?"


class SessionTranscriptWritesMixin:
    def _stale_holder(row, now: float) -> bool:

        """A lock/lease row whose holder is expired or a provably dead local process."""

        from hermes_state import _compression_lock_holder_process_is_dead

        return float(row["expires_at"]) <= now or _compression_lock_holder_process_is_dead(row["holder"])



    def _check_transcript_write_guards(self, conn, session_id: str, compression_lock_holder: Optional[str],

            turn_lease_holder: Optional[str] = None, turn_lease_ttl_seconds: float = 300.0,

            reject_active_turn_lease: bool = False, reject_active_compression_lock: bool = False,

            allow_closed_compression_parent: bool = False) -> None:

            """Transcript-write admission checks, run INSIDE the write txn by every writer. Ordinary appends do

            NOT check compression_locks: the lock only stops two COMPRESSIONS colliding and archive_and_compact()

            commits against a watermark, so concurrent appends are safe (blocking them killed turns during slow

            summaries). Destructive user mutations opt in via ``reject_active_*`` so a compressor that captured

            its watermark cannot resurrect the removed turn.



            Shared by :meth:`append_message` and :meth:`append_messages_batch` so the two writers can never

            diverge on these correctness invariants (this guard has already needed targeted fixes — see the

            #74478 patience note below). User-initiated transcript mutations may opt in to rejecting an active

            unowned turn lease in that same transaction.

            """

            from hermes_state import SessionCompressionInProgressError

            from hermes_state_errors import CompressionSessionClosedError, SessionTurnLeaseLostError

            # NOTE (#75316 redesign): appends do NOT check compression_locks. The lock's job is to stop two

            # COMPRESSIONS colliding, not to fence ordinary transcript writes. Concurrent appends during a

            # compression are safe by construction: archive_and_compact() commits against a watermark captured

            # at compression start and clones every row that arrived after it back into the live transcript, in

            # the same write transaction. Blocking appends here was the root cause of a whole symptom family —

            # turns dying as session_persistence_failed while a slow provider summary held the lease (#74568,

            # #77386), including stale locks from dead PIDs blocking writes for the full TTL. Keep that narrow

            # fence opt-in so ordinary appends retain the watermark behavior.

            if reject_active_compression_lock:

                active_lock = conn.execute(_COMPRESSION_LOCK_ROW_SQL, (session_id,)).fetchone()

                if active_lock is not None:

                    if _stale_holder(active_lock, time.time()):

                        conn.execute(_DELETE_COMPRESSION_LOCK_SQL, (session_id, active_lock["holder"]))

                    elif active_lock["holder"] != compression_lock_holder:

                        raise SessionCompressionInProgressError(

                            f"Session {session_id!r} is being compressed by another writer")

            if turn_lease_holder or reject_active_turn_lease:

                conversation_id = self._session_turn_lease_key_on_conn(conn, session_id)

                lease = conn.execute(_TURN_LEASE_ROW_SQL, (conversation_id,)).fetchone()

                now = time.time()

                if turn_lease_holder:

                    if lease is None or lease["holder"] != turn_lease_holder:

                        raise SessionTurnLeaseLostError(

                            f"Session turn lease lost; refusing transcript write for {session_id!r}")

                    if float(lease["expires_at"]) <= now:

                        # Expiry makes the row reclaimable, not taken over; BEGIN IMMEDIATE serializes this

                        # renewal with acquisition, so a still-matching owner recovers from a starved refresher.

                        conn.execute("UPDATE session_turn_leases SET expires_at = ? "

                            "WHERE conversation_id = ? AND holder = ?",

                            (now + max(0.1, float(turn_lease_ttl_seconds)), conversation_id, turn_lease_holder))

                elif lease is not None:

                    if not _stale_holder(lease, now):

                        raise SessionTurnLeaseLostError(

                            f"Session has an active turn lease; refusing transcript mutation for {session_id!r}")

                    # Same reclaim rule as acquisition; deleting also fences a stale late flush after the mutation.

                    conn.execute("DELETE FROM session_turn_leases WHERE conversation_id = ? AND holder = ?",

                        (conversation_id, lease["holder"]))

            if _ended_by_compression(conn.execute(_ENDED_ROW_SQL, (session_id,)).fetchone()) and not allow_closed_compression_parent:

                raise CompressionSessionClosedError(session_id)



    @staticmethod
    def _bump_session_counters(conn, session_id: str, inserted: int, tool_calls: int, *, unit: bool) -> None:

            """Bump sessions.* counters after an insert; *unit* bakes the ``+ 1`` literal into the SQL."""

            inc, params = ("1", ()) if unit else ("?", (inserted,))

            if tool_calls > 0:

                conn.execute(

                    f"""UPDATE sessions SET message_count = message_count + {inc},

                           tool_call_count = tool_call_count + ? WHERE id = ?""",

                    (*params, tool_calls, session_id))

            elif inserted > 0:

                conn.execute(

                    f"UPDATE sessions SET message_count = message_count + {inc} WHERE id = ?", (*params, session_id))



    def append_messages_with_receipt(

            self, session_id: str, messages: List[Dict[str, Any]], *, producer: str,

            event_id: str, origin_turn_id: str,

        ) -> Dict[str, Any]:

            """Atomically append a small passive transcript and durable idempotency receipt.



            This deliberately accepts only plain user/assistant text.  Callers cannot supply

            arbitrary row fields, and the receipt identity is bound to the canonical bytes of

            those messages and their declared origin.  Replays return the original receipt;

            changed content for an existing event is rejected before any row can be inserted.

            """

            if not all(isinstance(value, str) and value for value in (session_id, producer, event_id, origin_turn_id)):

                raise ValueError("Session, producer, event, and origin identifiers must be non-empty strings")

            if not isinstance(messages, list) or not 1 <= len(messages) <= 2:

                raise ValueError("Passive receipt writes require one or two messages")

            canonical_messages = []

            payload_bytes = 0

            for message in messages:

                if not isinstance(message, dict) or set(message) - {"role", "content"}:

                    raise ValueError("Passive receipt messages accept only role and content")

                role, content = message.get("role"), message.get("content")

                if role not in {"user", "assistant"} or not isinstance(content, str):

                    raise ValueError("Passive receipt messages require user/assistant text")

                content_bytes = len(content.encode("utf-8"))

                if content_bytes > 65_536:

                    raise ValueError("Passive receipt message exceeds 65536 bytes")

                payload_bytes += content_bytes

                canonical_messages.append({"role": role, "content": content})

            if payload_bytes > 163_840:

                raise ValueError("Passive receipt payload exceeds 163840 bytes")

            payload = json.dumps(

                {"origin_turn_id": origin_turn_id, "messages": canonical_messages},

                ensure_ascii=False, separators=(",", ":"), sort_keys=True,

            )

            payload_digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()



            def _receipt(row, *, replayed: bool) -> Dict[str, Any]:

                message_ids = json.loads(row["message_ids"])

                return {

                    "producer": row["producer"], "event_id": row["event_id"],

                    "origin_turn_id": row["origin_turn_id"], "conversation_id": row["session_id"],

                    "session_id": row["session_id"], "message_ids": message_ids,

                    "revision": int(row["revision"]), "replayed": replayed,

                }



            def _do(conn):

                if conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone() is None:

                    raise ValueError("Passive receipt session does not exist")

                conn.execute(

                    """CREATE TABLE IF NOT EXISTS passive_message_receipts (

                        session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,

                        producer TEXT NOT NULL, event_id TEXT NOT NULL,

                        origin_turn_id TEXT NOT NULL, payload_sha256 TEXT NOT NULL,

                        message_ids TEXT NOT NULL, revision INTEGER NOT NULL,

                        PRIMARY KEY (session_id, producer, event_id)

                    )"""

                )

                existing = conn.execute(

                    "SELECT r.* FROM passive_message_receipts r JOIN sessions s ON s.id = r.session_id "
                    "WHERE r.session_id = ? AND r.producer = ? AND r.event_id = ?",

                    (session_id, producer, event_id),

                ).fetchone()

                if existing is not None:

                    if existing["payload_sha256"] != payload_digest or existing["origin_turn_id"] != origin_turn_id:

                        raise ValueError("Passive receipt event conflicts with its original payload")

                    return _receipt(existing, replayed=True)

                self._check_transcript_write_guards(conn, session_id, None)

                inserted, tool_calls = self._insert_message_rows(conn, session_id, [dict(msg) for msg in canonical_messages])

                self._bump_session_counters(conn, session_id, inserted, tool_calls, unit=False)

                rows = conn.execute(

                    "SELECT id FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?", (session_id, inserted)

                ).fetchall()

                message_ids = list(reversed([int(row["id"]) for row in rows]))

                revision = message_ids[-1]

                conn.execute(

                    """INSERT INTO passive_message_receipts

                       (session_id, producer, event_id, origin_turn_id, payload_sha256, message_ids, revision)

                       VALUES (?, ?, ?, ?, ?, ?, ?)""",

                    (session_id, producer, event_id, origin_turn_id, payload_digest,

                     json.dumps(message_ids, separators=(",", ":")), revision),

                )

                return {

                    "producer": producer, "event_id": event_id, "origin_turn_id": origin_turn_id,

                    "conversation_id": session_id, "session_id": session_id, "message_ids": message_ids,

                    "revision": revision, "replayed": False,

                }



            return self._execute_transcript_write(

                _do, [dict(message) for message in canonical_messages],

                patience_s=self._TRANSCRIPT_WRITE_PATIENCE_S,

            )



    def get_message_receipt(self, session_id: str, *, producer: str, event_id: str) -> Optional[Dict[str, Any]]:

            """Return a passive-message receipt without exposing the receipt table to plugins."""

            if not all(isinstance(value, str) and value for value in (session_id, producer, event_id)):

                raise ValueError("Session, producer, and event identifiers must be non-empty strings")

            with self._read_ctx() as conn:

                try:

                    row = conn.execute(

                        "SELECT * FROM passive_message_receipts WHERE session_id = ? AND producer = ? AND event_id = ?",

                        (session_id, producer, event_id),

                    ).fetchone()

                except Exception as exc:

                    if "no such table" in str(exc).lower():

                        return None

                    raise

                if row is None:

                    return None

                return {

                    "producer": row["producer"], "event_id": row["event_id"],

                    "origin_turn_id": row["origin_turn_id"], "conversation_id": row["session_id"],

                    "session_id": row["session_id"], "message_ids": json.loads(row["message_ids"]),

                    "revision": int(row["revision"]), "replayed": True,

                }
