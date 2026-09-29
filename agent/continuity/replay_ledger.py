"""Conservative native tool replay protection for one guarded consumer session."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import threading
import time
from pathlib import Path
from typing import Any, Callable

_LOCK_MODE = 0o600
_DIR_MODE = 0o700
_FILE_MODE = 0o600
_TERMINAL = {"completed", "unknown"}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def invocation_hash(tool_name: str, arguments: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical({"tool": tool_name, "args": arguments}).encode()).hexdigest()


def call_fingerprint(session_id: str, turn_id: str, tool_call_id: str) -> str:
    """Identity is stable session + native turn + call ID; args are not identity."""
    return hashlib.sha256(_canonical({"session": session_id, "turn": turn_id, "call": tool_call_id}).encode()).hexdigest()


def _assert_owned_regular(path: Path, *, mode: int) -> None:
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise RuntimeError(f"continuity path is not a regular file: {path}")
    if st.st_uid != os.getuid():
        raise RuntimeError(f"continuity path is not owned by current user: {path}")
    if stat.S_IMODE(st.st_mode) != mode:
        raise RuntimeError(f"continuity path has unsafe mode: {path}")


def _assert_owned_directory(path: Path, *, mode: int) -> None:
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        raise RuntimeError(f"continuity path is not a directory: {path}")
    if st.st_uid != os.getuid():
        raise RuntimeError(f"continuity path is not owned by current user: {path}")
    if stat.S_IMODE(st.st_mode) != mode:
        raise RuntimeError(f"continuity path has unsafe mode: {path}")


class _FileLock:
    def __init__(self, path: Path):
        self.path = path
        self.fd: int | None = None

    def __enter__(self):
        if self.path.exists():
            _assert_owned_regular(self.path, mode=_LOCK_MODE)
        self.fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, _LOCK_MODE)
        os.fchmod(self.fd, _LOCK_MODE)
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *_):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None


class ReplayLedger:
    """A custody-checked JSONL ledger with atomic cross-process claims."""

    def __init__(self, path: str | os.PathLike[str], *, session_id: str, turn_id: str):
        self.path = Path(path).expanduser()
        self.session_id, self.turn_id = str(session_id), str(turn_id)
        if not self.session_id or not self.turn_id:
            raise ValueError("native session_id and stable turn_id are required")
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=_DIR_MODE)
        _assert_owned_directory(self.path.parent, mode=_DIR_MODE)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self._lock = threading.RLock()
        if self.path.exists():
            _assert_owned_regular(self.path, mode=_FILE_MODE)
        elif self.path.is_symlink():
            raise RuntimeError("continuity ledger cannot be a symlink")
        if self.lock_path.exists():
            _assert_owned_regular(self.lock_path, mode=_LOCK_MODE)

    def _records(self) -> tuple[dict[str, dict[str, Any]], bool]:
        found: dict[str, dict[str, Any]] = {}
        if not self.path.exists():
            return found, True
        _assert_owned_regular(self.path, mode=_FILE_MODE)
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            return {}, False
        valid = True
        for line in lines:
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    valid = False; continue
            except (json.JSONDecodeError, TypeError):
                valid = False; continue
            if (row.get("session_id"), row.get("turn_id")) != (self.session_id, self.turn_id):
                continue
            fp = row.get("fingerprint")
            if isinstance(fp, str) and fp:
                found[fp] = row
        return found, valid

    def _append(self, fingerprint: str, status: str, **extra: Any) -> None:
        row = {"version": 2, "ts": time.time(), "session_id": self.session_id,
               "turn_id": self.turn_id, "fingerprint": fingerprint, "status": status, **extra}
        if self.path.exists():
            _assert_owned_regular(self.path, mode=_FILE_MODE)
        else:
            fd0 = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, _FILE_MODE)
            os.close(fd0)
        os.chmod(self.path, _FILE_MODE)
        _assert_owned_regular(self.path, mode=_FILE_MODE)
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_NOFOLLOW, _FILE_MODE)
        try:
            os.write(fd, (_canonical(row) + "\n").encode())
            os.fsync(fd)
        finally:
            os.close(fd)

    def state(self, tool_call_id: str) -> dict[str, Any] | None:
        fp = call_fingerprint(self.session_id, self.turn_id, tool_call_id)
        with self._lock, _FileLock(self.lock_path):
            records, valid = self._records()
            return records.get(fp) if valid else None

    def dispatch(self, tool_call_id: str, tool_name: str, arguments: dict[str, Any], leaf: Callable[[dict[str, Any]], Any]) -> tuple[Any, str]:
        fp = call_fingerprint(self.session_id, self.turn_id, tool_call_id)
        args_hash = invocation_hash(tool_name, arguments)
        with self._lock, _FileLock(self.lock_path):
            records, valid = self._records()
            prior = records.get(fp) if valid else {"status": "unknown"}
            if prior:
                if prior.get("invocation_hash") != args_hash:
                    return {"error": "Tool call identity reused with different invocation", "status": "unknown"}, "unknown"
                if prior.get("status") == "completed":
                    return prior.get("result"), "replayed"
                return {"error": "Tool call was not replayed automatically", "status": "unknown"}, "unknown"
            # Claim is committed while the cross-process lock is held. No args are persisted.
            self._append(fp, "started", tool_call_id=tool_call_id, tool_name=tool_name, invocation_hash=args_hash)
        try:
            result = leaf(arguments)
        except BaseException as exc:
            with self._lock, _FileLock(self.lock_path):
                self._append(fp, "unknown", invocation_hash=args_hash, error_type=type(exc).__name__)
            raise
        with self._lock, _FileLock(self.lock_path):
            self._append(fp, "completed", invocation_hash=args_hash, result=result)
        return result, "completed"


def _stable_turn_id(agent: Any) -> str:
    for name in ("_current_turn_id", "current_turn_id", "stable_turn_id"):
        value = getattr(agent, name, None)
        if value:
            return str(value)
    raise ValueError("native stable turn identity is required; API request IDs are not accepted")


def ledger_for_agent(agent: Any, task_id: str = "") -> ReplayLedger:
    path = os.environ.get("HERMES_CONTINUITY_REPLAY_LEDGER", "~/.hermes/continuity/replay-ledger.jsonl")
    return ReplayLedger(path, session_id=str(getattr(agent, "session_id", "") or ""), turn_id=_stable_turn_id(agent))
