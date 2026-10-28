"""Profile-scoped bridge for native task-host plugins.

The gateway owns registration and lifetime.  Providers receive only JSON-safe values so an
isolated plugin never receives an adapter, request, or Discord SDK object.
"""
from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod
from typing import Any


class NativeTaskHostProvider(ABC):
    """One synchronous, JSON-safe provider boundary for native task integrations."""

    @abstractmethod
    def invoke(self, operation: str, context: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
        """Perform *operation* for an already authenticated host request."""


def native_task_host_scope(profile_home: str | os.PathLike[str] | None = None) -> str:
    """Return the authoritative profile-home registry key without reading ambient state later."""
    if profile_home is None:
        from hermes_constants import get_hermes_home
        profile_home = get_hermes_home()
    return os.path.normcase(os.path.abspath(os.fspath(profile_home)))


_LOCK = threading.RLock()
_PROVIDERS: dict[str, NativeTaskHostProvider] = {}


def get_native_task_host_provider(profile_home: str | os.PathLike[str] | None = None) -> NativeTaskHostProvider | None:
    with _LOCK:
        return _PROVIDERS.get(native_task_host_scope(profile_home))


def snapshot_native_task_host_provider(profile_home: str | os.PathLike[str] | None = None) -> NativeTaskHostProvider | None:
    return get_native_task_host_provider(profile_home)


def register_native_task_host_provider(provider: NativeTaskHostProvider, *, profile_home: str | os.PathLike[str] | None = None) -> None:
    if not isinstance(provider, NativeTaskHostProvider):
        raise TypeError('provider must inherit NativeTaskHostProvider')
    with _LOCK:
        _PROVIDERS[native_task_host_scope(profile_home)] = provider


def restore_native_task_host_provider(previous: NativeTaskHostProvider | None, *, profile_home: str | os.PathLike[str] | None = None) -> bool:
    key = native_task_host_scope(profile_home)
    with _LOCK:
        if previous is None:
            _PROVIDERS.pop(key, None)
        else:
            _PROVIDERS[key] = previous
    return True


# A host restart invalidates every outstanding surface attachment. Isolated
# plugins obtain this value from their parent context, never their own PID.
import secrets
HOST_EPOCH = secrets.token_hex(24)


def native_task_host_context(*, profile_home=None, profile_name=None):
    """Resolve the actual native store; callers must authenticate before using it."""
    import hashlib
    from pathlib import Path
    from hermes_constants import get_hermes_home
    from hermes_cli.profiles import get_active_profile_name
    home = Path(profile_home) if profile_home is not None else get_hermes_home()
    home = home.resolve(strict=True)
    state = (home / "state.db").resolve(strict=True)
    stat = state.stat()
    store_id = hashlib.sha256(f"{state}\0{stat.st_dev}\0{stat.st_ino}".encode()).hexdigest()
    return {
        "principal_id": "api-server", "principal_kind": "native_bearer",
        "profile_name": profile_name or get_active_profile_name(),
        "profile_home": str(home), "store_id": store_id, "host_epoch": HOST_EPOCH,
    }
