"""Scoped context-capability registration used by PluginContext."""
from __future__ import annotations
import logging
logger = logging.getLogger(__name__)

def register_context_engine(self, engine):
    """Register the (single) ``agent.context_engine.ContextEngine`` replacing the built-in
    ContextCompressor; a second registration is rejected with a warning."""
    if self._manager._context_engine is not None:
        logger.warning("Plugin '%s' tried to register a context engine, but one is "
                       "already registered. Only one context engine plugin is allowed.",
                       self.manifest.name)
        return
    from agent.context_engine import ContextEngine
    if self._wrong_type(engine, ContextEngine, "context engine"):
        return
    previous = self._manager._context_engine  # always None here; kept for the restore contract
    self._manager._context_engine = engine
    handle = self._track_replacement(
        "context_engine", engine.name, slot=("manager_value", id(self._manager), "_context_engine"),
        current=engine, previous=previous,
        restore=lambda replacement: self._manager._restore_value("_context_engine", engine, replacement),
    )
    logger.info("Plugin '%s' registered context engine: %s", self.manifest.name, engine.name)
    return handle


def register_context_reference(self, provider) -> None:
    """Register a :class:`agent.context_references.ContextReferenceProvider`; ``provider.prefix``
    defines ``@<prefix>:``. Built-in prefixes (diff, staged, file, folder, git, url) are
    rejected."""
    from agent.context_references import (
        ContextReferenceProvider as _CRP, register_context_reference_provider as _register,
    )
    if self._wrong_type(provider, _CRP, "context reference provider"):
        return
    try:
        _register(provider)
    except ValueError as exc:
        logger.warning("Plugin '%s' context reference registration failed: %s", self.manifest.name, exc)
        return
    logger.info("Plugin '%s' registered context reference: @%s:", self.manifest.name, provider.prefix)


def register_native_task_host_provider(self, provider):
    """Register this profile's JSON-safe native task-host provider."""
    from gateway import native_task_host
    if self._wrong_type(provider, native_task_host.NativeTaskHostProvider, "native task-host provider", "a"):
        return None
    scope = self._manager.scope_key
    previous = native_task_host.snapshot_native_task_host_provider(scope)
    native_task_host.register_native_task_host_provider(provider, profile_home=scope)
    return self._track_replacement(
        "native_task_host_provider", "native",
        slot=("native_task_host_provider", scope), current=provider, previous=previous,
        restore=lambda replacement: native_task_host.restore_native_task_host_provider(
            replacement, profile_home=scope),
    )
