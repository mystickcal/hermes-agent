"""Authorized plugin command invocation on the gateway's native loop."""
import asyncio
from functools import partial
import logging
logger = logging.getLogger(__name__)

async def dispatch_quick_and_plugin_commands(
    self, event, source, command
):
    """Drain gate, user-defined quick commands (exec/alias) and plugin slash commands →
    ``(handled, result, command)``; an alias quick command rewrites ``command``."""
    from gateway.run_inbound import build_session_context, t
    if self._draining:
        return True, t("gateway.busy.drain_rejected_new_work", action=self._status_action_gerund()), command

    # User-defined quick commands (bypass agent loop, no LLM call)
    qcmd = self._hm_quick_commands().get(command) if command else None
    if qcmd is not None:
        # Quick commands are slash capabilities too — and type:exec ones run a shell command in
        # the gateway process. They are never in the registry, so the early gate never fires for
        # them; apply the same admin/user policy to the raw typed name here.
        # The early gate above only fires for registry-known commands, so quick commands (never in the
        # registry) would otherwise reach this dispatch sink unchecked. (#44727)
        _denied = self._check_slash_access(source, command)
        if _denied is not None:
            return True, _denied, command
        qtype = qcmd.get("type")
        if qtype == "exec":
            exec_cmd = qcmd.get("command", "")
            if not exec_cmd:
                return True, t("gateway.quick_command.no_command", command=command), command
            return True, await self._hm_run_exec_quick_command(command, exec_cmd), command
        if qtype != "alias":
            return True, t("gateway.quick_command.unsupported_type", command=command), command
        new_command = self._hm_expand_alias_quick_command(event, qcmd)
        if new_command is None:
            return True, t("gateway.quick_command.no_target", command=command), command
        command = new_command  # Fall through to normal command dispatch below

    # Plugin-registered slash commands. Underscores normalize to hyphens so Telegram's
    # underscored autocomplete form matches plugin commands registered with hyphens.
    if command:
        try:
            from hermes_cli.plugins import get_plugin_command_entry
            plugin_entry = get_plugin_command_entry(command.replace("_", "-"))
            plugin_handler = plugin_entry["handler"] if plugin_entry else None
            if plugin_handler:
                # The agent-turn path binds HERMES_SESSION_* via _set_session_env; this dispatch
                # sits before it, so a handler reading get_session_env() would see an empty or a
                # foreign (cron agent's os.environ) session (#108698). No session_entry exists yet,
                # so session_key is derived from source. Sync handlers run on the gateway pool
                # (contextvars carried), never the loop thread: blocking I/O there starves the
                # liveness watchdog and the process exits 75 mid-handler (#105279).
                _plugin_context = build_session_context(source, self.config)
                _plugin_context.session_key = self._session_key_for_source(source)
                user_args = event.get_command_args().strip()
                invocation_context = None
                if plugin_entry.get("invocation_context"):
                    from gateway.discord_task_context import native_plugin_invocation_context
                    anchor_entry = await self._run_in_executor_with_context(partial(
                        self.session_store.get_or_create_session, source,
                        force_new=False, touch_activity=False))
                    anchor_session_id = str(anchor_entry.session_id)
                    invocation_context = await native_plugin_invocation_context(
                        event, source, self._delivery_adapter_for(source), anchor_session_id)
                with self._session_env_scope(_plugin_context):
                    if asyncio.iscoroutinefunction(plugin_handler):
                        result = await (plugin_handler(user_args, invocation_context) if invocation_context is not None
                                        else plugin_handler(user_args))
                    else:
                        args = (user_args, invocation_context) if invocation_context is not None else (user_args,)
                        result = await self._run_in_executor_with_context(plugin_handler, *args)
                        if asyncio.iscoroutine(result):
                            result = await result
                return True, str(result) if result else None, command
        except Exception:
            logger.exception('Plugin command dispatch failed (details redacted)',
                exc_info=(RuntimeError, RuntimeError("Native boundary error (details redacted)"), None),
                stack_info=True)
    return False, None, command
