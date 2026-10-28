"""Native Discord task command admission and explicit listener grants."""
from __future__ import annotations
import inspect

def _numeric_ids(values):
    return {int(v) for v in values if isinstance(v, (str, int))
            and not isinstance(v, bool) and str(v).isdecimal() and int(v) > 0}

def _explicitly_allowed(adapter, member):
    if (member is None or type(getattr(member, "id", None)) is not int
            or getattr(member, "bot", None) is not False):
        return False
    users = _numeric_ids(getattr(adapter, "_allowed_user_ids", ()) or ())
    roles = _numeric_ids(getattr(adapter, "_allowed_role_ids", ()) or ())
    member_roles = {r.id for r in getattr(member, "roles", ()) if type(getattr(r, "id", None)) is int}
    return member.id in users or bool(roles & member_roles)

async def native_plugin_invocation_context(event, source, adapter, anchor_session_id):
    build = getattr(adapter, "native_task_invocation_context", None)
    if not callable(build):
        return None
    envelope = build(getattr(event, "raw_message", None), source, anchor_session_id)
    if inspect.isawaitable(envelope):
        envelope = await envelope
    if (not isinstance(envelope, dict) or type(envelope.get("version")) is not int
            or envelope["version"] != 1 or not isinstance(envelope.get("discord_task_context"), dict)):
        return None
    return envelope
