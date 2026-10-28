"""Native voice startup and the live room authority used by task attachments."""
from __future__ import annotations
import asyncio
import hashlib
import json
import logging
from gateway.discord_task_context import _explicitly_allowed, _numeric_ids
logger = logging.getLogger(__name__)

def _id(value):
    ident = getattr(value, "id", None)
    return ident if type(ident) is int and 0 < ident < 2**63 else None

def _advance_generation(adapter, guild_id):
    generations = getattr(adapter, "_native_task_voice_generations", None)
    if generations is None:
        generations = adapter._native_task_voice_generations = {}
    generations[guild_id] = generations.get(guild_id, 0) + 1

def _member_revision(channel, member):
    permissions = channel.permissions_for(member)
    return [_id(member), member.bot,
            sorted(_id(role) for role in member.roles if _id(role) is not None),
            permissions.value]

def snapshot_voice_room(adapter, guild, operator):
    guild_id, operator_id = _id(guild), _id(operator)
    if guild_id is None or operator_id is None or not _explicitly_allowed(adapter, operator):
        return None
    voice = getattr(adapter, "_voice_clients", {}).get(guild_id)
    if voice is None or not voice.is_connected():
        return None
    channel = voice.channel
    if _id(channel.guild) != guild_id or _id(channel) is None:
        return None
    members = tuple(channel.members)
    bot_id = _id(getattr(getattr(adapter, "_client", None), "user", None))
    ids = [_id(member) for member in members]
    if None in ids or len(ids) != len(set(ids)) or operator_id not in ids or bot_id not in ids:
        return None
    # A second bot can receive audio too; it is not covered by human listener grants.
    if any(type(getattr(m, "bot", None)) is not bool or (m.bot and m.id != bot_id) for m in members):
        return None
    humans = tuple(member for member in members if not member.bot)
    if not humans or any(not _explicitly_allowed(adapter, member) for member in humans):
        return None
    revision = [guild_id, channel.id, id(voice), getattr(voice, "session_id", None),
                getattr(adapter, "_native_task_voice_generations", {}).get(guild_id, 0),
                [_member_revision(channel, m) for m in sorted(members, key=lambda m: m.id)],
                sorted(_numeric_ids(getattr(adapter, "_allowed_user_ids", ()) or ())),
                sorted(_numeric_ids(getattr(adapter, "_allowed_role_ids", ()) or ()))]
    return {
        "bot_connected": True, "operator_present": True, "native_authorized": True,
        "guild_id": guild_id, "channel_id": channel.id, "operator_user_id": operator_id,
        "audience_user_ids": sorted(m.id for m in humans),
        "audience_revision": hashlib.sha256(json.dumps(revision, separators=(",", ":")).encode()).hexdigest(),
    }

def current_discord_room(adapter, locator):
    if not isinstance(locator, dict):
        return None
    values = [locator.get(k) for k in ("guild_id", "channel_id", "operator_user_id")]
    if any(type(v) is not int or v <= 0 for v in values):
        return None
    guild = adapter._client.get_guild(values[0]) if adapter and adapter._client else None
    if guild is None:
        return None
    operator = guild.get_member(values[2])
    room = snapshot_voice_room(adapter, guild, operator)
    return room if room and room["channel_id"] == values[1] else None

async def native_task_invocation_context(adapter, interaction, source, anchor_session_id):
    if (not isinstance(anchor_session_id, str) or not anchor_session_id
            or str(getattr(source, "user_id", "")) != str(_id(interaction.user))):
        return None
    room = snapshot_voice_room(adapter, interaction.guild, interaction.user)
    if room is None:
        return None
    from gateway.native_task_host import get_native_task_host_provider, native_task_host_context
    context = native_task_host_context()
    provider = get_native_task_host_provider(context["profile_home"])
    if provider is None:
        return None
    result = await asyncio.to_thread(provider.invoke, "discord.mint", {**context, "room": room},
                                     {"session_id": anchor_session_id})
    if not isinstance(result, dict) or result.get("ok") is not True or not isinstance(result.get("proof"), str):
        return None
    # Recheck after the ledger write; do not return a grant for a room that changed while awaiting it.
    if snapshot_voice_room(adapter, interaction.guild, interaction.user) != room:
        return None
    return {"version": 1, "discord_task_context": {
        **{k: room[k] for k in ("guild_id", "channel_id", "operator_user_id")},
        "profile": context["profile_name"], "anchor_session_id": anchor_session_id,
        "proof": result["proof"],
    }}

async def join_voice_channel(self, channel, *, receiver_type, discord_available, text_channel_id=None, source=None):
    """Join a voice channel; returns True on success. ``text_channel_id`` stores the
    transcription-routing binding so programmatic joins work without ``/voice join``."""
    if not self._client or not discord_available:
        return False
    guild_id = channel.guild.id
    async with self._voice_locks.setdefault(guild_id, asyncio.Lock()):
        existing = self._voice_clients.get(guild_id)
        if existing and existing.is_connected():
            if existing.channel.id == channel.id:
                self._reset_voice_timeout(guild_id)
                return True
            await existing.move_to(channel)
            _advance_generation(self, guild_id)
            self._reset_voice_timeout(guild_id)
            return True
        try:
            from voice_startup import connect_with_receiver
        except ImportError:
            from .voice_startup import connect_with_receiver
        vc, receiver = await connect_with_receiver(
            channel, lambda voice: receiver_type(voice, allowed_user_ids=self._allowed_user_ids)
        )
        self._voice_clients[guild_id] = vc
        _advance_generation(self, guild_id)
        self._reset_voice_timeout(guild_id)
        if text_channel_id is not None:
            self._voice_text_channels[guild_id] = text_channel_id
        if source is not None:
            self._voice_sources[guild_id] = source
        try:
            if receiver is None:
                receiver = receiver_type(vc, allowed_user_ids=self._allowed_user_ids)
            receiver.start()
            self._voice_receivers[guild_id] = receiver
            self._voice_listen_tasks[guild_id] = asyncio.ensure_future(
                self._voice_listen_loop(guild_id)
            )
        except Exception as e:
            logger.warning("Voice receiver failed to start: %s", e)
        # Mixer is best-effort; failure falls back to one-shot FFmpegPCMAudio playback.
        if getattr(self, "_voice_fx_cfg", {}).get("enabled"):
            try:
                await self._install_voice_mixer(guild_id, vc)
            except Exception as e:
                logger.warning("Voice mixer failed to start: %s", e)
        return True
