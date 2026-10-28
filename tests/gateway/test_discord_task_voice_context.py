"""Task room admission through the real native adapter and Discord SDK cache."""
import asyncio
import sqlite3
from types import SimpleNamespace

import discord
import pytest

from gateway.config import PlatformConfig
from gateway.discord_task_context import _explicitly_allowed
from gateway.native_task_host import (
    NativeTaskHostProvider, native_task_host_context, register_native_task_host_provider,
    restore_native_task_host_provider,
)
from plugins.platforms.discord.adapter import DiscordAdapter
from plugins.platforms.discord.task_voice_context import snapshot_voice_room


def user_data(ident, *, bot=False):
    return {"id": str(ident), "username": "fixture", "discriminator": "0", "avatar": None, "bot": bot}


@pytest.fixture
def room():
    client = discord.Client(intents=discord.Intents.all())
    state = client._connection
    state.user = discord.ClientUser(state=state, data=user_data(99, bot=True))
    guild = discord.Guild(state=state, data={
        "id": "10", "name": "fixture", "owner_id": "11", "features": [],
        "roles": [{"id": "10", "name": "@everyone", "permissions": "104324673", "position": 0}],
        "emojis": [], "members": [], "channels": [],
    })
    state._guilds[10] = guild
    channel = discord.VoiceChannel(state=state, guild=guild, data={
        "id": "20", "name": "voice", "type": 2, "position": 0,
        "permission_overwrites": [], "bitrate": 64000, "user_limit": 0,
    })
    guild._channels[20] = channel
    for ident in (11, 12, 99):
        guild._members[ident] = discord.Member(state=state, guild=guild, data={
            "user": user_data(ident, bot=ident == 99), "roles": [], "flags": 0,
            "joined_at": "2020-01-01T00:00:00+00:00", "deaf": False, "mute": False,
        })
        guild._voice_states[ident] = discord.VoiceState(channel=channel, data={
            "user_id": str(ident), "session_id": "fixture-session", "self_mute": False,
            "self_deaf": False, "mute": False, "deaf": False, "suppress": False,
        })
    interaction = discord.Interaction(state=state, data={
        "id": "1000", "type": 2, "application_id": "99", "token": "fixture-token",
        "version": 1, "guild_id": "10", "channel_id": "20",
        "attachment_size_limit": 10485760, "channel": {"id": "20", "type": 2},
        "data": {"id": "1001", "name": "talk", "type": 1},
        "member": {"user": user_data(11), "roles": [], "flags": 0, "permissions": "104324673",
                   "joined_at": "2020-01-01T00:00:00+00:00", "deaf": False, "mute": False},
    })
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="fixture-token"))
    adapter._client = client
    adapter._allowed_user_ids, adapter._allowed_role_ids = {"11", "12"}, set()
    adapter._voice_clients[10] = SimpleNamespace(
        channel=channel, session_id="fixture-session", is_connected=lambda: True,
    )
    return adapter, guild, channel, interaction


def test_native_room_requires_explicit_humans_and_present_operator(room):
    adapter, guild, channel, interaction = room
    current = snapshot_voice_room(adapter, guild, interaction.user)
    assert current["audience_user_ids"] == [11, 12]
    public = adapter.native_task_room_context(10, 20, 11)
    assert public["surface"] == "discord" and public["audience_revision"] == current["audience_revision"]
    assert adapter.native_task_room_context(10, 999, 11) is None
    assert _explicitly_allowed(adapter, guild.get_member(11))
    adapter._allowed_user_ids = {"*"}
    assert snapshot_voice_room(adapter, guild, interaction.user) is None
    adapter._allowed_user_ids = {"11", "12"}
    guild._voice_states.pop(11)
    assert snapshot_voice_room(adapter, guild, interaction.user) is None


def test_room_revision_changes_for_permissions_connection_and_membership(room):
    adapter, guild, channel, interaction = room
    first = snapshot_voice_room(adapter, guild, interaction.user)
    guild.default_role._permissions ^= 1024
    second = snapshot_voice_room(adapter, guild, interaction.user)
    assert second["audience_revision"] != first["audience_revision"]
    adapter._voice_clients[10].session_id = "reconnected"
    third = snapshot_voice_room(adapter, guild, interaction.user)
    assert third["audience_revision"] != second["audience_revision"]
    adapter._allowed_user_ids.remove("12")
    assert snapshot_voice_room(adapter, guild, interaction.user) is None


def test_actual_adapter_mints_only_native_context_and_rechecks_room(room, tmp_path, monkeypatch):
    adapter, guild, channel, interaction = room
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sqlite3.connect(tmp_path / "state.db").close()
    context = native_task_host_context()
    calls = []
    class Provider(NativeTaskHostProvider):
        def invoke(self, operation, trusted, payload):
            calls.append((operation, trusted, payload))
            return {"ok": True, "proof": "private-native-proof"}
    register_native_task_host_provider(Provider(), profile_home=tmp_path)
    source = SimpleNamespace(user_id="11")
    try:
        result = asyncio.run(adapter.native_task_invocation_context(interaction, source, "native-anchor"))
        assert result["version"] == 1
        assert result["discord_task_context"]["anchor_session_id"] == "native-anchor"
        operation, trusted, payload = calls[0]
        assert operation == "discord.mint" and payload == {"session_id": "native-anchor"}
        assert {k: trusted[k] for k in context} == context
        assert trusted["room"]["audience_user_ids"] == [11, 12]
        assert asyncio.run(adapter.native_task_invocation_context({}, source, "native-anchor")) is None
        guild._voice_states.pop(12)
        class ChangingProvider(NativeTaskHostProvider):
            def invoke(self, operation, trusted, payload):
                adapter._allowed_user_ids.remove("11")
                return {"ok": True, "proof": "stale-proof"}
        register_native_task_host_provider(ChangingProvider(), profile_home=tmp_path)
        assert asyncio.run(adapter.native_task_invocation_context(interaction, source, "native-anchor")) is None
    finally:
        restore_native_task_host_provider(None, profile_home=tmp_path)
