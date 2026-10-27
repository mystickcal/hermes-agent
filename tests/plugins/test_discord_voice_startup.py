"""Initial Discord SPEAKING frames must reach the native voice receiver."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

from gateway.config import PlatformConfig
import plugins.platforms.discord.adapter as platform

DEREK = 121051467497865218
EUNICE = 1240252543833997384


class VoiceClient:
    def __init__(self, client, channel):
        self.channel = channel
        self.user = SimpleNamespace(id=999)
        self._connection = SimpleNamespace(
            hook=None,
            secret_key=bytes(32),
            dave_session=None,
            dave_protocol_version=0,
            dave_downgraded=False,
            ssrc=900,
            add_socket_listener=lambda callback: None,
            remove_socket_listener=lambda callback: None,
        )


class VoiceChannel:
    id = 42
    guild = SimpleNamespace(id=123)
    members = [SimpleNamespace(id=DEREK), SimpleNamespace(id=EUNICE)]

    async def connect(self, *, cls=VoiceClient):
        voice = cls(SimpleNamespace(), self)
        for ssrc, user in [(701, DEREK), (702, EUNICE)]:
            if voice._connection.hook:
                await voice._connection.hook(
                    None, {"op": 5, "d": {"ssrc": ssrc, "user_id": str(user)}}
                )
        return voice


def test_join_retains_two_speakers_announced_during_handshake(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr(platform.discord, "VoiceClient", VoiceClient)

    async def exercise():
        adapter = platform.DiscordAdapter(PlatformConfig(enabled=True))
        adapter._client = SimpleNamespace()
        adapter._allowed_user_ids = {str(DEREK), str(EUNICE)}
        adapter._voice_fx_cfg = {"enabled": False}
        adapter._reset_voice_timeout = lambda guild: None

        async def listen(guild):
            return None

        adapter._voice_listen_loop = listen
        assert await adapter.join_voice_channel(VoiceChannel())
        receiver = adapter._voice_receivers[123]
        try:
            assert receiver._ssrc_to_user == {701: DEREK, 702: EUNICE}
            assert receiver._infer_user_for_ssrc(9999) == 0
        finally:
            receiver.stop()
            await asyncio.gather(*adapter._voice_listen_tasks.values())

    asyncio.run(exercise())


def test_start_handles_speaking_once_after_hook_preinstallation():
    async def exercise():
        voice = VoiceClient(SimpleNamespace(), VoiceChannel())
        receiver = platform.VoiceReceiver(voice)
        receiver._install_speaking_hook(voice._connection)
        spy = Mock(wraps=receiver.map_ssrc)
        receiver.map_ssrc = spy
        receiver.start()
        try:
            await voice._connection.hook(
                None, {"op": 5, "d": {"ssrc": 703, "user_id": str(DEREK)}}
            )
            spy.assert_called_once_with(703, DEREK)
        finally:
            receiver.stop()

    asyncio.run(exercise())
