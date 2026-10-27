"""Prepare Discord speaker attribution before the voice handshake emits events."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def install_speaking_hook(receiver, conn):
    """Wrap the voice websocket hook to capture SPEAKING events (op 5); ``conn.hook`` is
    re-passed on each (re)connect, so wrap it on the state AND the live websocket."""
    if receiver._speaking_hook is not None and conn.hook is receiver._speaking_hook:
        return
    original_hook = conn.hook
    receiver_self = receiver

    async def wrapped_hook(ws, msg):
        if isinstance(msg, dict) and msg.get("op") == 5:
            data = msg.get("d", {})
            ssrc = data.get("ssrc")
            user_id = data.get("user_id")
            if ssrc and user_id:
                logger.info("SPEAKING event: ssrc=%d -> user=%s", ssrc, user_id)
                receiver_self.map_ssrc(int(ssrc), int(user_id))
        if original_hook:
            await original_hook(ws, msg)
        # Re-resolve decryption state after the events that change it.
        # DiscordVoiceWebSocket.received_message dispatches the op (which
        # is what updates conn.secret_key / conn.dave_session /
        # conn.dave_protocol_version) BEFORE calling this hook, so the
        # refresh reads the new values:
        #   op 4  SESSION_DESCRIPTION — new transport key, and
        #         reinit_dave_session() may replace conn.dave_session
        #   op 24 DAVE_PREPARE_EPOCH  — epoch 1 recreates the MLS group
        #   op 21 DAVE_PREPARE_TRANSITION with protocol_version 0 —
        #         discord.py opens a 120s plaintext passthrough window
        #         (set_passthrough_mode(True, 120)); mirror it so the
        #         unmapped-SSRC gate doesn't drop legitimate plaintext
        #   op 22 DAVE_EXECUTE_TRANSITION — the protocol version just
        #         changed; refresh, plus a 10s grace mirroring the
        #         upgrade path's set_passthrough_mode(True, 10)
        if isinstance(msg, dict):
            op = msg.get("op")
            if op in (4, 24):
                receiver_self.refresh_credentials(f"voice ws op {op}")
            elif op == 21:
                if (msg.get("d") or {}).get("protocol_version") == 0:
                    # Pending downgrade: upstream enables plaintext
                    # passthrough for up to 120s (set_passthrough_mode).
                    receiver_self.note_dave_passthrough_window(120.0)
            elif op == 22:
                # A transition may have executed; refresh and detect the
                # upgrade-after-downgrade EDGE (dave_downgraded flipping
                # True -> False) — the only case upstream grants the 10s
                # passthrough grace.  Same-version transitions and
                # unknown/duplicate ids change no state upstream, so no
                # edge fires and no window opens.
                was_downgraded = receiver_self._dave_downgraded
                receiver_self.refresh_credentials("voice ws op 22")
                if (
                    was_downgraded
                    and not receiver_self._dave_downgraded
                    and receiver_self._dave_protocol_version > 0
                ):
                    receiver_self.note_dave_passthrough_window(10.0)

    # Set on connection state (for future reconnects)
    conn.hook = wrapped_hook
    receiver._speaking_hook = wrapped_hook
    try:
        from discord.utils import MISSING

        if hasattr(conn, "ws") and conn.ws is not MISSING:
            conn.ws._hook = wrapped_hook
            logger.info("Speaking hook installed on live websocket")
    except Exception as e:
        logger.warning("Could not install hook on live ws: %s", e)


async def connect_with_receiver(channel, receiver_factory):
    """Join with the receiver's normal hook already installed on connection state.

    Discord announces users already speaking during connect(). Installing the
    hook afterwards loses those SSRC/user pairs, and a room with two people
    cannot safely recover them through the single-member inference path.
    """
    from discord import VoiceClient

    receiver = None

    def create_voice_client(client, voice_channel):
        nonlocal receiver
        voice = VoiceClient(client, voice_channel)
        try:
            receiver = receiver_factory(voice)
            receiver._install_speaking_hook(voice._connection)
        except Exception:
            # Joining for playback remains possible when capture is unavailable,
            # matching the adapter's existing best-effort receiver startup.
            logger.warning("Voice receiver handshake hook unavailable", exc_info=True)
            receiver = None
        return voice

    voice = await channel.connect(cls=create_voice_client)
    return voice, receiver
