"""Prepare Discord speaker attribution before the voice handshake emits events."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


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
