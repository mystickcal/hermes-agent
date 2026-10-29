"""Resolved Discord prompts must not be repainted by a queued expiry callback.

Load the candidate adapter explicitly so scope-sparse test runs cannot
accidentally exercise a different installed adapter.
"""
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import plugins.platforms.discord as discord_package

# Keep the canonical package name so sibling relative imports work, while
# selecting the exact candidate file rather than the installed adapter.
_candidate_dir = Path(__file__).resolve().parents[2] / "plugins/platforms/discord"
discord_package.__path__.insert(0, str(_candidate_dir))
sys.modules.pop("plugins.platforms.discord.adapter", None)
adapter = importlib.import_module("plugins.platforms.discord.adapter")
assert adapter.__file__ is not None
assert Path(adapter.__file__).resolve() == (_candidate_dir / "adapter.py").resolve()


@pytest.fixture(autouse=True)
def _view_stop_contract(monkeypatch):
    """Older gateway SDK doubles omit discord.ui.View's timer lifecycle API."""
    base = adapter.discord.ui.View
    if not hasattr(base, "stop"):
        monkeypatch.setattr(base, "stop", lambda self: setattr(self, "_test_stopped", True), raising=False)
        monkeypatch.setattr(base, "is_finished", lambda self: getattr(self, "_test_stopped", False), raising=False)


def interaction(user_id=1240252543833997384):
    embed = SimpleNamespace(color=None, footer="")
    embed.set_footer = lambda *, text: setattr(embed, "footer", text)
    message = SimpleNamespace(embeds=[embed], edit=AsyncMock())
    return SimpleNamespace(
        user=SimpleNamespace(id=user_id, display_name="Eunice", roles=[]),
        message=message,
        response=SimpleNamespace(edit_message=AsyncMock(), send_message=AsyncMock(), defer=AsyncMock()),
    )


def approval():
    return adapter.ExecApprovalView(session_key="test-only", allowed_user_ids={"1240252543833997384"})


def clarify():
    return adapter.ClarifyChoiceView(choices=["Continue"], clarify_id="test-only", allowed_user_ids={"1240252543833997384"})


@pytest.mark.asyncio
@pytest.mark.parametrize("choice", ["once", "session", "always", "deny"])
async def test_command_decision_survives_late_timeout(monkeypatch, choice):
    import tools.approval
    resolve = Mock(return_value=1)
    monkeypatch.setattr(tools.approval, "resolve_gateway_approval", resolve)
    view, event = approval(), interaction()
    view._message = event.message
    await view._resolve(event, choice, None, "platform.discord.approval.resolved_once")
    footer = event.message.embeds[0].footer
    assert view.resolved and view.is_finished()
    assert "Eunice" in footer
    resolve.assert_called_once_with("test-only", choice)
    await view.on_timeout()  # already-queued timeout must also be harmless
    assert event.message.embeds[0].footer == footer
    event.message.edit.assert_not_awaited()


@pytest.mark.asyncio
async def test_clarification_answer_survives_late_timeout(monkeypatch):
    import tools.clarify_gateway
    resolve = Mock(return_value=True)
    monkeypatch.setattr(tools.clarify_gateway, "resolve_gateway_clarify", resolve)
    view, event = clarify(), interaction()
    view._message = event.message
    await view._resolve_choice(event, 0, "Continue")
    footer = event.message.embeds[0].footer
    assert view.resolved and view.is_finished()
    resolve.assert_called_once_with("test-only", "Continue")
    await view.on_timeout()
    assert event.message.embeds[0].footer == footer
    event.message.edit.assert_not_awaited()


@pytest.mark.asyncio
async def test_other_still_arms_backend_text_capture(monkeypatch):
    import tools.clarify_gateway
    mark = Mock()
    resolve = Mock()
    monkeypatch.setattr(tools.clarify_gateway, "mark_awaiting_text", mark)
    monkeypatch.setattr(tools.clarify_gateway, "resolve_gateway_clarify", resolve)
    view, event = clarify(), interaction()
    view._message = event.message
    await view._on_other(event)
    footer = event.message.embeds[0].footer
    mark.assert_called_once_with("test-only")
    resolve.assert_not_called()  # clicking Other is not a completed backend answer
    assert view.is_finished()
    await view.on_timeout()
    assert event.message.embeds[0].footer == footer
    event.message.edit.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", [approval, clarify])
async def test_unanswered_prompt_still_expires(factory):
    view, event = factory(), interaction()
    view._message = event.message
    await view.on_timeout()
    assert view.resolved
    assert all(child.disabled for child in view.children)
    assert "expired" in event.message.embeds[0].footer.lower()
    event.message.edit.assert_awaited_once()
    view.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("factory", [approval, clarify])
async def test_unauthorized_user_cannot_resolve(factory, monkeypatch):
    for key in ("DISCORD_ALLOW_ALL_USERS", "GATEWAY_ALLOW_ALL_USERS", "GATEWAY_ALLOWED_USERS"):
        monkeypatch.delenv(key, raising=False)
    view, event = factory(), interaction(user_id=999)
    assert not await view._gate(event, resolved_msg="Already answered", unauth_msg="Unauthorized")
    assert not view.resolved and not view.is_finished()
    event.response.send_message.assert_awaited_once_with("Unauthorized", ephemeral=True)
    event.response.edit_message.assert_not_awaited()
    view.stop()


@pytest.mark.asyncio
async def test_clarify_edit_failure_keeps_accepted_state(monkeypatch):
    import tools.clarify_gateway
    monkeypatch.setattr(tools.clarify_gateway, "resolve_gateway_clarify", Mock(return_value=True))
    view, event = clarify(), interaction()
    event.response.edit_message.side_effect = RuntimeError("simulated expired interaction token")
    view._message = event.message
    await view._resolve_choice(event, 0, "Continue")
    assert view.resolved and view.is_finished()
    event.response.defer.assert_awaited_once()
    await view.on_timeout()
    event.message.edit.assert_not_awaited()
