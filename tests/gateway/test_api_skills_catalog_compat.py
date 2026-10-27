"""Catalog routes must use the installed discovery APIs and retain bearer auth."""
import asyncio

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


@pytest.mark.parametrize("route", ["skills", "toolsets"])
def test_catalog_uses_installed_discovery(route, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))
    skill = tmp_path / "skills" / "catalog-probe"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: catalog-probe\ndescription: Catalog discovery probe.\n---\nProbe.\n"
    )

    async def exercise():
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "catalog-test"}))
        app = web.Application()
        app.router.add_get("/v1/" + route, getattr(adapter, "_handle_" + route))
        async with TestClient(TestServer(app)) as client:
            response = await client.get("/v1/" + route, headers={"Authorization": "Bearer catalog-test"})
            body = await response.json()
            assert response.status == 200, body
            assert body["object"] == "list"
            assert body["data"]
            if route == "skills":
                assert any(row["name"] == "catalog-probe" for row in body["data"])
            else:
                assert all(isinstance(row["tools"], list) for row in body["data"])
                assert all(isinstance(row["configured"], bool) for row in body["data"])

    asyncio.run(exercise())


@pytest.mark.parametrize("route", ["skills", "toolsets"])
def test_catalog_rejects_wrong_bearer(route, tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_RUNTIME_DIR", str(tmp_path / "runtime"))

    async def exercise():
        adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={"key": "catalog-test"}))
        app = web.Application()
        app.router.add_get("/v1/" + route, getattr(adapter, "_handle_" + route))
        async with TestClient(TestServer(app)) as client:
            response = await client.get("/v1/" + route, headers={"Authorization": "Bearer wrong-key"})
            assert response.status == 401

    asyncio.run(exercise())
