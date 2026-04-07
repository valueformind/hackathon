"""Minimal HTTP client for local interaction with the environment server."""

from __future__ import annotations

from typing import Any

import requests


class OpenEnvHTTPClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def reset(self) -> dict[str, Any]:
        response = requests.post(f"{self.base_url}/reset", json={}, timeout=30)
        response.raise_for_status()
        return response.json()

    def step(self, action_type: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        response = requests.post(
            f"{self.base_url}/step",
            json={"action_type": action_type, "payload": payload or {}},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
