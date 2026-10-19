"""Client for SyncPoint (github.com/JayaChandraDadi/syncpoint), the service that knows where a listener is
in each title across their devices."""

from __future__ import annotations

from dataclasses import dataclass

import httpx


@dataclass(frozen=True)
class ListenerPosition:
    position_ms: int
    furthest_ms: int
    device: str | None
    version: int


class SyncPointClient:
    def __init__(self, base_url: str, timeout: float = 2.0, client: httpx.Client | None = None):
        self.client = client or httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def position(self, listener_id: str, title_id: str) -> ListenerPosition | None:
        """The listener's current position, or None if they have not started the title."""
        resp = self.client.get(f"/v1/listeners/{listener_id}/titles/{title_id}/position")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        body = resp.json()
        return ListenerPosition(body["positionMs"], body.get("furthestPositionMs", body["positionMs"]),
                                body.get("lastDeviceId"), body.get("version", 0))
