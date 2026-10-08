"""Höflicher HTTP-Client mit Datei-Cache.

- Eigener User-Agent, Mindestabstand pro Host, Backoff bei 429/5xx.
- Jede Antwort landet unter data/raw/<source>/<name>; existiert die Datei, wird
  nicht erneut abgerufen (außer refresh=True, z.B. für die laufende Saison).
- Neben jeder Datei liegt <name>.meta.json mit URL, Zeitstempel und Status.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests

from fussball import config

log = logging.getLogger(__name__)

DEFAULT_MIN_INTERVAL = 2.0  # Sekunden zwischen zwei Requests an denselben Host


class BlockedError(RuntimeError):
    """Server verweigert den Zugriff (403 / Bot-Schutz). Wird nicht umgangen."""


class Fetcher:
    def __init__(self, source: str, min_interval: float = DEFAULT_MIN_INTERVAL,
                 raw_dir: Path | None = None, session: requests.Session | None = None):
        self.source = source
        self.min_interval = min_interval
        self.cache_dir = (raw_dir or config.RAW_DIR) / source
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = config.USER_AGENT
        self._last_request: dict[str, float] = {}
        self.n_requests = 0

    def path(self, name: str) -> Path:
        return self.cache_dir / name

    def cached(self, name: str) -> bool:
        return self.path(name).exists()

    def fetched_at(self, name: str) -> str | None:
        """Abrufzeitpunkt (UTC, ISO) einer gecachten Datei laut .meta.json."""
        meta = self.path(name).with_name(self.path(name).name + ".meta.json")
        return json.loads(meta.read_text())["fetched_at"] if meta.exists() else None

    def get(self, url: str, name: str, *, refresh: bool = False,
            headers: dict[str, str] | None = None, retries: int = 3) -> bytes:
        """Liefert den Inhalt von `url`, aus dem Cache falls vorhanden."""
        target = self.path(name)
        if target.exists() and not refresh:
            return target.read_bytes()

        host = urlparse(url).netloc
        for attempt in range(retries + 1):
            self._throttle(host)
            log.info("GET %s", url)
            try:
                resp = self.session.get(url, headers=headers, timeout=30)
            except (requests.ConnectionError, requests.Timeout) as e:
                if attempt >= retries:
                    raise
                wait = self.min_interval * 2 ** (attempt + 2)
                log.warning("%s: %s – neuer Versuch in %.0fs", url, type(e).__name__, wait)
                time.sleep(wait)
                continue
            self.n_requests += 1
            if resp.status_code == 200:
                break
            if resp.status_code in (401, 403):
                raise BlockedError(f"{resp.status_code} für {url} – Zugriff verweigert, Abbruch.")
            if resp.status_code == 404:
                raise FileNotFoundError(url)
            if resp.status_code in (429, 500, 502, 503, 504) and attempt < retries:
                wait = self.min_interval * 2 ** (attempt + 2)
                log.warning("%s -> %s, warte %.0fs", url, resp.status_code, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()

        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(resp.content)
        tmp.replace(target)
        meta = {
            "url": url,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": resp.status_code,
            "bytes": len(resp.content),
        }
        target.with_name(target.name + ".meta.json").write_text(json.dumps(meta, indent=1))
        return resp.content

    def _throttle(self, host: str) -> None:
        last = self._last_request.get(host)
        if last is not None:
            wait = self.min_interval - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
        self._last_request[host] = time.monotonic()
