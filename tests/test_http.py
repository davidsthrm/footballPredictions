"""Fetcher: Bot-Schutz wird erkannt, nicht umgangen und nie gespeichert."""

import pytest
import requests

from fussball.http import BlockedError, Fetcher


class FakeSession(requests.Session):
    def __init__(self, status: int, headers: dict, body: bytes = b""):
        super().__init__()
        self.status, self.resp_headers, self.body, self.calls = status, headers, body, 0

    def get(self, url, **kw):
        self.calls += 1
        r = requests.Response()
        r.status_code, r._content, r.url = self.status, self.body, url
        r.headers.update(self.resp_headers)
        return r


@pytest.mark.parametrize("status,headers", [(202, {"x-amzn-waf-action": "challenge"}), (202, {}), (403, {})])
def test_bot_protection_raises_and_caches_nothing(tmp_path, status, headers):
    s = FakeSession(status, headers)
    f = Fetcher("tm", min_interval=0, raw_dir=tmp_path, session=s)
    with pytest.raises(BlockedError):
        f.get("https://example.org/x", "x.html")
    assert s.calls == 1                       # kein erneuter Versuch
    assert not f.cached("x.html") and not list(tmp_path.rglob("*"))


def test_ok_response_is_cached(tmp_path):
    s = FakeSession(200, {}, b"<html>ok</html>")
    f = Fetcher("tm", min_interval=0, raw_dir=tmp_path, session=s)
    assert f.get("https://example.org/x", "x.html") == b"<html>ok</html>"
    assert f.get("https://example.org/x", "x.html") == b"<html>ok</html>" and s.calls == 1
