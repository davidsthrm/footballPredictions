"""Web-App: Dashboard als Website auf Cloudflare Pages (Home-Bildschirm-App auf dem iPhone).

`build_site` legt unter data/site/ alles ab, was eine installierbare Web-App braucht: Seite,
App-Symbole (180/192/512 px), Manifest, Header (kein Cache, keine Suchmaschinen). `deploy`
lädt den Ordner mit Cloudflares Werkzeug `wrangler` hoch (läuft per npx, keine Projekt-Abhängigkeit).

Zugangsdaten nur in .env (nie committen):
  CLOUDFLARE_API_TOKEN   Token mit Recht „Cloudflare Pages: Bearbeiten“
  CLOUDFLARE_ACCOUNT_ID  Konto-ID (Cloudflare-Dashboard, rechte Spalte)
  FUSSBALL_PAGES_PROJECT Projektname, ergibt https://<name>.pages.dev (Standard: tippspiel)

Zugangsschutz (`setup_access`): Cloudflare Access vor der Seite – Anmeldung per Einmal-Code an die
freigegebene E-Mail-Adresse, Sitzung 30 Tage. Ohne Anmeldung liefert Cloudflare die Seite nicht aus.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import requests

from fussball import config, dashboard

SITE_DIR = config.DATA_DIR / "site"
WRANGLER = "wrangler@4"
HEADERS = """/*
  X-Robots-Tag: noindex, nofollow
  Referrer-Policy: no-referrer
/index.html
  Cache-Control: no-cache
/
  Cache-Control: no-cache
"""


def project() -> str:
    return os.environ.get("FUSSBALL_PAGES_PROJECT", "tippspiel")


def _icon(size: int, out: Path) -> None:
    subprocess.run(["sips", "-z", str(size), str(size), str(dashboard.APP_ICON), "--out", str(out)],
                   check=True, capture_output=True)


def build_site(html: Path = dashboard.OUT_FILE, out: Path = SITE_DIR) -> Path:
    """Website aus dem fertigen Dashboard zusammenstellen."""
    out.mkdir(parents=True, exist_ok=True)
    page = html.read_text(encoding="utf-8").replace(
        "<head>", '<head><link rel="manifest" href="manifest.webmanifest">', 1)
    (out / "index.html").write_text(page, encoding="utf-8")
    for size, name in ((180, "apple-touch-icon.png"), (192, "icon-192.png"), (512, "icon-512.png")):
        _icon(size, out / name)
    (out / "manifest.webmanifest").write_text(json.dumps({
        "name": "Dein Tippspiel-Assistent", "short_name": "Tippspiel", "start_url": "./", "scope": "./",
        "display": "standalone", "background_color": "#000000", "theme_color": "#000000", "lang": "de",
        "icons": [{"src": "icon-192.png", "sizes": "192x192", "type": "image/png"},
                  {"src": "icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"}],
    }, ensure_ascii=False, indent=1))
    (out / "_headers").write_text(HEADERS)
    (out / "robots.txt").write_text("User-agent: *\nDisallow: /\n")
    return out


def _env() -> dict:
    missing = [k for k in ("CLOUDFLARE_API_TOKEN", "CLOUDFLARE_ACCOUNT_ID") if not os.environ.get(k)]
    if missing:
        raise RuntimeError(f"In .env fehlt: {', '.join(missing)}")
    if not shutil.which("npx"):
        raise RuntimeError("npx nicht gefunden (brew install node)")
    return {**os.environ, "WRANGLER_SEND_METRICS": "false"}


def _wrangler(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["npx", "--yes", WRANGLER, *args], env=_env(), capture_output=True, text=True)


def create_project() -> str:
    """Pages-Projekt einmalig über die API anlegen (bestehendes Projekt ist kein Fehler)."""
    names = {p["name"] for p in _api("GET", "/pages/projects")}
    if project() not in names:
        _api("POST", "/pages/projects", json={"name": project(), "production_branch": "main"})
    return f"https://{domain()}"


def deploy(site: Path = SITE_DIR) -> str:
    r = _wrangler("pages", "deploy", str(site), "--project-name", project(), "--branch", "main",
                  "--commit-dirty=true")
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip()[-800:] or r.stdout.strip()[-800:])
    return f"https://{domain()}"


# --- Cloudflare-API: Adresse und Zugangsschutz -------------------------------------------------

API = "https://api.cloudflare.com/client/v4"


def _api(method: str, path: str, **kw) -> dict | list:
    _env()
    r = requests.request(method, f"{API}/accounts/{os.environ['CLOUDFLARE_ACCOUNT_ID']}{path}", timeout=30,
                         headers={"Authorization": f"Bearer {os.environ['CLOUDFLARE_API_TOKEN']}"}, **kw)
    body = r.json()
    if not body.get("success"):
        raise RuntimeError(f"Cloudflare {method} {path}: {body.get('errors')}")
    return body["result"]


def domain() -> str:
    """Tatsächliche pages.dev-Adresse (bei vergebenem Namen hängt Cloudflare ein Kürzel an)."""
    return _api("GET", f"/pages/projects/{project()}")["subdomain"]


def setup_access(email: str) -> str:
    """Seite hinter Cloudflare Access: nur `email`, Anmeldung per Einmal-Code, 30 Tage Sitzung."""
    idps = _api("GET", "/access/identity_providers")
    otp = next((i for i in idps if i["type"] == "onetimepin"), None)
    if otp is None:
        try:
            otp = _api("POST", "/access/identity_providers",
                       json={"name": "Einmal-Code per E-Mail", "type": "onetimepin", "config": {}})
        except RuntimeError:   # Token ohne Recht dafür: Seite trotzdem schützen, Login-Methode im Dashboard aktivieren
            otp = None
    dom = domain()
    policy = {"name": "Nur ich", "decision": "allow", "include": [{"email": {"email": email}}], "precedence": 1}
    app = {"name": "Tippspiel", "type": "self_hosted", "domain": dom, "self_hosted_domains": [dom, f"*.{dom}"],
           "session_duration": "720h", "app_launcher_visible": False, "policies": [policy]}
    if otp:
        app |= {"allowed_idps": [otp["id"]], "auto_redirect_to_identity": True}
    existing = next((a for a in _api("GET", "/access/apps") if a.get("domain") == dom), None)
    if existing:
        _api("PUT", f"/access/apps/{existing['id']}", json=app)
    else:
        _api("POST", "/access/apps", json=app)
    hint = "" if otp else " (Login-Methode „Einmal-PIN“ noch im Zero-Trust-Dashboard aktivieren)"
    return f"https://{dom} – Zugang nur für {email}{hint}"
