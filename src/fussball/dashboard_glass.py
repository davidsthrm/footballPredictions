"""„Liquid Glass“ wie in iOS 26 / macOS Tahoe – nur CSS.

Glas für Bedienelemente und Navigation (schwebende Tab-Leiste, Sprungmarken, Knöpfe), Inhalte
bleiben massiv (Karten). Dahinter ein dezentes Umgebungslicht, damit das Glas etwas zu brechen hat.
Glas-Rezept: durchscheinender Grund + Weichzeichnung/Sättigung des Hintergrunds + Lichtkante
(::before mit Verlauf, als Rand maskiert) + innerer Glanz oben. Wird nach allen anderen Styles
eingebunden und überschreibt sie.
"""

from __future__ import annotations


def ambient() -> str:
    return "<div class='ambient' aria-hidden='true'><i></i><i></i><i></i></div><div class='edgefade' aria-hidden='true'></div>"


CSS = """
:root{--radius:24px;
--glass:rgba(255,255,255,.56);--glass-strong:rgba(255,255,255,.72);--glass-lens:rgba(120,120,128,.16);
--glass-rim:rgba(255,255,255,.95);--glass-rim2:rgba(255,255,255,.35);
--glass-shadow:0 12px 34px rgba(30,40,80,.12),0 2px 6px rgba(30,40,80,.06);
--glass-hi:inset 0 1px 0 rgba(255,255,255,.95),inset 0 -1px 1px rgba(255,255,255,.35)}
@media (prefers-color-scheme:dark){:root{--glass:rgba(40,40,44,.48);--glass-strong:rgba(48,48,52,.62);--glass-lens:rgba(255,255,255,.14);
--glass-rim:rgba(255,255,255,.34);--glass-rim2:rgba(255,255,255,.08);
--glass-shadow:0 14px 36px rgba(0,0,0,.55),0 2px 6px rgba(0,0,0,.35);
--glass-hi:inset 0 1px 0 rgba(255,255,255,.22),inset 0 -1px 1px rgba(255,255,255,.05)}}
body{font-family:-apple-system,"SF Pro Text",BlinkMacSystemFont,"Plus Jakarta Sans","Segoe UI",Roboto,sans-serif;
letter-spacing:-.005em}
h1,h2,.top h1{font-family:-apple-system,"SF Pro Display",BlinkMacSystemFont,"Plus Jakarta Sans",sans-serif}
.top h1{font-size:34px;font-weight:800;letter-spacing:-.03em}

/* Umgebungslicht */
.ambient{position:fixed;inset:0;z-index:-2;pointer-events:none;overflow:hidden}
.ambient i{position:absolute;border-radius:50%;filter:blur(70px);opacity:.55}
.ambient i:nth-child(1){width:60vw;height:60vw;left:-15vw;top:-20vw;background:#9cc3ff}
.ambient i:nth-child(2){width:50vw;height:50vw;right:-18vw;top:-8vw;background:#cdb8ff}
.ambient i:nth-child(3){width:55vw;height:55vw;left:20vw;bottom:-30vw;background:#b8ead9;opacity:.4}
@media (prefers-color-scheme:dark){.ambient i{opacity:.32}.ambient i:nth-child(1){background:#1f3d9e}
.ambient i:nth-child(2){background:#4b2390}.ambient i:nth-child(3){background:#0f5b55;opacity:.25}}
main{position:relative;z-index:0}

/* Glas-Grundrezept */
.tabs,.chips a,.refresh,.more{position:relative;background:var(--glass);
-webkit-backdrop-filter:blur(22px) saturate(1.9) brightness(1.03);backdrop-filter:blur(22px) saturate(1.9) brightness(1.03);
box-shadow:var(--glass-hi),var(--glass-shadow);border:0}
.tabs::before,.chips a::before,.refresh::before,.more::before{content:"";position:absolute;inset:0;border-radius:inherit;padding:1px;
pointer-events:none;background:linear-gradient(145deg,var(--glass-rim),var(--glass-rim2) 30%,transparent 55%,var(--glass-rim2) 80%,var(--glass-rim));
-webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);-webkit-mask-composite:xor;
mask:linear-gradient(#000 0 0) content-box exclude,linear-gradient(#000 0 0)}

/* Segmentschalter (Mac) als Glaskapsel mit Linse */
.tabs{border-radius:999px;padding:4px;gap:2px;margin:18px 0 6px}
.tabs label{border-radius:999px;padding:9px 8px;color:var(--text);opacity:.72;font-weight:600}
#t-spiele:checked~.tabs label[for=t-spiele],#t-eu:checked~.tabs label[for=t-eu],#t-erg:checked~.tabs label[for=t-erg],
#t-tab:checked~.tabs label[for=t-tab],#t-fakt:checked~.tabs label[for=t-fakt]{opacity:1;color:var(--accent);
background:var(--glass-lens);box-shadow:var(--glass-hi),0 2px 10px rgba(0,0,0,.08)}

/* Tab-Leiste (iPhone): schwebende Glaskapsel */
@media (max-width:700px){
.tabs{position:fixed;top:auto;left:14px;right:14px;bottom:calc(10px + env(safe-area-inset-bottom));margin:0;
padding:5px;border-radius:32px;border-top:0;background:var(--glass)}
.tabs label{padding:6px 0 5px;border-radius:26px;font-size:10px;font-weight:600;gap:2px;opacity:.8}
.tabs svg{width:24px;height:24px}
#t-spiele:checked~.tabs label[for=t-spiele],#t-eu:checked~.tabs label[for=t-eu],#t-erg:checked~.tabs label[for=t-erg],
#t-tab:checked~.tabs label[for=t-tab],#t-fakt:checked~.tabs label[for=t-fakt]{background:var(--glass-lens);color:var(--accent);
box-shadow:var(--glass-hi)}
main{padding-bottom:calc(104px + env(safe-area-inset-bottom))}
.edgefade{position:fixed;left:0;right:0;bottom:0;height:calc(110px + env(safe-area-inset-bottom));z-index:15;pointer-events:none;
background:linear-gradient(0deg,var(--bg) 18%,color-mix(in srgb,var(--bg) 70%,transparent) 50%,transparent)}}

/* Sprungmarken, Knöpfe – liegen auf dem Seitengrund: Glas-Optik ohne Weichzeichner (vermeidet Artefakte im Scrollbereich) */
.chips{padding:14px 2px 6px}
.chips a{-webkit-backdrop-filter:none;backdrop-filter:none;background:var(--glass-strong)}
.chips a{padding:7px 15px 7px 7px;font-weight:600}
.refresh{color:var(--accent);padding:9px 16px;font-weight:600}
.more{padding:8px 16px;border-radius:999px;font-weight:600;margin-top:14px}
.record .more{display:inline-flex}
.tabs label,.chips a,.refresh,.more{transition:transform .2s cubic-bezier(.3,1.4,.5,1),background-color .25s,opacity .2s}
.chips a:active,.refresh:active,.more:active,.tabs label:active{transform:scale(.94)}

/* Inhalte: massive Karten mit iOS-26-Radien */
.group,.stats,.ltable,.fact,.kpis div,.tie,.cup{border-radius:var(--radius)}
.group{box-shadow:0 1px 2px rgba(0,0,0,.04),0 8px 28px rgba(30,40,80,.06)}
.det{border-radius:18px}
.tip,.sc{border-radius:10px}
@media (prefers-color-scheme:dark){.group{box-shadow:none}}
"""
