"""„Liquid Glass“ wie in iOS 26 / macOS Tahoe – nur CSS.

Glas für Bedienelemente und Navigation (schwebende Tab-Leiste, Sprungmarken, Knöpfe), Inhalte
bleiben massiv (Karten). Farben, Schriftgrößen, Trennlinien und Abstände nach Apples Systemwerten,
dahinter ein dezenter, langsam fließender Farbverlauf.
Glas-Rezept: durchscheinender Grund + Weichzeichnung/Sättigung des Hintergrunds + Lichtkante
(::before mit Verlauf, als Rand maskiert) + innerer Glanz oben. Wird nach allen anderen Styles
eingebunden und überschreibt sie.
"""

from __future__ import annotations


def ambient() -> str:
    """Dezenter, langsam fließender Farbverlauf als Hintergrund, Milchglas hinter der Statusleiste und
    weicher Übergang unter der schwebenden Tab-Leiste (iOS „scroll edge“)."""
    return ("<div class='aurora' aria-hidden='true'><i></i><i></i><i></i></div>"
            "<div class='statusbar' aria-hidden='true'></div><div class='edgefade' aria-hidden='true'></div>")


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
.chips a{-webkit-backdrop-filter:none;backdrop-filter:none;background:var(--glass-strong);
box-shadow:var(--glass-hi),0 1px 2px rgba(0,0,0,.08)}   /* kleiner Schatten: der Scrollbereich schneidet große ab */
.chips a{padding:7px 15px 7px 7px;font-weight:600}
.refresh{color:var(--accent);padding:9px 16px;font-weight:600}
.more{padding:8px 16px;border-radius:999px;font-weight:600;margin-top:14px}
.record .more{display:inline-flex}
.tabs label,.chips a,.refresh,.more{transition:transform .2s cubic-bezier(.3,1.4,.5,1),background-color .25s,opacity .2s}
.chips a:active,.refresh:active,.more:active,.tabs label:active{transform:scale(.94)}

/* iOS-Systemwerte: Farben (systemBlue/Green/Red, Label-Grautöne), Typografie, Trennlinien */
:root{--accent:#007aff;--title:#007aff;--good:#34c759;--bad:#ff3b30;--down:#ff3b30;--bg:#f2f2f7;--surface:#fff;
--text:#000;--muted:rgba(60,60,67,.6);--line:rgba(60,60,67,.29);--fill:rgba(118,118,128,.12);--press:#d1d1d6}
@media (prefers-color-scheme:dark){:root{--accent:#0a84ff;--title:#0a84ff;--good:#30d158;--bad:#ff453a;--down:#ff453a;
--bg:#000;--surface:#1c1c1e;--text:#fff;--muted:rgba(235,235,245,.6);--line:rgba(84,84,88,.65);--fill:rgba(118,118,128,.24);
--press:#3a3a3c}}
body{font-size:17px;line-height:1.29;letter-spacing:-.022em}
.top h1{font-size:34px;line-height:41px;font-weight:700;letter-spacing:.012em}
.title .t1{animation:none;text-shadow:0 0 18px rgba(0,122,255,.18)}
.top p{font-size:15px;letter-spacing:-.01em}
.league h2,.plain h2{font-size:22px;line-height:28px;font-weight:700;letter-spacing:.016em}
h3.day{font-size:15px;font-weight:600;letter-spacing:-.01em;margin:22px 16px 7px;color:var(--muted)}
.tm{font-size:17px;font-weight:400}.tm.win{font-weight:600}
.tip,.sc{font-size:17px;font-weight:600;letter-spacing:0}
.ko,.pr{font-size:15px;font-weight:400}
.meta{font-size:13px;font-weight:400;letter-spacing:-.005em}
.det,.stats,.notes li{font-size:15px}
.dt tbody th{font-size:13px;font-weight:400}.abl small,.oddcells div{font-size:12px}
.row{padding:11px 16px;min-height:44px}
.res{padding:11px 16px}
/* Trennlinien wie in iOS-Listen: Haarlinie, links eingerückt */
.group>.match+.match,.group>.resw+.resw,.group>.res+.res,.group>.resw+.res,.group>.res+.resw{border-top:0}
.group>.match+.match::before,.group>.resw+.resw::before,.group>.res+.res::before,.group>.resw+.res::before,
.group>.res+.resw::before{content:"";display:block;height:.5px;margin-left:16px;background:var(--line)}
li.res::before{list-style:none}
/* Antippen: sofortige graue Hervorhebung wie bei UITableView */
.row:active,.res:active{background:var(--press)!important;transition:none}
/* Tab-Leiste: gewähltes Symbol gefüllt (wie SF Symbols) */
#t-spiele:checked~.tabs label[for=t-spiele] svg,#t-eu:checked~.tabs label[for=t-eu] svg,#t-erg:checked~.tabs label[for=t-erg] svg,
#t-tab:checked~.tabs label[for=t-tab] svg,#t-fakt:checked~.tabs label[for=t-fakt] svg{fill:currentColor;fill-opacity:.2}
@media (max-width:560px){.tm{font-size:15px;letter-spacing:-.016em}.row{padding:11px 14px 9px}}

/* Hintergrund: Aurora – drei weiche Farbfelder, sehr langsam in Bewegung, kaum sichtbar */
@keyframes drift1{0%,100%{transform:translate(0,0) scale(1)}50%{transform:translate(12vw,8vh) scale(1.15)}}
@keyframes drift2{0%,100%{transform:translate(0,0) scale(1.1)}50%{transform:translate(-10vw,12vh) scale(.95)}}
@keyframes drift3{0%,100%{transform:translate(0,0) scale(1)}50%{transform:translate(6vw,-10vh) scale(1.2)}}
.aurora{position:fixed;inset:-10%;z-index:-2;pointer-events:none;overflow:hidden}
.aurora i{position:absolute;border-radius:50%;filter:blur(80px);opacity:.30;will-change:transform}
.aurora i:nth-child(1){width:70vmax;height:70vmax;left:-25vmax;top:-30vmax;background:#5b8cff;animation:drift1 38s ease-in-out infinite}
.aurora i:nth-child(2){width:60vmax;height:60vmax;right:-25vmax;top:10vh;background:#a77bff;animation:drift2 46s ease-in-out infinite}
.aurora i:nth-child(3){width:65vmax;height:65vmax;left:-10vmax;bottom:-35vmax;background:#2fd0b5;opacity:.22;animation:drift3 52s ease-in-out infinite}
@media (prefers-color-scheme:dark){.aurora i{opacity:.20}.aurora i:nth-child(1){background:#2747c9}
.aurora i:nth-child(2){background:#5a2aa8}.aurora i:nth-child(3){background:#0e6e62;opacity:.16}}
body:has(#egg-stadium:checked) .aurora{display:none}
/* Statusleiste (Web-App im Vollbild): Inhalt läuft unter Milchglas statt unter die Uhr */
.statusbar{position:fixed;top:0;left:0;right:0;height:env(safe-area-inset-top);z-index:30;pointer-events:none;
background:color-mix(in srgb,var(--bg) 55%,transparent);-webkit-backdrop-filter:blur(20px) saturate(1.8);backdrop-filter:blur(20px) saturate(1.8)}
/* Tabellen: kurze Spaltenköpfe am Handy */
.ls{display:none}
@media (max-width:560px){.league-table .lf{display:none}.league-table .ls{display:inline}
.league-table th:nth-child(n+7){width:38px}.league-table th.prog{width:30px}.league-table th:nth-child(3){width:24px}
.league-table{font-size:13px}.league-table td.pc{font-size:12px}
/* Derby-Hinweis mittig unter der Spielzeile */
.row .tags{grid-column:1/-1;justify-self:center;margin-top:2px}}

/* Inhalte: massive Karten mit iOS-26-Radien */
.group,.stats,.ltable,.fact,.kpis div,.tie,.cup{border-radius:var(--radius)}
.group{box-shadow:0 1px 2px rgba(0,0,0,.04),0 8px 28px rgba(30,40,80,.06)}
.det{border-radius:18px}
.tip,.sc{border-radius:10px}
@media (prefers-color-scheme:dark){.group{box-shadow:none}}
"""
