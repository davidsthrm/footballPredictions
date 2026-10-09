"""Animationen, Effekte und Easter Eggs fürs Dashboard – nur CSS (kein JavaScript, läuft auch in der
iOS-Dateien-Vorschau). Mit „Bewegung reduzieren“ im System ist alles aus.

Easter Eggs (Checkbox + Label, :has() für seitenweite Effekte):
- Titel dreimal antippen: „Dein Tippspiel-Orakel 🔮“
- 🏟️ in der Fußzeile: Flutlicht-Modus (echtes Stadionfoto, Flutlicht zündet)
- Pokal im Turnierbaum antippen: Pokal wird gestemmt
- 0:0-Tipps schlafen 💤; um Halloween 🎃 statt ⚽, im Dezember Schnee
"""

from __future__ import annotations

import base64
from datetime import date

from fussball import config

def ball(today: date) -> str:
    return "🎃" if (today.month, today.day) >= (10, 25) and today.month == 10 else "⚽"


def eggs_inputs() -> str:
    """Checkboxen vor dem Kopf (Geschwister-Selektoren brauchen sie davor)."""
    return ("<input type='checkbox' id='egg-o1' class='tg'><input type='checkbox' id='egg-o2' class='tg'>"
            "<input type='checkbox' id='egg-o3' class='tg'>"
            "<input type='checkbox' id='egg-stadium' class='tg'>")


def title() -> str:
    return ("<h1 class='title'><span class='t1'>Dein Tippspiel-Assistent</span>"
            "<span class='t2'>Dein Tippspiel-Orakel 🔮</span>"
            "<label for='egg-o1' class='o l1'></label><label for='egg-o2' class='o l2'></label>"
            "<label for='egg-o3' class='o l3'></label></h1>")


def ball_icon(today: date) -> str:
    return f"<span class='kickball'>{ball(today)}</span>"


STADIUM_PHOTO = config.SEED_DIR / "img" / "stadium_night.jpg"   # gemeinfrei, siehe data_seed/img/QUELLEN.md


def stadium_fx() -> str:
    """Kulisse des Flutlicht-Modus: echtes Stadionfoto, abgedunkelt, Flutlicht „zündet“ beim Einschalten."""
    return "<div class='stadiumfx' aria-hidden='true'><div class='photo'></div><div class='shade'></div></div>"


def stadium_css() -> str:
    if not STADIUM_PHOTO.exists():
        return ""
    data = base64.b64encode(STADIUM_PHOTO.read_bytes()).decode()
    return f".stadiumfx .photo{{background-image:url(data:image/jpeg;base64,{data})}}"


def stadium_label() -> str:
    return "<label for='egg-stadium' class='stadium' title='Flutlicht'>🏟️</label>"


def seasonal(today: date) -> str:
    if today.month == 12 and today.day <= 26:
        return "<div class='snow' aria-hidden='true'>" + "".join(
            f"<i style='--x:{(i * 41) % 100}vw;--d:{(i * 3) % 9}s;--t:{8 + i % 6}s'>❄</i>" for i in range(24)) + "</div>"
    return ""


CSS = """
@keyframes fadeUp{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
@keyframes grow{from{transform:scaleX(0)}to{transform:scaleX(1)}}
@keyframes pop{0%{transform:scale(.6);opacity:0}70%{transform:scale(1.12);opacity:1}100%{transform:scale(1)}}
@keyframes flicker{0%,100%{transform:rotate(-6deg) scale(1)}50%{transform:rotate(6deg) scale(1.18)}}
@keyframes shimmer{from{background-position:200% 0}to{background-position:-200% 0}}
@keyframes zz{0%{opacity:0;transform:translate(0,0) scale(.6)}30%{opacity:1}100%{opacity:0;transform:translate(8px,-10px) scale(1)}}
@keyframes titleglow{0%,100%{text-shadow:0 0 8px rgba(124,77,255,.28),0 0 22px rgba(124,77,255,.14)}
50%{text-shadow:0 0 12px rgba(124,77,255,.45),0 0 30px rgba(124,77,255,.25)}}
@keyframes glow{0%,100%{text-shadow:0 0 6px #b388ff,0 0 14px #7c4dff}50%{text-shadow:0 0 14px #b388ff,0 0 30px #7c4dff}}
@keyframes lift{0%{transform:translateY(0) rotate(0)}30%{transform:translateY(-18px) rotate(-12deg)}
60%{transform:translateY(-18px) rotate(12deg)}100%{transform:translateY(0) rotate(0)}}
@keyframes snow{from{transform:translateY(-5vh)}to{transform:translateY(105vh)}}
@keyframes drawin{from{opacity:0;transform:translateX(-10px)}to{opacity:1;transform:none}}

/* Auftritt */
.top .headrow,.top>p{animation:fadeUp .6s both}.top>p{animation-delay:.12s}
.tabs{animation:fadeUp .6s .2s both}
.view{animation:fadeIn .35s ease}
.league .match{animation:fadeUp .45s both}
.league .match:nth-of-type(2){animation-delay:.04s}.league .match:nth-of-type(3){animation-delay:.08s}
.league .match:nth-of-type(4){animation-delay:.12s}.league .match:nth-of-type(5){animation-delay:.16s}
.league .match:nth-of-type(6){animation-delay:.2s}.league .match:nth-of-type(n+7){animation-delay:.24s}
.chips a,.tabs label{transition:transform .15s,background-color .2s,color .2s}
.chips a:active,.tabs label:active{transform:scale(.95)}
@media (hover:hover){.chips a:hover{transform:translateY(-2px)}
.row,.res{transition:background-color .2s}
.row:hover,.res:hover{background:color-mix(in srgb,var(--accent) 6%,transparent)}}

/* Details */
.det{transform-origin:top}
.match>.tg:checked+.row+.det{animation:fadeUp .3s both}
.match>.tg:checked+.row+.det .odds i{animation:grow .8s cubic-bezier(.2,.8,.2,1) both;transform-origin:left}
.match>.tg:checked+.row+.det .odds i:nth-child(2){animation-delay:.1s}
.match>.tg:checked+.row+.det .odds i:nth-child(3){animation-delay:.2s}
.match>.tg:checked+.row .tip{animation:pop .4s both}
.mx .matrix td{animation:fadeIn .5s both}

/* Abzeichen */
.banker i{display:inline-block;font-style:normal;margin-right:2px;animation:flicker 1.4s ease-in-out infinite}
.banker{color:var(--text);font-weight:600}
.row .tags{grid-column:2/5;justify-self:center;font-size:11px;margin-top:-5px;line-height:1}
.derby{font-weight:700;background:linear-gradient(90deg,var(--bad),#e0a32e,#8338ec,var(--bad));background-size:200% 100%;
-webkit-background-clip:text;background-clip:text;color:transparent;animation:shimmer 5s linear infinite}
.tip.zz{position:relative}.tip.zz::after{content:"💤";position:absolute;font-size:10px;top:-4px;right:-2px;
animation:zz 2.4s ease-out infinite}
.pt.p4{display:inline-block;animation:pop .6s both}
.ginfo{margin:0 0 10px;font-size:12px;color:var(--muted)}

/* Easter Egg: dreimal auf den Titel */
.title{position:relative}.title .t2{display:none}
.title .t1{animation:titleglow 4s ease-in-out infinite}
.title .o{position:absolute;inset:0;cursor:default}.title .l2,.title .l3{pointer-events:none}
#egg-o1:checked~.top .l1{pointer-events:none}#egg-o1:checked~.top .l2{pointer-events:auto}
#egg-o2:checked~.top .l2{pointer-events:none}#egg-o2:checked~.top .l3{pointer-events:auto}
#egg-o1:checked~#egg-o2:checked~#egg-o3:checked~.top .t1{display:none}
#egg-o1:checked~#egg-o2:checked~#egg-o3:checked~.top .t2{display:inline;color:#b388ff;animation:glow 2s infinite,pop .5s}

/* Easter Egg: Flutlicht – echtes Stadionfoto (gemeinfrei), abgedunkelt; das Licht zündet wie echtes Flutlicht */
@keyframes ignite{0%{filter:brightness(.05) saturate(.6)}12%{filter:brightness(.5) saturate(.9)}18%{filter:brightness(.12) saturate(.7)}
30%{filter:brightness(.55) saturate(.95)}38%{filter:brightness(.3) saturate(.85)}100%{filter:brightness(.78) saturate(1.05)}}
@keyframes breathe{0%,100%{transform:scale(1.04)}50%{transform:scale(1.07)}}
.stadium{cursor:pointer;margin-left:6px;opacity:.55}
.stadiumfx{display:none;position:fixed;inset:0;z-index:-1;pointer-events:none;overflow:hidden;background:#05070b}
#egg-stadium:checked~.stadiumfx{display:block}
.stadiumfx .photo{position:absolute;inset:-3%;background-position:50% 30%;background-size:cover;filter:brightness(.78) saturate(1.05);
animation:ignite 2.8s ease-out both,breathe 30s ease-in-out 2.8s infinite}
.stadiumfx .shade{position:absolute;inset:0;background:linear-gradient(180deg,rgba(5,7,11,.05) 0%,rgba(5,7,11,.3) 35%,rgba(5,7,11,.72) 75%,rgba(5,7,11,.92) 100%),
radial-gradient(ellipse 90% 60% at 50% 30%,transparent 40%,rgba(5,7,11,.6) 100%)}
body:has(#egg-stadium:checked){background:#05070b;--bg:rgba(8,11,17,.7);--surface:rgba(20,25,34,.62);--fill:rgba(255,255,255,.07);
--line:rgba(255,255,255,.1);--text:#f3f5f9;--muted:#a3adbb;--ink:243 245 249;--accent:#ffd34d;color-scheme:dark}
body:has(#egg-stadium:checked) main{position:relative;z-index:1}
body:has(#egg-stadium:checked) :is(.group,.chips a,.tabs,.tie,.cup,.kpis div,.fact,.stats,.ltable){-webkit-backdrop-filter:blur(14px) saturate(1.2);backdrop-filter:blur(14px) saturate(1.2)}

/* Saison */
.snow{position:fixed;inset:0;pointer-events:none;z-index:30}
.snow i{position:absolute;left:var(--x);top:0;color:#cfe3ff;font-style:normal;opacity:.8;animation:snow var(--t) linear var(--d) infinite}

@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important}
.snow{display:none!important}.title .t1{text-shadow:0 0 10px rgba(124,77,255,.35)}}
"""
