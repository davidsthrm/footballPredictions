"""Animationen, Effekte und Easter Eggs fürs Dashboard – nur CSS (kein JavaScript, läuft auch in der
iOS-Dateien-Vorschau). Mit „Bewegung reduzieren“ im System ist alles aus.

Easter Eggs (Checkbox + Label, :has() für seitenweite Effekte):
- ⚽ im Kopf antippen: Schuss ins Netz, „TOOOR!“, Konfetti
- Titel dreimal antippen: „Dein Tippspiel-Orakel 🔮“
- 🏟️ in der Fußzeile: Flutlicht-Modus (Rasenstreifen, Flutlicht-Kegel)
- Pokal im Turnierbaum antippen: Pokal wird gestemmt, Konfetti
- 0:0-Tipps schlafen 💤; um Halloween 🎃 statt ⚽, im Dezember Schnee
"""

from __future__ import annotations

from datetime import date

CONFETTI_COLORS = ["#e63946", "#f4a261", "#2a9d8f", "#457b9d", "#e9c46a", "#8338ec", "#06d6a0", "#ff006e"]


def confetti(n: int = 18) -> str:
    """Konfetti-Teilchen mit festen, gestreuten Startpunkten, Farben und Verzögerungen."""
    return "".join(
        f"<i style='--x:{(i * 37) % 100}%;--d:{(i * 7) % 10 / 20:.2f}s;--c:{CONFETTI_COLORS[i % len(CONFETTI_COLORS)]};"
        f"--r:{(i * 53) % 360}deg'></i>" for i in range(n))


def ball(today: date) -> str:
    return "🎃" if (today.month, today.day) >= (10, 25) and today.month == 10 else "⚽"


def eggs_inputs() -> str:
    """Checkboxen vor dem Kopf (Geschwister-Selektoren brauchen sie davor)."""
    return ("<input type='checkbox' id='egg-o1' class='tg'><input type='checkbox' id='egg-o2' class='tg'>"
            "<input type='checkbox' id='egg-o3' class='tg'><input type='checkbox' id='egg-ball' class='tg'>"
            "<input type='checkbox' id='egg-stadium' class='tg'>")


def title() -> str:
    return ("<h1 class='title'><span class='t1'>Dein Tippspiel-Assistent</span>"
            "<span class='t2'>Dein Tippspiel-Orakel 🔮</span>"
            "<label for='egg-o1' class='o l1'></label><label for='egg-o2' class='o l2'></label>"
            "<label for='egg-o3' class='o l3'></label></h1>")


def ball_label(today: date) -> str:
    return f"<label for='egg-ball' class='kickball' title='Schuss!'>{ball(today)}</label>"


def goal_fx() -> str:
    return (f"<div class='goalfx' aria-hidden='true'><span class='net'>🥅</span><span class='shot'>⚽</span>"
            f"<b>TOOOR!</b><div class='confetti'>{confetti()}</div></div>")


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
@keyframes shot{0%{transform:translate(0,0) rotate(0)}80%{transform:translate(calc(100vw - 120px),-6px) rotate(720deg)}
100%{transform:translate(calc(100vw - 130px),0) rotate(760deg)}}
@keyframes netshake{0%,78%{transform:none}84%{transform:translateX(6px) scale(1.15)}90%{transform:translateX(-4px)}100%{transform:none}}
@keyframes goal{0%,75%{opacity:0;transform:scale(.3)}85%{opacity:1;transform:scale(1.25)}100%{opacity:1;transform:scale(1)}}
@keyframes fall{0%{transform:translate(0,-20px) rotate(0);opacity:1}100%{transform:translate(20px,70vh) rotate(var(--r));opacity:0}}
@keyframes fadeout{0%,85%{opacity:1}100%{opacity:0;visibility:hidden}}
@keyframes glow{0%,100%{text-shadow:0 0 6px #b388ff,0 0 14px #7c4dff}50%{text-shadow:0 0 14px #b388ff,0 0 30px #7c4dff}}
@keyframes lift{0%{transform:translateY(0) rotate(0)}30%{transform:translateY(-18px) rotate(-12deg)}
60%{transform:translateY(-18px) rotate(12deg)}100%{transform:translateY(0) rotate(0)}}
@keyframes snow{from{transform:translateY(-5vh)}to{transform:translateY(105vh)}}
@keyframes sweep{0%,100%{transform:rotate(-18deg)}50%{transform:rotate(18deg)}}
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
.row:hover .crest{transform:rotate(360deg)}.crest{transition:transform .6s cubic-bezier(.3,.7,.3,1)}}

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

/* Easter Egg: Schuss ins Netz */
.kickball{cursor:pointer;display:inline-block;transition:transform .2s}.kickball:active{transform:scale(.8)}
.goalfx{display:none;position:fixed;left:0;right:0;top:70px;height:120px;z-index:50;pointer-events:none;
animation:fadeout 3.4s forwards}
#egg-ball:checked~.top .goalfx{display:block}
.goalfx .shot{position:absolute;left:20px;top:40px;font-size:30px;animation:shot 1.2s cubic-bezier(.3,.6,.4,1) forwards}
.goalfx .net{position:absolute;right:20px;top:26px;font-size:54px;animation:netshake 1.4s forwards}
.goalfx b{position:absolute;left:0;right:0;top:20px;text-align:center;font-size:44px;font-weight:800;letter-spacing:.04em;
color:var(--accent);text-shadow:0 2px 0 var(--bg),0 0 24px var(--accent);animation:goal 1.6s forwards}
.confetti i{position:absolute;left:var(--x);top:0;width:7px;height:12px;background:var(--c);border-radius:2px;opacity:0;
animation:fall 2.2s calc(1s + var(--d)) ease-in forwards}

/* Easter Egg: dreimal auf den Titel */
.title{position:relative}.title .t2{display:none}
.title .o{position:absolute;inset:0;cursor:default}.title .l2,.title .l3{pointer-events:none}
#egg-o1:checked~.top .l1{pointer-events:none}#egg-o1:checked~.top .l2{pointer-events:auto}
#egg-o2:checked~.top .l2{pointer-events:none}#egg-o2:checked~.top .l3{pointer-events:auto}
#egg-o1:checked~#egg-o2:checked~#egg-o3:checked~.top .t1{display:none}
#egg-o1:checked~#egg-o2:checked~#egg-o3:checked~.top .t2{display:inline;color:#b388ff;animation:glow 2s infinite,pop .5s}

/* Easter Egg: Flutlicht */
.stadium{cursor:pointer;margin-left:6px;opacity:.5}
body:has(#egg-stadium:checked){background:repeating-linear-gradient(90deg,#174d24 0 70px,#1b5a2a 70px 140px) fixed;
--bg:#123a1c;--surface:#184826;--line:#2c6b3a;--text:#f2f7ef;--muted:#b9d3b8;--ink:242 247 239}
body:has(#egg-stadium:checked)::before,body:has(#egg-stadium:checked)::after{content:"";position:fixed;top:-40px;
width:60vw;height:120vh;pointer-events:none;z-index:40;transform-origin:top center;
background:linear-gradient(180deg,rgba(255,255,230,.32),rgba(255,255,230,0) 70%);clip-path:polygon(45% 0,55% 0,100% 100%,0 100%);
animation:sweep 6s ease-in-out infinite}
body:has(#egg-stadium:checked)::before{left:-20vw}body:has(#egg-stadium:checked)::after{right:-20vw;animation-delay:-3s}
body:has(#egg-stadium:checked) .tabs{background:transparent}

/* Saison */
.snow{position:fixed;inset:0;pointer-events:none;z-index:30}
.snow i{position:absolute;left:var(--x);top:0;color:#cfe3ff;font-style:normal;opacity:.8;animation:snow var(--t) linear var(--d) infinite}

@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important}
.goalfx,.snow{display:none!important}}
"""
