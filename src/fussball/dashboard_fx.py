"""Animationen, Effekte und Easter Eggs fürs Dashboard – nur CSS (kein JavaScript, läuft auch in der
iOS-Dateien-Vorschau). Mit „Bewegung reduzieren“ im System ist alles aus.

Easter Eggs (Checkbox + Label, :has() für seitenweite Effekte):
- Titel dreimal antippen: „Dein Tippspiel-Orakel 🔮“
- 🏟️ in der Fußzeile: Flutlicht-Modus (Rasenstreifen, Flutlicht-Kegel)
- Pokal im Turnierbaum antippen: Pokal wird gestemmt
- 0:0-Tipps schlafen 💤; um Halloween 🎃 statt ⚽, im Dezember Schnee
"""

from __future__ import annotations

from datetime import date

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


def stadium_fx() -> str:
    """Kulisse des Flutlicht-Modus: Himmel, Masten mit Lampenfeldern, Lichtkegel, Dunst, Rasen."""
    masts = "".join(f"<i class='mast m{i}'><b></b></i><i class='beam b{i}'></i>" for i in range(1, 5))
    return (f"<div class='stadiumfx' aria-hidden='true'><div class='sky'></div><div class='pitchwrap'><div class='pitch'></div></div><div class='horizon'></div>"
            f"{masts}<div class='haze'></div></div>")


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
.row:hover,.res:hover{background:color-mix(in srgb,var(--accent) 7%,transparent);border-radius:8px}}

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

/* Easter Egg: Flutlicht – Nachthimmel, vier Masten, Lichtkegel, Dunst, Rasen in Perspektive */
@keyframes ignite{0%{opacity:0}8%{opacity:1}14%{opacity:.25}22%{opacity:1}30%{opacity:.6}40%,100%{opacity:1}}
@keyframes beamin{0%,35%{opacity:0}100%{opacity:var(--o,.42)}}
@keyframes breathe{0%,100%{filter:blur(16px) brightness(1)}50%{filter:blur(18px) brightness(1.12)}}
@keyframes drift{from{transform:translateX(-6%)}to{transform:translateX(6%)}}
@keyframes twinkle{0%,100%{opacity:.35}50%{opacity:.9}}
.stadium{cursor:pointer;margin-left:6px;opacity:.55}
.stadiumfx{display:none;position:fixed;inset:0;z-index:-1;pointer-events:none;overflow:hidden}
#egg-stadium:checked~.stadiumfx{display:block}
body:has(#egg-stadium:checked){background:#060a12;--bg:rgba(7,11,19,.78);--surface:rgba(17,24,38,.66);
--line:rgba(255,255,255,.1);--text:#eef2f8;--muted:#9fb0c6;--ink:238 242 248;--accent:#ffd34d;color-scheme:dark}
body:has(#egg-stadium:checked) main{position:relative;z-index:1}
body:has(#egg-stadium:checked) .tabs{background:rgba(6,10,18,.72);-webkit-backdrop-filter:blur(10px);backdrop-filter:blur(10px)}
body:has(#egg-stadium:checked) :is(.det,.chips a,.tabs label,.tie,.cup){-webkit-backdrop-filter:blur(8px);backdrop-filter:blur(8px)}
.stadiumfx .sky{position:absolute;inset:0;background:
radial-gradient(1px 1px at 12% 18%,#fff,transparent),radial-gradient(1px 1px at 33% 9%,#fff,transparent),
radial-gradient(1px 1px at 58% 14%,#fff,transparent),radial-gradient(1px 1px at 81% 6%,#fff,transparent),
radial-gradient(1px 1px at 92% 22%,#fff,transparent),radial-gradient(1.5px 1.5px at 24% 28%,#fff,transparent),
radial-gradient(ellipse 120% 70% at 50% -10%,#1d2b47 0,#0b1220 55%,#060a12 100%);animation:twinkle 5s ease-in-out infinite}
.stadiumfx .pitchwrap{position:absolute;inset:0;perspective:260px;perspective-origin:50% 10%}
.stadiumfx .horizon{position:absolute;inset:0;background:linear-gradient(180deg,#060a12 0,#060a12 38%,rgba(6,10,18,.55) 58%,
rgba(6,10,18,.62) 100%),radial-gradient(ellipse 70% 40% at 50% 100%,transparent,rgba(6,10,18,.5))}
.stadiumfx .pitch{position:absolute;left:-40%;right:-40%;bottom:-8%;height:80%;transform:rotateX(60deg);transform-origin:50% 100%;
background:linear-gradient(180deg,#060a12 0,rgba(6,10,18,.85) 12%,transparent 38%),
radial-gradient(circle at 50% 46%,transparent 0 7.5%,rgba(255,255,255,.55) 7.8% 8.2%,transparent 8.5%),
linear-gradient(90deg,transparent calc(50% - 1px),rgba(255,255,255,.55) calc(50% - 1px) calc(50% + 1px),transparent calc(50% + 1px)),
radial-gradient(ellipse 55% 60% at 50% 50%,rgba(255,250,215,.32),transparent 72%),
repeating-linear-gradient(90deg,#217d38 0 5%,#1b6c2f 5% 10%);filter:saturate(.85) brightness(.7)}
.stadiumfx .mast{position:absolute;z-index:3;top:5vh;width:74px;height:30px;opacity:0;animation:ignite 1.6s ease-out forwards}
.stadiumfx .mast b{position:absolute;inset:0;border-radius:3px;background:#2a2f38;
background-image:radial-gradient(circle,#fffef2 0 2.4px,transparent 3px);background-size:12.3px 10px;
box-shadow:0 0 18px 6px rgba(255,248,214,.85),0 0 70px 26px rgba(255,244,200,.45),0 0 160px 60px rgba(255,240,190,.18)}
.stadiumfx .mast::after{content:"";position:absolute;left:50%;top:30px;width:4px;height:42vh;transform:translateX(-50%);
background:linear-gradient(180deg,#39404c,#141821 70%,transparent)}
.stadiumfx .m1{left:3vw;transform:rotate(14deg)}.stadiumfx .m2{left:22vw;transform:rotate(6deg);top:2vh;animation-delay:.35s}
.stadiumfx .m3{right:22vw;transform:rotate(-6deg);top:2vh;animation-delay:.7s}.stadiumfx .m4{right:3vw;transform:rotate(-14deg);animation-delay:1.05s}
.stadiumfx .beam{position:absolute;z-index:2;top:7vh;width:46vw;height:110vh;transform-origin:50% 0;opacity:0;mix-blend-mode:screen;
filter:blur(16px);animation:beamin 2.6s ease-out forwards,breathe 7s ease-in-out 2.6s infinite}
.stadiumfx .beam::before{content:"";position:absolute;inset:0;clip-path:polygon(46% 0,54% 0,100% 100%,0 100%);
background:linear-gradient(180deg,rgba(255,250,224,.85),rgba(255,248,215,.3) 35%,rgba(255,248,215,.08) 65%,transparent 85%)}
.stadiumfx .b1{left:calc(3vw - 19vw);transform:rotate(-30deg);--o:.38;animation-delay:.2s,2.6s}
.stadiumfx .b2{left:calc(22vw - 19vw);transform:rotate(-12deg);--o:.3;animation-delay:.55s,3s}
.stadiumfx .b3{right:calc(22vw - 19vw);transform:rotate(12deg);--o:.3;animation-delay:.9s,3.4s}
.stadiumfx .b4{right:calc(3vw - 19vw);transform:rotate(30deg);--o:.38;animation-delay:1.25s,3.8s}
.stadiumfx .haze{position:absolute;inset:-10%;background:radial-gradient(ellipse 40% 25% at 30% 40%,rgba(255,255,255,.06),transparent 70%),
radial-gradient(ellipse 45% 30% at 70% 55%,rgba(255,255,255,.05),transparent 70%);animation:drift 24s ease-in-out infinite alternate}

/* Saison */
.snow{position:fixed;inset:0;pointer-events:none;z-index:30}
.snow i{position:absolute;left:var(--x);top:0;color:#cfe3ff;font-style:normal;opacity:.8;animation:snow var(--t) linear var(--d) infinite}

@media (prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition:none!important}
.stadiumfx .mast{opacity:1}.stadiumfx .beam{opacity:.35}
.snow{display:none!important}.title .t1{text-shadow:0 0 10px rgba(124,77,255,.35)}}
"""
