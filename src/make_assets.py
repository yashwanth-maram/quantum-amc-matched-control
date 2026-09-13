"""Generate the README assets. Oscilloscope / signal-analyser vernacular.

Palette, chosen to stay legible on both GitHub light (#ffffff) and dark (#0d1117):
  phosphor  #00C2A8   classical / reference trace
  indigo    #7C6BF5   quantum trace
  amber     #F2A03D   measurement callouts
  slate     #64748B   gridlines, axis labels, body text

Animation is progressive enhancement throughout: every element is fully visible
in its base attributes, and SMIL/CSS only adds motion on top. A viewer that
ignores animation still sees the complete figure.
"""
import math
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "assets"
OUT.mkdir(exist_ok=True)

PHOS, INDIGO, AMBER, SLATE = "#00C2A8", "#7C6BF5", "#F2A03D", "#64748B"
FONT = "ui-sans-serif,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif"


def iq_path(x0, x1, ymid, amp, cycles, phase=0.0, n=340):
    """An amplitude-modulated carrier, the shape an AMC receiver actually sees."""
    pts = []
    for i in range(n + 1):
        t = i / n
        x = x0 + (x1 - x0) * t
        a = amp * (0.45 + 0.55 * abs(math.sin(math.pi * t * 1.5)))
        y = ymid - a * math.sin(2 * math.pi * cycles * t + phase)
        pts.append(f"{x:.1f},{y:.1f}")
    return "M" + "L".join(pts)


# --------------------------------------------------------------------------- #
# 1. Banner. Title block left, instrument readout right, live scope band below.
# --------------------------------------------------------------------------- #
W, H = 1200, 300
bt, bh = 186, 100
bm = bt + bh / 2

grid = "".join(f'<line x1="{x}" y1="{bt}" x2="{x}" y2="{bt+bh}"/>' for x in range(0, W + 1, 48)) + \
       "".join(f'<line x1="0" y1="{y}" x2="{W}" y2="{y}"/>' for y in range(bt, bt + bh + 1, 25))

banner = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" aria-label="Isolating the Quantum Contribution. A matched control for hybrid quantum-classical modulation classification. Transmission coefficient 0.641 for the quantum head against 0.825 for a linear head.">
<defs>
  <linearGradient id="ink" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0%" stop-color="{INDIGO}"/><stop offset="100%" stop-color="{PHOS}"/>
  </linearGradient>
  <linearGradient id="fade" x1="0" y1="0" x2="1" y2="0">
    <stop offset="0%" stop-color="{PHOS}" stop-opacity="0"/>
    <stop offset="22%" stop-color="{PHOS}"/><stop offset="100%" stop-color="{PHOS}"/>
  </linearGradient>
</defs>
<style>
  .grid  {{ stroke: {SLATE}; stroke-opacity: .2; stroke-width: 1; }}
  .trace {{ fill: none; stroke-linecap: round; }}
  .dim   {{ stroke-opacity: .4; stroke-width: 1.7; }}
  .sweep {{ stroke-width: 2.6; stroke-dasharray: 44 956; animation: sweep 6.5s linear infinite; }}
  .sweep2 {{ animation-delay: -.4s; }}
  @keyframes sweep {{ from {{ stroke-dashoffset: 0 }} to {{ stroke-dashoffset: -1000 }} }}
  @media (prefers-reduced-motion: reduce) {{ .sweep {{ animation: none; opacity: 0 }} }}
  .t1 {{ font: 700 40px {FONT}; fill: url(#ink); letter-spacing: -.5px; }}
  .t2 {{ font: 400 19px {FONT}; fill: {SLATE}; }}
  .lb {{ font: 500 11.5px {FONT}; fill: {SLATE}; }}
  .rd {{ font: 700 21px {FONT}; }}
  .rl {{ font: 400 12px {FONT}; fill: {SLATE}; }}
</style>

<text class="t1" x="56" y="72">Isolating the Quantum Contribution</text>
<text class="t2" x="58" y="104">A matched control for hybrid quantum-classical modulation classification</text>
<line x1="58" y1="126" x2="238" y2="126" stroke="{AMBER}" stroke-width="3"/>
<text class="lb" x="58" y="152">WiCOMM-2026, DEAL/DRDO Dehradun, 30-31 October 2026</text>

<g transform="translate(892,44)">
  <rect x="0" y="0" width="252" height="98" rx="8" fill="{SLATE}" fill-opacity=".07" stroke="{SLATE}" stroke-opacity=".28"/>
  <text class="lb" x="18" y="26">signal transmitted to the head</text>
  <text class="rd" x="18" y="58" fill="{PHOS}">0.825</text><text class="rl" x="96" y="58">linear</text>
  <text class="rd" x="18" y="84" fill="{INDIGO}">0.641</text><text class="rl" x="96" y="84">quantum</text>
</g>

<rect x="0" y="{bt}" width="{W}" height="{bh}" fill="{SLATE}" fill-opacity=".035"/>
<g class="grid">{grid}</g>
<line x1="0" y1="{bm:.0f}" x2="{W}" y2="{bm:.0f}" stroke="{SLATE}" stroke-opacity=".38"/>
<path class="trace dim" stroke="{PHOS}"   d="{iq_path(0, W, bm, 32, 11)}"/>
<path class="trace dim" stroke="{INDIGO}" d="{iq_path(0, W, bm, 24, 11, math.pi/2)}"/>
<path class="trace sweep" pathLength="1000" stroke="url(#fade)" d="{iq_path(0, W, bm, 32, 11)}"/>
<path class="trace sweep sweep2" pathLength="1000" stroke="{INDIGO}" d="{iq_path(0, W, bm, 24, 11, math.pi/2)}"/>
<text class="lb" x="56" y="{bt+18}">I and Q, 2 x 128 samples, RadioML 2016.10a</text>
</svg>'''
(OUT / "banner.svg").write_text(banner)


# --------------------------------------------------------------------------- #
# 2. Architecture. The shared aperture, and the one thing that varies.
# --------------------------------------------------------------------------- #
AW, AH = 1100, 310


def box(x, y, w, h, label, sub, color):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{color}" '
            f'fill-opacity=".10" stroke="{color}" stroke-width="1.6"/>'
            f'<text class="bx" x="{x+w/2}" y="{y+h/2-1}" fill="{color}">{label}</text>'
            f'<text class="bs" x="{x+w/2}" y="{y+h/2+17}">{sub}</text>')


def packet(d, delay):
    return (f'<circle r="4.5" fill="{AMBER}" opacity="0">'
            f'<animateMotion dur="2.8s" begin="{delay}s" repeatCount="indefinite" path="{d}"/>'
            f'<animate attributeName="opacity" dur="2.8s" begin="{delay}s" '
            f'repeatCount="indefinite" values="0;1;1;0" keyTimes="0;0.15;0.85;1"/></circle>')


arch = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {AW} {AH}" width="{AW}" height="{AH}" role="img" aria-label="A convolutional extractor is trained once and frozen, producing a six-dimensional embedding shared by a classical MLP, a linear layer and a variational quantum circuit. Only the head varies.">
<style>
  .bx {{ font: 600 15px {FONT}; text-anchor: middle; }}
  .bs {{ font: 400 11.5px {FONT}; fill: {SLATE}; text-anchor: middle; }}
  .ax {{ font: 400 12.5px {FONT}; fill: {SLATE}; }}
  .em {{ font: 600 12.5px {FONT}; fill: {AMBER}; }}
  .wire {{ fill: none; stroke: {SLATE}; stroke-opacity: .5; stroke-width: 1.6; }}
</style>

{box(48, 118, 176, 76, "I/Q input", "2 x 128 samples", SLATE)}
<line class="wire" x1="224" y1="156" x2="322" y2="156"/>
{box(322, 118, 176, 76, "CNN extractor", "trained once, frozen", PHOS)}
<line class="wire" x1="498" y1="156" x2="562" y2="156"/>
{packet("M 224 156 L 322 156", 0)}
{packet("M 498 156 L 562 156", 0.5)}

<rect x="562" y="102" width="46" height="108" rx="6" fill="{AMBER}" fill-opacity=".12" stroke="{AMBER}" stroke-width="1.6"/>
{"".join(f'<circle cx="585" cy="{118+i*16}" r="3.6" fill="{AMBER}"/>' for i in range(6))}
<text class="em" x="612" y="238" text-anchor="end">shared aperture, d = n = 6</text>

<path class="wire" d="M 608 156 C 654 156 654 62 704 62"/>
<path class="wire" d="M 608 156 L 704 156"/>
<path class="wire" d="M 608 156 C 654 156 654 250 704 250"/>

{box(704, 34, 202, 56, "classical MLP", "6 - 64 - 64 - C", PHOS)}
{box(704, 128, 202, 56, "linear", "single affine map", PHOS)}
{box(704, 222, 202, 56, "quantum VQC", "6 qubits, 4 layers", INDIGO)}

<line x1="924" y1="34" x2="924" y2="278" stroke="{AMBER}" stroke-width="2" stroke-opacity=".6"/>
<text class="em" x="936" y="160">only this varies</text>
<text class="ax" x="48" y="292">The embedding is common to every arm, so the head is the only difference left to measure.</text>
</svg>'''
(OUT / "architecture.svg").write_text(arch)


# --------------------------------------------------------------------------- #
# 3. Transmission. Figure 2(b): proportional loss, not additive.
# --------------------------------------------------------------------------- #
KW, KH = 940, 400
ox, oy, span, smax = 96, 340, 248, 0.45
sc = span / smax
pts = [(0.108, 0.070, "10 dB and up"), (0.191, 0.124, "0 to 8 dB"),
       (0.268, 0.172, "-14 to -6 dB"), (0.363, 0.233, "-8 to 0 dB")]

ticks = ""
for v in (0.1, 0.2, 0.3, 0.4):
    X, Y = ox + v * sc, oy - v * sc
    ticks += (f'<line x1="{X:.0f}" y1="{oy}" x2="{X:.0f}" y2="{oy+5}" stroke="{SLATE}" stroke-opacity=".55"/>'
              f'<text class="tk" x="{X:.0f}" y="{oy+20}">{v:.1f}</text>'
              f'<line x1="{ox-5}" y1="{Y:.0f}" x2="{ox}" y2="{Y:.0f}" stroke="{SLATE}" stroke-opacity=".55"/>'
              f'<text class="tky" x="{ox-10}" y="{Y+4:.0f}">{v:.1f}</text>')

dots = ""
for i, (c, q, lab) in enumerate(pts):
    X, Y = ox + c * sc, oy - q * sc
    dots += (f'<g opacity="1"><animate attributeName="opacity" from="0" to="1" dur="0.45s" '
             f'begin="{0.9+i*0.13:.2f}s" fill="freeze"/>'
             f'<circle cx="{X:.0f}" cy="{Y:.0f}" r="5.5" fill="{INDIGO}"/>'
             f'<text class="pl" x="{X+10:.0f}" y="{Y+17:.0f}">{lab}</text></g>')


def ray(slope, color, begin):
    return (f'<line x1="{ox}" y1="{oy}" x2="{ox+span}" y2="{oy-slope*span:.0f}" stroke="{color}" '
            f'stroke-width="2.4" stroke-linecap="round" stroke-dasharray="1000" stroke-dashoffset="0">'
            f'<animate attributeName="stroke-dashoffset" from="1000" to="0" dur="1.3s" '
            f'begin="{begin}s" fill="freeze"/></line>')


kappa = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {KW} {KH}" width="{KW}" height="{KH}" role="img" aria-label="Quantum against matched classical accuracy for four extractors. The points fall on a ray through the origin of slope 0.641, against 0.825 for a linear head, so the loss is proportional rather than additive.">
<style>
  .tk  {{ font: 400 11px {FONT}; fill: {SLATE}; text-anchor: middle; }}
  .tky {{ font: 400 11px {FONT}; fill: {SLATE}; text-anchor: end; }}
  .pl  {{ font: 400 11px {FONT}; fill: {SLATE}; }}
  .ttl {{ font: 600 17px {FONT}; fill: {SLATE}; }}
  .axl {{ font: 500 12px {FONT}; fill: {SLATE}; }}
  .big {{ font: 700 26px {FONT}; }}
  .p   {{ font: 400 13.5px {FONT}; fill: {SLATE}; }}
</style>
<text class="ttl" x="{ox-40}" y="46">Every extractor loses the same fraction, not the same amount</text>

<line x1="{ox}" y1="{oy}" x2="{ox+span+22}" y2="{oy}" stroke="{SLATE}" stroke-opacity=".65"/>
<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{oy-span-22}" stroke="{SLATE}" stroke-opacity=".65"/>
{ticks}
<text class="axl" x="{ox+span/2:.0f}" y="{oy+42}" text-anchor="middle">matched classical accuracy</text>
<text class="axl" x="28" y="{oy-span/2:.0f}" text-anchor="middle" transform="rotate(-90 28 {oy-span/2:.0f})">quantum accuracy</text>

<line x1="{ox}" y1="{oy}" x2="{ox+span}" y2="{oy-span}" stroke="{SLATE}" stroke-opacity=".5" stroke-width="1.4" stroke-dasharray="3 4"/>
<text class="pl" x="{ox+span-52}" y="{oy-span+4}">parity</text>
{ray(0.825, PHOS, 0.15)}
{ray(0.641, INDIGO, 0.0)}
{dots}

<g transform="translate(452,96)">
  <text class="p" x="0" y="0">Four extractors trained on different SNR bands,</text>
  <text class="p" x="0" y="21">all evaluated on one fixed test set.</text>
  <line x1="0" y1="44" x2="52" y2="44" stroke="{AMBER}" stroke-width="3"/>
  <text class="big" x="0" y="92" fill="{PHOS}">0.825</text>
  <text class="p"   x="86" y="92">linear head, +- 0.075</text>
  <text class="big" x="0" y="132" fill="{INDIGO}">0.641</text>
  <text class="p"   x="86" y="132">quantum head, +- 0.032</text>
  <text class="p" x="0" y="178">A constant-ratio model fits these points</text>
  <text class="p" x="0" y="199">8.6 times better than a constant-offset one,</text>
  <text class="p" x="0" y="220">so the low-SNR deficit is proportional loss,</text>
  <text class="p" x="0" y="241">not sensitivity to noise.</text>
</g>
</svg>'''
(OUT / "transmission.svg").write_text(kappa)

print("wrote:", ", ".join(p.name for p in sorted(OUT.glob("*.svg"))))
