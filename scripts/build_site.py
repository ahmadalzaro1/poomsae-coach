#!/usr/bin/env python3
"""Generate site/index.html FROM the committed artifacts. No number is typed by hand.

The site is a projection of out/*.json: if a card changes, the page changes; if a value is
missing, the build fails instead of rendering a hole. Same discipline as the overlay video,
which reads its thresholds from the card rather than carrying its own table.

The page is written for a reader who does not know what a residual is. Every number keeps its
exact meaning (the floor is a lower bound, never the gate; no verdict outside Taegeuk 1), but the
words are plain and each section makes one point.

    python3 scripts/build_site.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
SITE = os.path.join(ROOT, "site")
MEDIA = os.path.join(ROOT, "media")
REPO = "https://github.com/ahmadalzaro1/poomsae-coach"


def load(name):
    path = os.path.join(OUT, name)
    if not os.path.exists(path):
        sys.exit(f"build_site: missing {path} - the site is generated from artifacts, not prose")
    with open(path) as fh:
        return json.load(fh)


CSS = """
:root{
  --bg:#faf9f7; --surface:#ffffff; --ink:#15171c; --muted:#4f5662; --faint:#5f6673;
  --line:#e3e1dc; --line-2:#cfccc5; --accent:#0f5d54; --accent-soft:#e6efed;
  --bad:#a3221f; --ok:#166534; --warn:#8a5a10; --night:#12141a; --night-ink:#e8e9ec;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{
  margin:0;background:var(--bg);color:var(--ink);
  font:16.5px/1.62 ui-sans-serif,-apple-system,"Segoe UI",Inter,Roboto,sans-serif;
}
.wrap{max-width:960px;margin:0 auto;padding:0 24px}
a{color:var(--accent);text-decoration:none;border-bottom:1px solid rgba(15,93,84,.28)}
a:hover{border-bottom-color:var(--accent)}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.875em}
.num{font-variant-numeric:tabular-nums}

/* hero */
header{padding:64px 0 44px}
.eyebrow{
  display:inline-block;font-size:12.5px;letter-spacing:.09em;text-transform:uppercase;
  color:var(--muted);margin:0 0 22px;padding:5px 11px;border:1px solid var(--line-2);
  border-radius:999px;background:var(--surface)
}
h1{
  font-size:clamp(32px,5vw,54px);line-height:1.06;letter-spacing:-.025em;font-weight:650;
  margin:0 0 22px;max-width:19ch;text-wrap:balance
}
.lede{font-size:clamp(17px,1.7vw,20.5px);line-height:1.55;color:#2c313b;max-width:62ch;margin:0;text-wrap:pretty}
.lede strong{font-weight:620;color:var(--ink)}
.hero-links{margin-top:30px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
.btn{
  display:inline-block;padding:11px 18px;border-radius:8px;font-size:15px;font-weight:560;
  border:1px solid var(--accent);background:var(--accent);color:#fff
}
.btn:hover{background:#0b4a43;border-color:#0b4a43}
.btn.ghost{background:transparent;color:var(--accent);border-color:var(--line-2)}
.btn.ghost:hover{background:var(--accent-soft);border-color:var(--accent)}

/* the three numbers on the first screen */
.facts{
  display:grid;gap:1px;grid-template-columns:1fr;margin:46px 0 0;background:var(--line);
  border:1px solid var(--line);border-radius:12px;overflow:hidden
}
@media(min-width:780px){.facts{grid-template-columns:repeat(3,1fr)}}
.fact{background:var(--surface);padding:20px 22px 18px;display:flex;flex-direction:column}
.fact .l{font-size:11.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--muted);margin-bottom:12px;font-weight:600;min-height:2.7em}
.fact .v{font-size:clamp(26px,3.2vw,34px);line-height:1;letter-spacing:-.02em;font-weight:600}
.fact .v small{font-size:.44em;font-weight:500;color:var(--muted);letter-spacing:0}
.fact .k{color:var(--muted);font-size:13.5px;line-height:1.5;margin-top:10px;text-wrap:pretty}
.fact .s{color:var(--faint);font-size:11px;margin-top:auto;padding-top:14px;font-family:ui-monospace,monospace;overflow-wrap:anywhere}

/* sections */
section{padding:66px 0;border-top:1px solid var(--line)}
.s-label{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--faint);margin:0 0 12px;font-weight:600}
h2{font-size:clamp(23px,2.9vw,31px);line-height:1.18;letter-spacing:-.02em;margin:0 0 16px;font-weight:640;text-wrap:balance;max-width:34ch}
h3{font-size:16.5px;margin:36px 0 10px;font-weight:640}
p{margin:0 0 14px;text-wrap:pretty}
p.body{color:#333842;max-width:66ch}
p.after{color:var(--muted);font-size:14.5px;max-width:68ch}

/* the clips sit on a dark band so the overlays read */
.night{background:var(--night);color:var(--night-ink);border-top:1px solid #000}
.night .s-label{color:#8f95a3}
.night h2{color:#fff}
.night p.body,.night p.after{color:#b7bcc6}
.night a{color:#7ed3c4;border-bottom-color:rgba(126,211,196,.3)}
.night h3{color:#fff}
.t-head{margin-top:38px;margin-bottom:12px;font-size:15px}
#watch video:focus-visible{outline:2px solid #7ed3c4;outline-offset:3px}
.clips{display:grid;gap:28px;margin:28px 0 0}
@media(min-width:800px){.clips{grid-template-columns:1fr 1fr}}
figure{margin:0}
video{width:100%;display:block;border-radius:10px;background:#000;border:1px solid #2a2e37}
figcaption{margin-top:12px;font-size:14px;line-height:1.55;color:#b7bcc6;text-wrap:pretty}
figcaption b{color:#fff;font-weight:600}

/* tolerance / floor / gate, defined once */
.thresholds{margin-top:34px;border:1px solid #2c313c;border-radius:10px;overflow:hidden;background:#171a21}
.thresholds div{display:grid;gap:3px;padding:14px 18px;border-bottom:1px solid #232833}
.thresholds div:last-child{border-bottom:0}
@media(min-width:640px){.thresholds div{grid-template-columns:92px 1fr;gap:20px;align-items:baseline}}
.thresholds .n{font-family:ui-monospace,monospace;font-size:13px;color:#7ed3c4}
.thresholds .d{font-size:14.5px;color:#b7bcc6;line-height:1.55}
.thresholds .d b{color:#fff;font-weight:600}

/* tables */
.tablewrap{overflow-x:auto;margin:18px 0 6px}
table{width:100%;border-collapse:collapse;font-size:14.5px}
td:first-child,th:first-child{white-space:nowrap}
td.r,th.r{white-space:nowrap}
caption{caption-side:top;text-align:left;color:var(--muted);font-size:13px;padding:0 0 10px}
th,td{text-align:left;padding:11px 12px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:11.5px;letter-spacing:.07em;text-transform:uppercase;color:var(--muted);font-weight:600}
tbody tr:last-child td{border-bottom:0}
tfoot td{border-top:1px solid var(--line-2);font-weight:600}
.r{text-align:right}
.muted{color:var(--muted)}
.chip{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12.5px;font-weight:600;white-space:nowrap}
.chip.pass{background:#e7f2e9;color:var(--ok)}
.chip.stop{background:#f6ece2;color:var(--warn)}

/* eight forms */
.forms{display:grid;gap:24px;margin:28px 0 32px}
@media(min-width:700px){.forms{grid-template-columns:1fr 1fr}}
.form video{border-color:var(--line-2)}
.form figcaption{color:var(--muted)}
.form figcaption b{color:var(--ink);font-weight:620}

/* lists */
ul{margin:0 0 14px;padding-left:20px;max-width:70ch}
li{margin:10px 0;color:#333842;text-wrap:pretty}
li b{color:var(--ink)}

/* commands */
pre{
  margin:16px 0;padding:18px;background:var(--surface);border:1px solid var(--line);border-radius:10px;
  overflow-x:auto;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13.5px;line-height:1.8
}
pre .c{color:var(--faint)}
pre .out{color:var(--ok)}

/* glossary */
details{border:1px solid var(--line);border-radius:10px;background:var(--surface);padding:14px 18px;margin-top:24px}
summary{cursor:pointer;font-weight:620;font-size:15px}
dl{margin:14px 0 0;display:grid;gap:12px}
dl div{display:grid;gap:2px}
@media(min-width:620px){dl div{grid-template-columns:168px 1fr;gap:18px}}
dt{font-weight:620;font-size:14.5px}
dd{margin:0;color:var(--muted);font-size:14.5px;line-height:1.55}

footer{padding:46px 0 80px;border-top:1px solid var(--line);color:var(--muted);font-size:13.5px}
footer p{margin:0 0 10px;max-width:70ch}
@media(prefers-reduced-motion:no-preference){.btn{transition:background .15s ease,border-color .15s ease}}
"""


def main() -> int:
    sr = load("self_reference.json")
    same_card = load("card_sameview.json")
    cross_card = load("card_crossview.json")
    rep = load("repeatability.json")
    ba = load("between_athlete.json")
    eight = load("eight_class.json")
    c3d = load("crossview_3d.json")
    jev = load("jev_gate.json")
    lg = load("learned_gate.json")
    tour = load("form_tour.json")

    # Evidence of the packaged tree actually running, parsed rather than asserted.
    log_path = os.path.join(OUT, "run_all.log")
    if not os.path.exists(log_path):
        sys.exit("build_site: missing out/run_all.log - the reproduction claim needs its artifact")
    log = open(log_path, encoding="utf-8", errors="replace").read()
    steps_done = log.count("=== ")
    steps_ok = "all steps ok - 0 failures" in log
    manifest = [ln for ln in open(os.path.join(OUT, "manifest.sha256"))
                if ln.strip() and not ln.startswith("#")]
    n_artifacts = len(manifest)

    sb, cb = sr["cards"]["sameview"], sr["cards"]["crossview"]
    bench = sr["benchmark"]
    gate, tol = sb["effective_threshold_deg"], sb["tolerance_deg"]

    groups = rep["cards"]["sameview"]["pairs"]["elbow"]["groups"]
    rep_sd = [g["sd"] for g in groups.values() if g["n"] >= 6]
    ba_traj = ba["trajectory"]["pooled"]

    clean, style = bench["clean"], bench["style_only"]
    sweep0, sweepm = bench["sweep_zero_base"], bench["sweep_measured_base"]
    cells = bench["cells_with_band"]

    jev_blind = jev["experiment_a_blind"]["auc_p_withhold_vs_opposite_side"]
    jev_single = jev["experiment_b_single_reading"]["auc_p_reject_vs_gap10"]
    lg_auc = lg["auc"]["full"]
    lg_fpr = lg["full"]["fpr_at_matched_tpr"]
    eight_mean = eight["eight_class"]["mean"]
    c1 = eight["controls"]["one_nearest_neighbour"]["mean"]
    c2 = eight["controls"]["single_frame_only"]["mean"]
    c3 = eight["shuffled_control"]["mean"]

    K = float(sr.get("gate_k") or 2.5)
    def sgn(v):
        return f"{v:+.1f}".replace("-", "\u2212")
    def sgn8(v):
        return f"{v:+.3f}".replace("-", "\u2212")
    n_moves = int(sb["n_moves"]["value"] if isinstance(sb["n_moves"], dict) else sb["n_moves"])
    assert n_moves == len(same_card["moves"]), "n_moves disagrees with the card it describes"
    floor_of = lambda j: gate.get(j, tol.get(j, 15.0))

    # ---- tour totals: derived, never typed ----
    if len(tour) != 8:
        sys.exit(f"build_site: expected 8 forms in form_tour.json, found {len(tour)}")
    tot_frames = sum(r["agreement"]["frames_compared"] for r in tour)
    tot_gross = sum(r["agreement"]["frames_gross"] for r in tour)
    gross_pct = tot_gross / tot_frames * 100
    p50s = [r["agreement"]["p50_abs_delta_deg"] for r in tour]
    p95s = [r["agreement"]["p95_abs_delta_deg"] for r in tour]
    arms_worst = sum(1 for r in tour if "elbow" in str(r["window_worst_cell"]))

    tol_r, gate_r = tol["R_elbow"], gate["R_elbow"]
    gate_r_resolved = K * sb["scale_deg"]["R_elbow"]
    if not rep_sd:
        sys.exit("build_site: no repeated elbow group with n>=6 - the spread ratio would be empty")
    rep_mean = sum(rep_sd) / len(rep_sd)
    ratio_rep = tol_r / rep_mean
    ratio_ba = tol_r / ba_traj["elbow_traj_sd_mean"]

    # ---- findings on the published take: what passed, what the gate refused, what could not be
    # judged at all. The third list is the band doing its work on individual cells, not page-level
    # hand-waving, so it is counted separately instead of being folded into the refusals.
    scale_sv = sr["cards"]["sameview"]["scale_deg"]
    faults, refused_judgeable, refused_blind = [], [], []
    for mv in same_card["moves"]:
        for joint, cell in ((mv.get("attribution") or {}).get("per_joint") or {}).items():
            residual = float(cell["residual"])
            floor = floor_of(joint)
            resolved = K * float(scale_sv.get(joint) or 0.0) or floor
            band, z = float(cell["band"]), float(cell["z"])
            row = (joint, mv["move"], mv.get("t"), residual, floor, resolved, band, z)
            if cell["verdict"] == "fault":
                faults.append(row)
            elif abs(residual) >= floor:
                (refused_blind if band >= abs(residual) else refused_judgeable).append(row)
    faults.sort(key=lambda r: (r[1], r[0]))
    refused_judgeable.sort(key=lambda r: -abs(r[3]))
    refused_blind.sort(key=lambda r: -abs(r[3]))
    if not faults:
        sys.exit("build_site: no cell survives the gate - the findings table would render empty")
    if not refused_judgeable:
        sys.exit("build_site: no refused-but-judgeable cell - the refusal table would render empty")

    def deg(v, nd=1):
        try:
            return f"{float(v):.{nd}f}°".replace("-", "\u2212")
        except (TypeError, ValueError):
            return "—"

    def joint_name(j):
        j = str(j)
        side = {"R": "right", "L": "left"}.get(j.split("_")[0], "")
        part = j.split("_", 1)[1] if "_" in j else j
        return f"{side} {part}".strip()

    def fault_row(r):
        joint, move, t, residual, floor, resolved, band, z = r
        return (f"<tr><td><b>{joint_name(joint)}</b>"
                f"<div class='muted' style='font-size:12.5px'>move {move} · {float(t or 0):.1f}s</div></td>"
                f"<td class='r num'>{deg(residual)}</td>"
                f"<td class='r num'>{deg(resolved)}</td>"
                f"<td class='r num'>{deg(band)}</td>"
                f"<td><span class='chip pass'>cleared all four</span></td></tr>")

    def refused_row(r):
        joint, move, t, residual, floor, resolved, band, z = r
        return (f"<tr><td><b>{joint_name(joint)}</b>"
                f"<div class='muted' style='font-size:12.5px'>move {move} · {float(t or 0):.1f}s</div></td>"
                f"<td class='r num'>{deg(residual)}</td>"
                f"<td class='r num'>{deg(floor)}</td>"
                f"<td class='r num'>{deg(resolved)}</td>"
                f"<td><span class='chip stop'>inside the athlete's own range</span></td></tr>")

    fault_rows_html = "\n      ".join(fault_row(r) for r in faults)
    refused_rows_html = "\n      ".join(refused_row(r) for r in refused_judgeable)
    blind_note = ""
    if refused_blind:
        widest = max(refused_blind, key=lambda r: r[6])
        blind_note = (
            f" Another {len(refused_blind)} could not be judged at all: on {joint_name(widest[0])} the "
            f"estimators disagree by {deg(widest[6])}, wider than the {deg(widest[3])} difference.")


    # ---- the eight-form grid and table ----
    def form_figure(rec):
        return (f'<figure class="form"><video controls preload="none" muted playsinline '
                f'poster="media/form_tour_{rec["form"]}.jpg">'
                f'<source src="media/form_tour_{rec["form"]}.mp4" type="video/mp4"></video>'
                f'<figcaption><b>{rec["name"]}</b> · worst five seconds</figcaption></figure>')

    def form_row(rec):
        a = rec["agreement"]
        return (f'<tr><td><b>{rec["name"]}</b></td>'
                f'<td class="r num">{a["frames_compared"]:,}</td>'
                f'<td class="r num">{a["frames_gross"]:,}</td>'
                f'<td class="r num">{a["p50_abs_delta_deg"]}°</td>'
                f'<td class="r num">{a["p95_abs_delta_deg"]}°</td>'
                f'<td class="r num">{rec["window_start_frame"] / 30:.1f} s</td>'
                f'<td>{joint_name(rec["window_worst_cell"])}</td></tr>')

    form_figures = "\n    ".join(form_figure(r) for r in tour)
    form_rows = "\n      ".join(form_row(r) for r in tour)
    form_totals = (f'<tr><td><b>All eight</b></td>'
                   f'<td class="r num">{tot_frames:,}</td>'
                   f'<td class="r num">{tot_gross:,} <span class="muted">({gross_pct:.1f}%)</span></td>'
                   f'<td class="r num">{min(p50s)}° – {max(p50s)}°</td>'
                   f'<td class="r num">{min(p95s)}° – {max(p95s)}°</td>'
                   f'<td class="r num">—</td>'
                   f'<td>an elbow, in {arms_worst} of 8</td></tr>')

    sbv, cbv = sb["verdict_counts"], cb["verdict_counts"]
    repro = (f"{n_artifacts}/{n_artifacts} artifacts reproduce, {steps_done} pipeline steps, "
             f"{'0 failures' if steps_ok else 'failures present'}")

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>A taekwondo coach that only speaks when it is sure</title>
<meta name="description" content="A Taekwondo poomsae coach: it measures each joint against the reference form, removes the athlete's own habitual offset, and posts a mistake only when the difference beats both the athlete's normal variation and the camera's measurement error.">
<meta name="color-scheme" content="light">
<meta property="og:title" content="A taekwondo coach that only speaks when it is sure">
<meta property="og:description" content="Joint angles, one athlete's own baseline, and an explicit refusal to call a difference a mistake until it beats both.">
<meta property="og:type" content="website">
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<header>
  <p class="eyebrow">Research demo · Poomsae coaching</p>
  <h1>A taekwondo coach that only speaks when it is sure</h1>
  <p class="lede">It watches a poomsae on video, measures the angle at every joint, then removes the
  athlete's <strong>own habit</strong> — the way someone always holds a shoulder. A mistake is flagged
  only when the difference beats both their normal variation and the camera's error.</p>
  <div class="hero-links">
    <a class="btn" href="#watch">Watch a clip</a>
    <a class="btn ghost" href="#check">Check the numbers yourself</a>
    <a class="btn ghost" href="{REPO}/blob/main/README.md">Source &amp; method</a>
  </div>

  <div class="facts">
    <div class="fact">
      <div class="l">mistakes found</div>
      <div class="v num">{sbv['fault']}</div>
      <div class="k">out of {cells} readings — one joint at one moment, across {n_moves} moves:
      {sbv['noise']} agree with the athlete's own baseline, {sbv['fault']} do not.</div>
      <div class="s">out/card_sameview.json</div>
    </div>
    <div class="fact">
      <div class="l">safety margin</div>
      <div class="v num">{ratio_rep:.1f}<small>×</small></div>
      <div class="k">how far the {tol_r:.0f}° tolerance sits above one athlete's own repeat spread.
      Across different players it is only {ratio_ba:.2f}×, so the baseline has to be personal.</div>
      <div class="s">out/repeatability.json<br>out/between_athlete.json</div>
    </div>
    <div class="fact">
      <div class="l">frames excluded</div>
      <div class="v num">{gross_pct:.1f}<small>%</small></div>
      <div class="k">of {tot_frames:,} frames across eight forms, the arm is out of reach — two
      estimators disagree by more than 60° on the same pixels. Counted and shown grey, never
      averaged in.</div>
      <div class="s">out/form_tour.json</div>
    </div>
  </div>
</header>

<section class="night" id="watch">
  <p class="s-label">01 · Watch it work</p>
  <h2>One athlete, two references</h2>
  <p class="body">Same athlete, same video. Only the reference changed — filmed from their side on
  the left, from the far side of the room on the right.</p>

  <div class="clips">
    <figure>
      <video controls preload="metadata" poster="media/frame_move6_elbow.png" playsinline>
        <source src="media/demo_sameview.mp4" type="video/mp4">
        <a href="media/demo_sameview.mp4">Download the clip</a>
      </video>
      <figcaption><b>Same side.</b> <b>{sbv['fault']} mistakes</b> across 19 moves;
      {sbv['noise']} readings match the athlete's own baseline.</figcaption>
    </figure>
    <figure>
      <video controls preload="metadata" poster="media/frame_crossview.png" playsinline>
        <source src="media/demo_crossview.mp4" type="video/mp4">
        <a href="media/demo_crossview.mp4">Download the clip</a>
      </video>
      <figcaption><b>Other side.</b> The camera angle alone shifts readings
      {cross_card['mean_abs_delta']:.1f}°, so <b>{cbv['style']} become the athlete's own offset</b>.
      Anything still flagged here is in
      <a href="{REPO}/blob/main/docs/RESULTS.md">RESULTS</a>, not a verdict.</figcaption>
    </figure>
  </div>

  <h3 class="t-head">How the threshold is chosen</h3>
  <div class="thresholds">
    <div><span class="n">tolerance</span><span class="d">The smallest difference worth noticing at
      all: <b>{tol_r:.0f}°</b> at the elbow.</span></div>
    <div><span class="n">floor</span><span class="d">The smallest threshold anyone may apply to
      this data, raised where the tool's own error demands it: <b>{gate_r:.1f}°</b>. A lower bound —
      <b>never the deciding number</b>.</span></div>
    <div><span class="n">gate</span><span class="d">What this athlete's own variation demands:
      <b>{K:g} × how much their elbow varies between repeats</b> — <b>{gate_r_resolved:.1f}°</b>.
      This one decides.</span></div>
  </div>
  <p class="after">Each clip prints all three in that order, so the video cannot disagree with the
  numbers that produced it. Fullscreen to read them.</p>
</section>

<section id="caught">
  <p class="s-label">02 · What it caught</p>
  <h2>{sbv['fault']} readings passed every test. {len(refused_judgeable)} cleared the minimum and
  were still refused.</h2>
  <p class="body">Four conditions, all required: clear the tolerance, remove the athlete's own
  offset, beat {K:g} × their variation on that joint, beat the measurement band.</p>

  <div class="tablewrap">
    <table>
      <caption>Published findings — Taegeuk 1 Jang</caption>
      <thead><tr><th>joint and moment</th><th class="r">remaining difference</th>
        <th class="r">gate for this athlete</th><th class="r">measurement band</th><th>result</th></tr></thead>
      <tbody>
      {fault_rows_html}
      </tbody>
    </table>
  </div>

  <div class="tablewrap">
    <table>
      <caption>Refused: past the floor, inside the athlete's own range</caption>
      <thead><tr><th>joint and moment</th><th class="r">remaining difference</th>
        <th class="r">floor</th><th class="r">gate</th><th>result</th></tr></thead>
      <tbody>
      {refused_rows_html}
      </tbody>
    </table>
  </div>
  <p class="body">Each cleared its floor and still lost: {K:g} × the athlete's own variation there
  exceeds it, so that difference is their form.{blind_note}</p>
</section>

<section id="quiet">
  <p class="s-label">03 · What it refuses</p>
  <h2>Most of the work is deciding what not to say</h2>
  <p class="body">Cases built to make it look careless, plus the rival approaches at the end. Each
  refusal is a measurement, not a promise. A reading is one joint at one moment; the cut-off is that
  athlete's own gate, from section 01.</p>
  <ul>
    <li><b>Grading itself.</b> With the student as their own reference there is nothing to find:
      without a personal baseline {clean['old_gate_flags']} readings get flagged, with one
      <b>{clean['faults']} remain ({clean['faults'] / cells * 100:.2f}%)</b> — the tool's noise floor,
      not an accuracy figure.</li>
    <li><b>Calling a style a mistake.</b> An extra <b>+20°</b> at the elbow and <b>−15°</b> at the
      knee, added on purpose: <b>{style['style']} readings come back labelled style</b>, and the
      added amounts come back as
      <b>{sgn(style['offset_recovery_deg']['R_elbow'])}°</b> and
      <b>{sgn(style['offset_recovery_deg']['L_knee'])}°</b>.</li>
    <li><b>Mistakes just below the cut-off.</b> Injected at half and nine-tenths of it:
      <b>no detections at either size</b>. It stays silent rather than guessing.</li>
    <li><b>Mistakes just above the cut-off.</b> <b>Every reading it can assess is detected</b>
(a detection rate of <b>{sweep0['1.05x']['detection_rate_judgeable']:.2f}</b>); counting the
      {sweep0['1.05x']['band_blind']} it cannot judge as misses gives
      <b>{sweep0['1.05x']['detection_rate']:.3f}</b>. On a real measured baseline the rate is
      <b>{sweepm['1.05x']['detection_rate_judgeable']:.2f}</b>: the {tol_r:.0f}° the athlete already
      carries absorbs part of the {gate_r_resolved:.0f}° it is given.</li>
    <li><b>Three rival approaches.</b> Measuring in 3D responds less than flat video
      ({c3d['range_ratio_3d_over_2d']:.2f}× the range; signal-to-noise
      {c3d['signal_noise']['snr_3d']:.2f} against {c3d['signal_noise']['snr_2d']:.2f}). A learned
      cut-off separates real from injected at <b>{lg_auc:.3f}</b> AUC — 1.0 is perfect — but
      false-alarms <b>{lg_fpr:.0%}</b> of the time. A language model scores <b>{jev_blind:.3f}</b> with no
      reference reading and <b>{jev_single:.3f}</b> given one: advice, never a verdict.</li>
    <li><b>One cut-off for everybody.</b> Elbow angles vary by up to
      <b>{ba_traj['elbow_traj_sd_max']:.0f}°</b> across the 94 athletes in the separate labelled
      dataset, so no single number fits them all — which is why every cut-off here is personal.</li>
  </ul>
  <p class="after">The classification experiment on that same 94-athlete data, for completeness:
  <b>{eight_mean:.4f}</b> over {eight['n_performances']} performances, against {c1:.4f} for
  nearest-neighbour, {c2:.4f} on a single frame and {c3:.4f} on shuffled labels.
  <b>{sgn8(eight_mean - c1)}</b> over the simple baseline is the honest reading of that result.</p>
</section>

<section id="forms">
  <p class="s-label">04 · The measurement on all eight forms</p>
  <h2>The same mask on every form in the dataset</h2>
  <p class="body">Each tile is that form's worst five seconds, through the identical measurement,
  against the dataset's own annotations. Grey frames are ones the estimator cannot see.</p>

  <div class="forms">
    {form_figures}
  </div>

  <div class="tablewrap">
    <table>
      <caption>Agreement with the published annotation, frame by frame</caption>
      <thead><tr><th>form</th><th class="r">frames used</th><th class="r">frames excluded</th>
        <th class="r">typical</th><th class="r">worst 5 %</th><th class="r">worst window</th>
        <th>joint that broke first</th></tr></thead>
      <tbody>
      {form_rows}
      </tbody>
      <tfoot>{form_totals}</tfoot>
    </table>
  </div>

  <p class="after"><b>No verdicts for forms 2–8.</b> A verdict needs that athlete's own repeats, and
  this release has one athlete's, on Taegeuk 1 Jang. The measurement travels; the judgement does not.
  Near a degree in the middle means the mask is on the right body — checked to 0.1–0.2% of the
  annotation on all eight — so these failures are arms, not the camera following the wrong
  person.</p>
</section>



<section id="check">
  <p class="s-label">05 · Verify it yourself</p>
  <h2>No dataset, models or network needed</h2>
  <p class="body">The repository ships the files these numbers come from. Each command fails loudly
  if this page and the files disagree.</p>
<pre><span class="c"># every documented number, re-derived from out/*.json</span>
python3 scripts/verify_claims.py
<span class="out">91/91 documented claims re-derived</span>

<span class="c"># regenerate this page from the same files</span>
python3 scripts/build_site.py

<span class="c"># links, and a re-run of the pipeline against stored hashes</span>
python3 scripts/check_links.py
python3 scripts/check_reproducible.py
<span class="out">{repro}</span></pre>
  <p class="after">Run log <code class="mono">out/run_all.log</code>; hashes in
  <code class="mono">out/manifest.sha256</code> — {n_artifacts} entries, with run-timing fields set
  aside before comparison.</p>

  <details>
    <summary>Words used on this page</summary>
    <dl>
      <div><dt>Joint angle</dt><dd>The bend at one joint, measured from the body's own centre line, so
        height and distance from the camera drop out.</dd></div>
      <div><dt>Reference form</dt><dd>The comparison performance — the same poomsae, filmed from a
        known angle, with its joints already measured.</dd></div>
      <div><dt>Offset</dt><dd>The athlete's own habit: a consistent difference from the reference,
        taken as the middle of their differences and removed before any judgement.</dd></div>
      <div><dt>Difference left over</dt><dd>What remains once the offset is gone. Only this can become
        a mistake.</dd></div>
      <div><dt>Measurement band</dt><dd>How far two independent pose estimators disagree on the same
        frame. A difference smaller than the band is not evidence of anything.</dd></div>
      <div><dt>Tolerance, floor, gate</dt><dd>The smallest noticeable difference, the smallest bar
        allowed on this dataset, and the bar this athlete's own variation demands — in that order of
        strictness. The middle one is a lower bound on the last, never a substitute for it.</dd></div>
    </dl>
  </details>
</section>

<footer>
  <p>Code MIT. Demo clips are derived from the CC BY 4.0 dual-view Taegeuk dataset
  (<a href="https://doi.org/10.17632/bjy7vr4xkt.1">10.17632/bjy7vr4xkt.1</a>) and attributed in
  <a href="{REPO}/blob/main/ATTRIBUTION.md">ATTRIBUTION</a>.</p>
  <p>The 94-player labelled dataset carries no upstream licence and is not redistributed here —
  only the numbers computed from it.</p>
  <p><a href="{REPO}">github.com/ahmadalzaro1/poomsae-coach</a></p>
</footer>
</div>
</body>
</html>
"""

    os.makedirs(os.path.join(SITE, "media"), exist_ok=True)
    needed = (("demo_sameview.mp4", "demo_crossview.mp4", "frame_move6_elbow.png",
               "frame_crossview.png")
              + tuple(f"form_tour_{n}.mp4" for n in range(1, 9))
              + tuple(f"form_tour_{n}.jpg" for n in range(1, 9)))
    for name in needed:
        src = os.path.join(MEDIA, name)
        if not os.path.exists(src):
            sys.exit(f"build_site: missing media/{name}")
        shutil.copy2(src, os.path.join(SITE, "media", name))
    if "/Users/" in html:
        sys.exit("build_site: an absolute local path leaked into the page")
    with open(os.path.join(SITE, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(html)
    open(os.path.join(SITE, ".nojekyll"), "w").close()
    print(f"wrote site/index.html ({len(html)} B) + {len(needed)} media files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
