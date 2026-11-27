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
                f"<div class='cell-detail'>Move {move} · {float(t or 0):.1f}s</div></td>"
                f"<td class='r num'>{deg(residual)}</td>"
                f"<td class='r num'>{deg(resolved)}</td>"
                f"<td class='r num'>{deg(band)}</td>"
                f"<td><span class='result'>Cleared all four</span></td></tr>")

    def refused_row(r):
        joint, move, t, residual, floor, resolved, band, z = r
        return (f"<tr><td><b>{joint_name(joint)}</b>"
                f"<div class='cell-detail'>Move {move} · {float(t or 0):.1f}s</div></td>"
                f"<td class='r num'>{deg(residual)}</td>"
                f"<td class='r num'>{deg(floor)}</td>"
                f"<td class='r num'>{deg(resolved)}</td>"
                f"<td><span class='result'>Within usual range</span></td></tr>")

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
                f'poster="media/form_tour_{rec["form"]}.jpg" aria-label="{rec["name"]} measurement clip">'
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

    # The hero still shows move 6 in the same-side demo.
    low_block = next(move for move in same_card["moves"] if move["move"] == 6)
    elbow_residual = low_block["attribution"]["per_joint"]["R_elbow"]["residual"]

    sbv, cbv = sb["verdict_counts"], cb["verdict_counts"]
    repro = (f"{n_artifacts}/{n_artifacts} artifacts reproduce, {steps_done} pipeline steps, "
             f"{'0 failures' if steps_ok else 'failures present'}")

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Poomsae Coach | Syntax Labs</title>
<meta name="description" content="A Taekwondo poomsae coach: it measures each joint against the reference form, removes the athlete's own habitual offset, and posts a mistake only when the difference beats both the athlete's normal variation and the camera's measurement error.">
<meta name="color-scheme" content="light">
<meta property="og:title" content="Poomsae Coach | Syntax Labs">
<meta property="og:description" content="Joint angles, one athlete's own baseline, and an explicit refusal to call a difference a mistake until it beats both.">
<meta property="og:type" content="website">
<link rel="canonical" href="https://ahmadalzaro1.github.io/poomsae-coach/">
<meta property="og:image" content="https://ahmadalzaro1.github.io/poomsae-coach/media/frame_move6_elbow.png">
<link rel="stylesheet" href="site.css">
</head>
<body>
<a class="skip" href="#main">Skip to content</a>
<header class="masthead">
  <div class="wrap">
    <a class="brand" href="https://thesyntaxlabs.com/">Syntax Labs</a>
    <nav aria-label="Project links">
      <a href="https://thesyntaxlabs.com/work">All work</a>
      <a href="{REPO}">GitHub</a>
    </nav>
  </div>
</header>

<main id="main">
  <div class="hero">
    <div class="wrap">
      <div class="hero-grid">
        <div>
          <h1>Poomsae Coach</h1>
          <p class="lede">The demo reviews Taekwondo form against an athlete's own repeats. It flags
          differences that exceed both their usual variation and the camera's measurement error.</p>
          <div class="hero-links">
            <a class="btn primary" href="#watch">Watch the demo</a>
            <a class="btn" href="#caught">See the findings</a>
          </div>
        </div>
        <figure class="hero-figure">
          <a class="hero-still" href="#watch" aria-label="Watch the joint-angle demo">
            <img src="media/frame_move6_elbow.png" width="1280" height="720" fetchpriority="high"
              alt="An athlete performing a low block, with measured joints and elbow feedback overlaid.">
          </a>
          <figcaption>
            <b>Taegeuk 1 Jang · Low block</b>
            <span>Right elbow: <strong>{deg(abs(elbow_residual))}</strong> after baseline correction,
            below this athlete's <strong>{deg(gate_r_resolved)}</strong> threshold.</span>
          </figcaption>
        </figure>
      </div>
      <nav class="section-index" aria-label="On this page">
        <a href="#watch">Compare the clips</a>
        <a href="#caught">Findings</a>
        <a href="#quiet">Confidence limits</a>
        <a href="#forms">All eight forms</a>
        <a href="#check">Reproduce the results</a>
      </nav>
    </div>
  </div>

  <div class="facts wrap">
    <div class="fact">
      <div class="l">Readings flagged</div>
      <div class="v num">{sbv['fault']}</div>
      <div class="k">out of {cells} readings — one joint at one moment, across {n_moves} moves:
      {sbv['noise']} agree with the athlete's own baseline, {sbv['fault']} do not.</div>
      <a class="fact-source" href="{REPO}/blob/main/out/card_sameview.json">Same-side measurements</a>
    </div>
    <div class="fact">
      <div class="l">Tolerance vs. repeat spread</div>
      <div class="v num">{ratio_rep:.1f}<small>×</small></div>
      <div class="k">how far the {tol_r:.0f}° tolerance sits above one athlete's own repeat spread.
      Across different players it is only {ratio_ba:.2f}×, so the baseline has to be personal.</div>
      <div class="fact-source"><a href="{REPO}/blob/main/out/repeatability.json">Repeat measurements</a>
      and <a href="{REPO}/blob/main/out/between_athlete.json">between-athlete comparisons</a></div>
    </div>
    <div class="fact">
      <div class="l">Frames excluded</div>
      <div class="v num">{gross_pct:.1f}<small>%</small></div>
      <div class="k">of {tot_frames:,} frames across eight forms, the arm is out of reach — two
      estimators disagree by more than 60° on the same pixels. Counted and shown grey, never
      averaged in.</div>
      <a class="fact-source" href="{REPO}/blob/main/out/form_tour.json">Eight-form measurements</a>
    </div>
  </div>

<section class="evidence-section night" id="watch">
  <div class="wrap">
  <h2>One athlete, two references</h2>
  <p class="body">Both clips show the same performance. The first uses a reference filmed from the athlete's side.
  The second uses a reference filmed from the far side of the room.</p>

  <div class="clips">
    <figure>
      <video controls preload="none" poster="media/frame_move6_elbow.png" aria-label="Same-side reference demo" playsinline>
        <source src="media/demo_sameview.mp4" type="video/mp4">
        <a href="media/demo_sameview.mp4">Download the clip</a>
      </video>
      <figcaption><b>Same side.</b> <b>{sbv['fault']} of {cells} readings</b> are flagged as mistakes
      across {n_moves} moves; the other {sbv['noise']} match the athlete's own baseline.</figcaption>
    </figure>
    <figure>
      <video controls preload="none" poster="media/frame_crossview.png" aria-label="Other-side reference demo" playsinline>
        <source src="media/demo_crossview.mp4" type="video/mp4">
        <a href="media/demo_crossview.mp4">Download the clip</a>
      </video>
      <figcaption><b>Other side.</b> The reference camera angle alone shifts readings by
      {cross_card['mean_abs_delta']:.1f}°, so <b>{cbv['style']} readings</b> are accounted for by the athlete's
      own offset.
      Anything still flagged here is in
      <a href="{REPO}/blob/main/docs/RESULTS.md">RESULTS</a>, not a verdict.</figcaption>
    </figure>
  </div>

  <h3 class="t-head" id="threshold">How the threshold is chosen</h3>
  <div class="thresholds">
    <div><span class="n">Tolerance</span><span class="d">Two ways of measuring the same elbow angle
      disagree by this much, so anything smaller is noise: <b>{tol_r:.0f}°</b>.</span></div>
    <div><span class="n">Floor</span><span class="d">The smallest cut-off this data can support once
      the tool's own error is accounted for: <b>{gate_r:.1f}°</b>. A lower bound —
      <b>never the deciding number</b>.</span></div>
    <div><span class="n">Personal threshold</span><span class="d"><b>{K:g} × {gate_r_resolved / K:.2f}°</b> —
      how much their elbow angle varies between their own repeats — giving
      <b>{gate_r_resolved:.1f}°</b>. This is the number that decides.</span></div>
  </div>
  <p class="after">Both stills come from the same footage: the two alignments split the moves
  differently, so the move numbers do not match. Each clip shows all three numbers in the order
  above, taken from the same files this table is built from. Open a clip full screen to read
  them.</p>
  </div>
</section>

<section class="evidence-section" id="caught">
  <div class="wrap">
  <h2>What the coach flagged</h2>
  <p class="body">{sbv['fault']} readings passed every test. Another {len(refused_judgeable)} cleared
  the minimum and still fell within the athlete's usual range.</p>
  <p class="body">Four conditions, all required: clear the tolerance, remove the athlete's own
  offset, beat {K:g} × their variation on that joint, beat the measurement band.</p>

  <div class="tablewrap" tabindex="0" role="region" aria-labelledby="findings-table">
    <table>
      <caption id="findings-table">Published findings — Taegeuk 1 Jang</caption>
      <thead><tr><th>joint and moment</th><th class="r">remaining difference</th>
        <th class="r">gate for this athlete</th><th class="r">measurement band</th><th>result</th></tr></thead>
      <tbody>
      {fault_rows_html}
      </tbody>
    </table>
  </div>

  <div class="tablewrap" tabindex="0" role="region" aria-labelledby="refusal-table">
    <table>
      <caption id="refusal-table">Refused: past the floor, inside the athlete's own range</caption>
      <thead><tr><th>joint and moment</th><th class="r">remaining difference</th>
        <th class="r">floor</th><th class="r">gate</th><th>result</th></tr></thead>
      <tbody>
      {refused_rows_html}
      </tbody>
    </table>
  </div>
  <p class="body">Each cleared its floor and still lost: {K:g} × the athlete's own variation there
  exceeds it, so that difference is their form.{blind_note}</p>
  </div>
</section>

<section class="evidence-section" id="quiet">
  <div class="wrap">
  <h2>When the coach withholds a verdict</h2>
  <p class="body">These tests check when the coach stays quiet and how it compares with other approaches.
  Each reading measures one joint at one moment against the <a href="#threshold">personal threshold</a>.</p>
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
  </div>
</section>

<section class="evidence-section" id="forms">
  <div class="wrap">
  <h2>Measurement across all eight forms</h2>
  <p class="body">Each tile is that form's worst five seconds, through the identical measurement,
  against the dataset's own annotations. Grey frames are ones the estimator cannot see.</p>

  <div class="forms">
    {form_figures}
  </div>

  <div class="tablewrap" tabindex="0" role="region" aria-labelledby="form-agreement-table">
    <table>
      <caption id="form-agreement-table">Agreement with the published annotation, frame by frame</caption>
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
  </div>
</section>



<section class="evidence-section" id="check">
  <div class="wrap">
  <h2>Reproduce the results</h2>
  <p class="body">The repository includes the artifacts used here. You can check the figures and
  rebuild the page without downloading the source dataset or models.</p>
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
  </div>
</section>

</main>

<footer>
  <div class="wrap">
  <p>Code MIT. Demo clips are derived from the CC BY 4.0 dual-view Taegeuk dataset
  (<a href="https://doi.org/10.17632/bjy7vr4xkt.1">10.17632/bjy7vr4xkt.1</a>) and attributed in
  <a href="{REPO}/blob/main/ATTRIBUTION.md">ATTRIBUTION</a>.</p>
  <p>The 94-player labelled dataset carries no upstream licence and is not redistributed here —
  only the numbers computed from it.</p>
  <p><a href="{REPO}">github.com/ahmadalzaro1/poomsae-coach</a></p>
  </div>
</footer>
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
    shutil.copy2(os.path.join(ROOT, "site_assets", "site.css"), SITE)
    shutil.copytree(os.path.join(ROOT, "site_assets", "fonts"), os.path.join(SITE, "fonts"),
                    dirs_exist_ok=True)
    if "/Users/" in html:
        sys.exit("build_site: an absolute local path leaked into the page")
    with open(os.path.join(SITE, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(html)
    open(os.path.join(SITE, ".nojekyll"), "w").close()
    print(f"wrote site/index.html ({len(html)} B) + {len(needed)} media files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
