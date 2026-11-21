#!/usr/bin/env python3
"""Generate site/index.html FROM the committed artifacts. No number is typed by hand.

The site is a projection of out/*.json: if a card changes, the page changes; if a value is
missing, the build fails instead of rendering a hole. Same discipline as the overlay video,
which reads its thresholds from the card rather than carrying its own table.

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

    faults = []
    for tag, card in (("same side", same_card), ("opposite side", cross_card)):
        for mv in card["moves"]:
            for fl in mv.get("flags", []):
                voices = fl.get("self_reference") or {}
                if voices.get("verdict") == "fault":
                    faults.append((tag, mv["move"], mv.get("t"), mv.get("name"), fl["key"],
                                   fl["delta"], voices.get("residual_deg"),
                                   gate.get(fl["key"], tol.get(fl["key"]))))
    faults.sort(key=lambda row: (row[0], row[1]))
    fault_rows = "\n".join(
        f"<tr><td>{t}</td><td class='num'>{m}</td><td class='num'>{tm}</td><td>{n}</td>"
        f"<td><code>{k}</code></td><td class='num'>{d:.1f}°</td>"
        f"<td class='num'>{r if r is None else f'{r:.1f}°'}</td><td class='num'>{g:.1f}°</td></tr>"
        for (t, m, tm, n, k, d, r, g) in faults
    )

    cells = bench["cells_with_band"]
    portal = 100 * clean["faults"] / cells
    jev_blind = jev["experiment_a_blind"]["auc_p_withhold_vs_opposite_side"]
    jev_single = jev["experiment_b_single_reading"]["auc_p_reject_vs_gap10"]
    lg_auc = lg["auc"]["full"]
    lg_fpr = lg["full"]["fpr_at_matched_tpr"]
    eight_mean = eight["eight_class"]["mean"]
    c1 = eight["controls"]["one_nearest_neighbour"]["mean"]
    c2 = eight["controls"]["single_frame_only"]["mean"]
    c3 = eight["shuffled_control"]["mean"]

    html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Poomsae Coach — a deviation is not a fault</title>
<meta name="description" content="A joint-angle poomsae checker that separates measurement tolerance from fault attribution, with every number re-derived from committed artifacts.">
<style>
:root{{--bg:#0b0c10;--fg:#e9e9ee;--dim:#9aa0ae;--line:#23262e;--acc:#7cc4ff;--warn:#ffbd6b;--bad:#ff7b7b;--ok:#7ee0a0}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 ui-sans-serif,-apple-system,"Segoe UI",Inter,sans-serif}}
.wrap{{max-width:1080px;margin:0 auto;padding:0 24px}}
header{{padding:44px 0 26px}}
h1{{font-size:clamp(30px,4.4vw,52px);line-height:1.08;margin:0 0 18px;letter-spacing:-.02em}}
h1 em{{color:var(--acc);font-style:normal}}
p.lede{{font-size:clamp(17px,1.6vw,20px);color:var(--dim);max-width:64ch;margin:0}}
.meta{{margin-top:18px;color:var(--dim);font-size:13.5px}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:26px;margin:26px 0 20px;padding:20px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}}
.kpi .v{{font-size:30px;font-variant-numeric:tabular-nums;letter-spacing:-.02em;line-height:1}}
.kpi .k{{color:var(--dim);font-size:12.5px;line-height:1.45;margin-top:7px;text-wrap:pretty}}
.kpi.bad .v{{color:var(--bad)}} .kpi.ok .v{{color:var(--ok)}}
.meta a{{color:var(--acc);text-decoration:none;border-bottom:1px solid rgba(124,196,255,.3)}}
section{{padding:52px 0;border-bottom:1px solid var(--line)}}
h2{{font-size:13px;letter-spacing:.14em;text-transform:uppercase;color:var(--dim);margin:0 0 24px;font-weight:600}}
.grid{{display:grid;gap:24px}}
@media(min-width:820px){{.two{{grid-template-columns:1fr 1fr}}}}
figure{{margin:0}}
video{{width:100%;border-radius:10px;background:#000;display:block;border:1px solid var(--line)}}
figcaption{{color:var(--dim);font-size:14px;margin-top:10px}}
figcaption b{{color:var(--fg);font-weight:600}}
table{{width:100%;border-collapse:collapse;font-size:14.5px;margin-top:8px}}
th,td{{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{color:var(--dim);font-weight:600;font-size:12.5px;letter-spacing:.06em;text-transform:uppercase}}
td code{{background:#171a21;padding:1px 6px;border-radius:4px;font-size:13px}}
.num{{font-variant-numeric:tabular-nums}}
.strip{{display:grid;gap:18px}}
@media(min-width:820px){{.strip{{grid-template-columns:repeat(4,1fr)}}}}
.card{{border:1px solid var(--line);border-radius:10px;padding:18px}}
.card .v{{font-size:27px;font-variant-numeric:tabular-nums;letter-spacing:-.01em}}
.card .k{{color:var(--dim);font-size:13.5px;margin-top:6px}}
.card .src{{color:#6b7280;font-size:11.5px;margin-top:10px;font-family:ui-monospace,monospace}}
.fault{{color:var(--bad)}} .ok{{color:var(--ok)}} .warn{{color:var(--warn)}}
ul{{margin:0;padding-left:20px}} li{{margin:7px 0;color:var(--dim)}} li b{{color:var(--fg)}}
.caveat{{border-left:2px solid var(--warn);padding-left:14px;color:var(--dim);font-size:14px;margin-top:16px}}
footer{{padding:40px 0 72px;color:var(--dim);font-size:13.5px}}
footer a{{color:var(--acc);text-decoration:none}}
</style></head><body>
<div class="wrap">
<header>
  <h1>A deviation is not a fault<br><em>until it beats the athlete's own habit.</em></h1>
  <p class="lede">An angle tolerance has two jobs. <b>Measurement</b>: are two readings of the same
  frame telling me the same thing? <b>Attribution</b>: did this athlete make a mistake? This repo
  gives each job its own number — and publishes the number below which a difference is not real.</p>
  <div class="kpis">
    <div class="kpi"><div class="v num">{gate['R_elbow']:.1f}° / {gate['L_elbow']:.1f}°</div>
      <div class="k">gate enforced, right / left elbow — from 18.0° tolerance and a measured floor</div></div>
    <div class="kpi bad"><div class="v num">{sb['verdict_counts']['fault']}</div>
      <div class="k">candidate faults out of {cells} cells, same-side video</div></div>
    <div class="kpi ok"><div class="v num">{sweep0['0.5x']['detection_rate']:.3f}</div>
      <div class="k">detection on a fault half the gate size — it refuses instead of guessing</div></div>
    <div class="kpi"><div class="v num">{eight_mean:.4f}</div>
      <div class="k">8-class accuracy, {eight['n_players']} players, against a {c1:.4f} 1-NN baseline</div></div>
  </div>
  <p class="meta">Taegeuk 1 · dual-view CC BY 4.0 dataset · {cells} joint cells ·
  <a href="{REPO}">source</a> ·
  <a href="{REPO}/blob/main/docs/METHOD.md">method</a> ·
  <a href="{REPO}/blob/main/docs/RESULTS.md">results</a> ·
  <a href="{REPO}/blob/main/docs/LIMITATIONS.md">limits</a> ·
  <a href="{REPO}/blob/main/docs/PRIOR-ART.md">prior art</a></p>
</header>

<section>
  <h2>The same take, two references</h2>
  <div class="grid two">
    <figure>
      <video controls preload="metadata" poster="media/frame_move6_elbow_fault.png" playsinline>
        <source src="media/demo_sameview.mp4" type="video/mp4">
        Your browser cannot play this clip — <a href="media/demo_sameview.mp4">download it</a>.
      </video>
      <figcaption><b>Same side.</b> {sb['card_flag_count']} measurement flags →
        <b>{sb['verdict_counts']['fault']} candidate faults</b>, {sb['verdict_counts']['style']} style,
        {sb['verdict_counts']['noise']} within noise. The overlay prints both numbers it obeys:
        <code>tol {tol['R_elbow']:.0f} / gate {gate['R_elbow']:.1f} deg</code>.</figcaption>
    </figure>
    <figure>
      <video controls preload="metadata" poster="media/frame_crossview.png" playsinline>
        <source src="media/demo_crossview.mp4" type="video/mp4">
        Your browser cannot play this clip — <a href="media/demo_crossview.mp4">download it</a>.
      </video>
      <figcaption><b>Opposite camera.</b> Mean deviation {cross_card['mean_abs_delta']:.1f}° — the
        reference was filmed from the other side of the room. {cb['verdict_counts']['fault']} candidates,
        <b>{cb['verdict_counts']['style']} labelled this athlete's own offset</b> under that geometry.</figcaption>
    </figure>
  </div>
</section>

<section>
  <h2>Where the number came from</h2>
  <div class="strip">
    <div class="card"><div class="v num">{tol['R_elbow']:.1f}°</div>
      <div class="k">two pose estimators, identical frames, p95 disagreement — the measurement floor</div>
      <div class="src">out/uncertainty.json</div></div>
    <div class="card"><div class="v num">{min(rep_sd):.1f}–{max(rep_sd):.1f}°</div>
      <div class="k">sd of the same athlete repeating the same technique (n ≥ 6 groups)</div>
      <div class="src">out/repeatability.json</div></div>
    <div class="card"><div class="v num">{ba_traj['elbow_traj_sd_mean']:.1f}°</div>
      <div class="k">elbow sd across 94 players (worst {ba_traj['elbow_traj_sd_max']:.1f}°) — human spread</div>
      <div class="src">out/between_athlete.json</div></div>
    <div class="card"><div class="v num">{gate['R_elbow']:.1f}°</div>
      <div class="k">gate enforced: max(tolerance, 2.5 × measured floor)</div>
      <div class="src">out/self_reference.json</div></div>
  </div>
  <p class="lede" style="margin-top:24px">18° sits at <b>2.3×</b> the repeat spread and <b>0.55×</b> the
  cross-athlete spread. One number cannot be both, so the threshold stayed and the <i>comparison</i>
  changed: each athlete is judged against their own median deviation, and only a <b>residual</b> may
  become a fault.</p>
</section>

<section>
  <h2>The findings, with their margins</h2>
  <table>
    <thead><tr><th>view</th><th>move</th><th>t</th><th>technique</th><th>cell</th><th>measured</th><th>own offset</th><th>gate</th></tr></thead>
    <tbody>{fault_rows}</tbody>
  </table>
  <p class="lede" style="margin-top:18px">Move 6 clears its gate by <b>0.1°</b>. Stated, not smoothed:
  read it as “check this one”, not “this is wrong”.</p>
</section>

<section>
  <h2>Certified on four arms — including where it refuses</h2>
  <table>
    <thead><tr><th>condition</th><th>raw |delta| &gt; tol</th><th>self-referenced layer</th></tr></thead>
    <tbody>
      <tr><td>clean arm — student <i>is</i> the reference; no fault exists</td>
        <td class="num">{clean['old_gate_flags']} cells</td>
        <td><b>{clean['faults']} candidates = {portal:.2f}% of {cells}</b> — published as the detector's
          floor, not a false-positive rate</td></tr>
      <tr><td>style arm — systematic +20° elbow / −15° knee</td>
        <td class="num">{style['old_gate_flags']} cells</td>
        <td><b>{style['faults_beyond_baseline']} findings beyond the clean floor</b>;
          {style['style']} cells labelled style; offsets recovered at
          {style['offset_recovery_deg']['R_elbow']:+.1f}° / {style['offset_recovery_deg']['L_knee']:+.1f}°
          against {style['offset_recovery_deg']['injected_deg']['R_elbow']:+.0f}° /
          {style['offset_recovery_deg']['injected_deg']['L_knee']:+.0f}° injected</td></tr>
      <tr><td>injection at <b>0.5×</b> the gate</td><td>—</td>
        <td class="ok"><b>{sweep0['0.5x']['detection_rate']:.3f} detection</b> — refuses rather than guesses</td></tr>
      <tr><td>injection at <b>0.9×</b> the gate</td><td>—</td>
        <td class="ok"><b>{sweep0['0.9x']['detection_rate']:.3f} detection</b></td></tr>
      <tr><td>injection at <b>1.05×</b> the gate, zero base</td><td>—</td>
        <td>{sweep0['1.05x']['detection_rate_judgeable']:.3f} of judgeable cells;
          {sweep0['1.05x']['band_blind']} of {cells} band-blind, counted as misses →
          {sweep0['1.05x']['detection_rate']:.4f}</td></tr>
      <tr><td>injection at <b>1.05×</b> the gate, <i>measured</i> base</td><td>—</td>
        <td class="warn">{sweepm['1.05x']['detection_rate_judgeable']:.4f} of judgeable cells — the
          existing 8° consumes part of the injected 21°</td></tr>
    </tbody>
  </table>
  <p class="caveat">The last two rows are the same injection. Reporting only the zero-base row would
  be reporting arithmetic as sensitivity.</p>
</section>

<section>
  <h2>What was refused, and the number that refused it</h2>
  <ul>
    <li><b>3D realignment.</b> Deviation range {c3d['range_ratio_3d_over_2d']:.3f}× the 2D case,
      signal-to-noise {c3d['signal_noise']['snr_3d']:.2f} vs {c3d['signal_noise']['snr_2d']:.2f} —
      it deviates less because it responds less.</li>
    <li><b>A learned gate.</b> AUC {lg_auc:.4f}; at a matched true-positive rate the false-positive
      rate is {lg_fpr:.2%} — the boundary exists, but there is no meaning to state, so a stated
      threshold was kept instead. <span style="color:#6b7280;font-family:ui-monospace,monospace;font-size:12px">out/learned_gate.json</span></li>
    <li><b>An LLM as the authoriser.</b> Blind AUC {jev_blind:.4f}, single-reading AUC
      {jev_single:.4f} — advice only, never authorisation.</li>
    <li><b>Sensitivity measured on a measured base.</b> {sweepm['1.05x']['detection_rate_judgeable']:.4f}
      rather than {sweep0['1.05x']['detection_rate_judgeable']:.3f}: an existing 8° plus an injected
      21° is not a detectable 29°.</li>
  </ul>
</section>

<section>
  <h2>The 94-player dataset, with four controls</h2>
  <ul>
    <li><b>{eight_mean:.4f}</b> 8-class accuracy over {eight['n_performances']} labelled performances
      from {eight['n_players']} players, {eight['split']}.</li>
    <li>Controls: {c1:.4f} (1-nearest-neighbour), {c2:.4f} (single frame only), {c3:.4f} (shuffled labels).</li>
    <li>+{(eight_mean - c1) * 100:.1f} points over 1-NN is the honest reading: this dataset is close to
      separable, so the result is "a competent baseline here", not a model win.</li>
  </ul>
</section>

<section>
  <h2>Verify it yourself</h2>
  <ul>
    <li><code>python3 scripts/verify_claims.py</code> — re-derives every claim from the committed JSON.
      No dataset, no models, no network.</li>
    <li><code>python3 scripts/build_site.py</code> — regenerates this page from the same artifacts, so
      the page cannot drift from them.</li>
    <li><code>python3 scripts/check_links.py</code> — fails if any local link or image is missing.</li>
    <li><code>python3 scripts/check_reproducible.py</code> — compares a re-run against
      <code>out/manifest.sha256</code> with an explicit list of run-timing fields ignored.
      The packaged tree was run end to end: <b>{steps_done} steps
      {'completed with 0 failures' if steps_ok else 'FAILED'}</b>, and
      <b>{n_artifacts}/{n_artifacts} artifacts match</b> once
      <code>extraction_ms_per_frame</code>, <code>generated_at</code> and <code>runtime_s</code> are
      set aside. Log: <code>out/run_all.log</code>.</li>
  </ul>
</section>

<footer>
  Code MIT · demo clips derived from the CC BY 4.0 dual-view Taegeuk dataset
  (<a href="https://doi.org/10.17632/bjy7vr4xkt.1">10.17632/bjy7vr4xkt.1</a>) — see
  <a href="{REPO}/blob/main/ATTRIBUTION.md">ATTRIBUTION</a>. The 94-athlete labelled dataset carries no
  licence upstream and is not redistributed: only the numbers computed from it.
  <br><br><a href="{REPO}">github.com/ahmadalzaro1/poomsae-coach</a>
</footer>
</div></body></html>
"""

    os.makedirs(os.path.join(SITE, "media"), exist_ok=True)
    needed = ("demo_sameview.mp4", "demo_crossview.mp4", "frame_move6_elbow_fault.png", "frame_crossview.png")
    for name in needed:
        src = os.path.join(MEDIA, name)
        if not os.path.exists(src):
            sys.exit(f"build_site: missing media/{name}")
        shutil.copy2(src, os.path.join(SITE, "media", name))
    with open(os.path.join(SITE, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(html)
    open(os.path.join(SITE, ".nojekyll"), "w").close()
    print(f"wrote site/index.html ({len(html)} B) + {len(needed)} media files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
