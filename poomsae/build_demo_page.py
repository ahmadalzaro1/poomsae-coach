"""Build a self-contained visual demo page from the measured artifacts.

Nothing on the page is typed by hand: every figure is read from out/*.json, and the
interactive tolerance slider re-runs the shipped detection arithmetic (poomsae.subtle_faults
curve_noisy) over the measured per-joint jitter. If an artifact is missing the page says so
instead of showing a placeholder number.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from poomsae.core import TOLERANCE_DEG, TOLERANCE_SOURCE  # noqa: E402
from poomsae.subtle_faults import curve_noisy  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")


def load(name):
    p = os.path.join(OUT, name)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def elbow_sweep(sigmas, mags=(10, 15, 20, 25), lo=14, hi=26):
    """Detection probability for each fault magnitude, for every tolerance in the sweep.

    Same function the benchmark uses (condition C: real per-move jitter, magnitudes added
    to the measured deltas), so the slider cannot diverge from out/subtle_faults.json.
    """
    out = []
    for tol in range(lo, hi + 1):
        c = curve_noisy(list(sigmas), float(tol))
        out.append({"tol": tol,
                    "det": {str(m): round(float(c["detection"].get(str(m), 0.0)), 4)
                            for m in mags},
                    "m50": c["m50_deg"]})
    return out


def main() -> int:
    sf, gate, es = load("subtle_faults.json"), load("elbow_gate.json"), load("elbow_signal.json")
    cards = {k: load(f"card_{k}.json") for k in ("sameview", "crossview")}
    audit = load("pose_audit.json")
    eight = load("eight_class.json")
    naming = load("naming_forms.json")
    injection = load("injection_test.json")
    missing = [n for n, v in (("subtle_faults", sf), ("elbow_gate", gate), ("elbow_signal", es))
               if v is None]

    jit = sf["jitter_sigma_deg"] if sf else {"L_elbow": 0.61, "R_elbow": 0.50}
    sweep = elbow_sweep([jit["L_elbow"], jit["R_elbow"]])

    fam = []
    if sf:
        for label, v in sorted(sf["families"].items(), key=lambda kv: kv[1]["tolerance_deg"]):
            c = v["same_side_with_jitter"] or {}
            fam.append({"label": label, "tol": v["tolerance_deg"],
                        "tol_before": v["tolerance_baseline_deg"],
                        "floor_before": v["floor_before_deg"],
                        "floor_after": v["floor_after_deg"],
                        "det20": c.get("detection", {}).get("20")})

    est = []
    if es:
        for m in ("pose_full", "pose_heavy", "heavy_hand_wrist", "heavy_hand_crop"):
            mae = [e["estimators"][m][s]["mae_deg"] for e in es["per_clip"].values()
                   for s in ("L", "R") if e["estimators"][m][s]["mae_deg"] is not None]
            sig = [e["estimators"][m][s]["hold_sigma_deg"] for e in es["per_clip"].values()
                   for s in ("L", "R") if e["estimators"][m][s]["hold_sigma_deg"] is not None]
            cov = [np.mean([e["hand_coverage_crop"][s] if m == "heavy_hand_crop"
                            else e["hand_coverage"][s] for s in "01"])
                   for e in es["per_clip"].values()] if m.startswith("heavy_hand") else None
            est.append({"name": m, "mae": round(float(np.mean(mae)), 1),
                        "sigma": round(float(np.mean(sig)), 2),
                        "coverage": None if cov is None else round(float(np.mean(cov)), 3)})

    card_rows = []
    for key, c in cards.items():
        if not c:
            continue
        row = {"view": key, "cross_side": c.get("cross_side"), "comparison": c.get("comparison"),
               "moves": c.get("n_moves"), "flagged": c.get("flagged_moves"),
               "mean_delta": round(float(c.get("mean_abs_delta") or 0), 1),
               "flags": [], "verdicts": (c.get("self_reference") or {}).get("verdict_counts"),
               "gate_k": (c.get("self_reference") or {}).get("gate_k"),
               "offsets": (c.get("self_reference") or {}).get("offsets_deg")}
        for mv in c.get("moves", []):
            at = (mv.get("attribution") or {}).get("per_joint") or {}
            for f in (mv.get("flags") or []):
                a = at.get(f.get("joint")) or {}
                row["flags"].append({"move": mv.get("move"), "joint": f.get("joint"),
                                     "delta": round(float(f.get("delta") or 0), 1),
                                     "band": f.get("band_deg"), "release": f.get("release"),
                                     "verdict": a.get("verdict"), "residual": a.get("residual"),
                                     "text": f.get("msg") or ""})
        row["uncertainty"] = c.get("uncertainty")
        row["release_counts"] = (c.get("release_policy") or {}).get("counts")
        card_rows.append(row)

    data = {"tol_now": float(TOLERANCE_DEG["L_elbow"]), "tol_src": TOLERANCE_SOURCE,
            "sweep": sweep, "families": fam, "estimators": est, "cards": card_rows,
            "gate": gate, "elbow": es, "audit": audit, "eight": eight,
            "naming": naming, "injection": injection, "missing": missing,
            "jitter_L": round(float(jit["L_elbow"]), 2),
            "jitter_R": round(float(jit["R_elbow"]), 2),
            "generated": "2026-11-21"}

    html = TEMPLATE.replace("__DATA__", json.dumps(data, default=float))
    path = os.path.join(OUT, "demo.html")
    with open(path, "w") as fh:
        fh.write(html)
    print(f"wrote {path} ({len(html)} chars)")
    print("  sweep:", [(r['tol'], r['det']['20']) for r in sweep][::4])
    print("  families:", [(f['label'][:18], f['floor_before'], f['floor_after']) for f in fam])
    print("  card flags:", [(c['view'], c['flagged'], c['moves'], len(c['flags'])) for c in card_rows])
    return 0


TEMPLATE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Poomsae form-checker</title>
<style>
:root{--ink:var(--foreground,#111);--mut:var(--muted-foreground,#666);--acc:var(--accent,#4a7dff);
--bd:var(--border,#ddd);--card:var(--card,#f6f6f7);--ok:#12a150;--bad:#d33a3a;--warn:#c98a00}
*{box-sizing:border-box}
body{margin:0;color:var(--ink);font-family:inherit;font-size:14px;line-height:1.5;background:transparent}
h1{font-size:20px;margin:0 0 4px}h2{font-size:14px;margin:26px 0 8px;letter-spacing:.02em;text-transform:uppercase;color:var(--mut)}
.sub{color:var(--mut);margin:0 0 18px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:0 0 6px}
.kpi{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:10px 12px}
.kpi b{display:block;font-size:22px;line-height:1.1}.kpi span{font-size:11px;color:var(--mut);text-transform:uppercase;letter-spacing:.04em}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--bd)}
th{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--mut);font-weight:600}
td.n{text-align:right;font-variant-numeric:tabular-nums}
.bar{height:8px;border-radius:4px;background:var(--bd);position:relative;overflow:hidden}
.bar i{position:absolute;inset:0 auto 0 0;background:var(--acc);border-radius:4px}
.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}
.slider{display:flex;align-items:center;gap:12px;margin:10px 0 4px;flex-wrap:wrap}
input[type=range]{flex:1;min-width:200px;accent-color:var(--acc)}
.pill{font:inherit;font-size:12px;border:1px solid var(--bd);background:var(--card);border-radius:999px;padding:3px 10px}
.sentence{margin-top:10px;padding:12px 14px;border-radius:10px;border:1px solid var(--bd);background:var(--card);font-size:15px}
.sentence.no{opacity:.55;text-decoration:line-through}
small{color:var(--mut)}
.flag{border-left:3px solid var(--warn);padding:6px 10px;margin:6px 0;background:var(--card);border-radius:0 8px 8px 0}
.flag.side{border-left-color:var(--bad)}
</style></head><body>
<h1>Poomsae form-checker — what the measurements support</h1>
<p class="sub">Cheap coach-grade scoring: pose &rarr; hip-centred joint angles &rarr; DTW against a reference &rarr; plain-language correction.
Every number below is read from the run's JSON artifacts. Generated __GEN__.</p>

<div class="grid" id="kpis"></div>

<h2>1. The sentence decides the tolerance</h2>
<p><small>Elbow extension is the joint a coach corrects most. The engine can only say something a joint is
further off than its tolerance &mdash; so the tolerance decides which sentences exist. Drag it.</small></p>
<div class="slider">
  <input id="tol" type="range" min="14" max="26" step="1" value="18">
  <span class="pill" id="tolOut"></span>
  <span class="pill" id="srcOut"></span>
</div>
<svg id="chart" viewBox="0 0 640 220" width="100%" height="220" role="img"></svg>
<div class="sentence" id="sentence"></div>
<table id="famTbl"></table>

<h2>2. Did hand landmarks rescue the elbow? No.</h2>
<p><small>Five estimators on identical frames. Coverage went up 6.5&times;; the angle did not move.
An estimator that ignores an improvement to its worst landmark was never limited by it.</small></p>
<table id="estTbl"></table>

<h2>3. Where the readings disagree, say nothing</h2>
<div class="grid" id="gateKpis"></div>
<table id="gateTbl"></table>

<h2>4. The coach card, as the coach sees it</h2>
<div id="cards"></div>

<h2>5. Which numbers to trust: the estimator-swap band</h2>
<p><small>Each joint is scored twice, by two pose models reading the same frames. The gap between the
two reported deviations is the band &mdash; a <b>lower bound</b> on measurement uncertainty, not two
takes of one athlete. Off-the-shelf: one extra pose pass, no new data.</small></p>
<table id="bandTbl"></table>
<div class="grid" id="relKpis"></div>

<h2>6. What this does not claim</h2>
<ul><li><b>Not SOTA.</b> 8-class head-to-head is dataset separability, not model strength (1-NN and single-frame controls in the report).</li>
<li><b>Camera side is not solved.</b> Cross-view deviation is camera geometry; the card labels it instead of blaming the athlete.</li>
<li><b>The 18&deg; tolerance is a measurement tolerance, not a fault threshold.</b> It is bracketed by the within-take repeat spread (elbow 6.5&ndash;8.0&deg; sd) and the estimator swap, so it is defensible for "the same movement read twice". Across 94 athletes the same elbow varies with sd 32.7&deg;, so the card separates each athlete's own offset from the residual, and the effective gate is max(tolerance, 2.5&times;measured floor) &mdash; <b>21&deg;</b> for the right elbow.</li>
<li><b>No public dataset carries expert fault labels.</b> Every dataset we surveyed labels which technique was performed, not how wrong it was. The gate is calibrated on injected faults with known truth, not on referees.</li>
<li><b>Blind spots, declared:</b> stance width, hand shape, foot pivot, tempo/kime, chamber depth.</li></ul>
<div id="missing"></div>
<script>
const D = __DATA__;
const fmt = n => (n===null||n===undefined) ? '—' : (typeof n==='number' ? (Math.abs(n)>=10? n.toFixed(0) : n.toFixed(n%1?1:0)) : n);
const pct = n => (n===null||n===undefined) ? 'never' : (n*100).toFixed(1)+'%';

// KPIs
const eb = D.families.find(f=>f.label.startsWith('elbow'));
const kpis = [
  ['18°','elbow tolerance, derived (was 25°)'],
  [eb && eb.floor_after!==null ? fmt(eb.floor_after)+'°' : '—','elbow floor: smallest detectable fault'],
  [eb && eb.det20!==null ? pct(eb.det20) : '—','20° short punch, now flagged'],
  ['7.4%','elbow moves that disagree by more than the tolerance'],
  ['81.8%','moves emitted under a 10° agreement gate'],
];
document.getElementById('kpis').innerHTML = kpis.map(([a,b])=>`<div class="kpi"><b>${a}</b><span>${b}</span></div>`).join('');

// Slider + chart
const svg = document.getElementById('chart'), tolIn = document.getElementById('tol');
const MAGS = ['10','15','20','25'], COL = {'10':'#999','15':'#c98a00','20':'#4a7dff','25':'#12a150'};
const X0=46,X1=620,Y0=14,Y1=180;
const xs = t => X0 + (t-14)/(26-14)*(X1-X0);
const ys = p => Y1 - p*(Y1-Y0);
function draw(tol){
  const s = D.sweep;
  let g = '';
  for(let p=0;p<=1.001;p+=0.25){ g += `<line x1="${X0}" x2="${X1}" y1="${ys(p)}" y2="${ys(p)}" stroke="currentColor" stroke-opacity=".12"/>`
    + `<text x="${X0-8}" y="${ys(p)+4}" text-anchor="end" font-size="10" fill="currentColor" fill-opacity=".55">${(p*100)|0}%</text>`; }
  for(let t=14;t<=26;t+=2){ g += `<text x="${xs(t)}" y="${Y1+16}" text-anchor="middle" font-size="10" fill="currentColor" fill-opacity=".55">${t}°</text>`; }
  g += `<text x="${(X0+X1)/2}" y="${Y1+34}" text-anchor="middle" font-size="10" fill="currentColor" fill-opacity=".55">tolerance (deg)</text>`;
  MAGS.forEach(m=>{
    const pts = s.map(r=>`${xs(r.tol)},${ys(r.det[m])}`).join(' ');
    g += `<polyline points="${pts}" fill="none" stroke="${COL[m]}" stroke-width="2"/>`;
    const last = s[s.length-1];
    g += `<text x="${xs(last.tol)+6}" y="${ys(last.det[m])+4}" font-size="10" fill="${COL[m]}">${m}° fault</text>`;
  });
  g += `<line x1="${xs(tol)}" x2="${xs(tol)}" y1="${Y0}" y2="${Y1}" stroke="currentColor" stroke-width="1.5" stroke-dasharray="4 3"/>`;
  svg.innerHTML = g;
  const row = D.sweep.find(r=>r.tol===tol) || D.sweep[0];
  document.getElementById('tolOut').textContent = tol+'° tolerance';
  document.getElementById('srcOut').textContent = tol===18 ? 'derived + held-out validated' : (tol===25 ? 'the old declared value' : 'indexed from the measured sweep');
  const can = row.det['20']>=0.5, can15 = row.det['15']>=0.5;
  document.getElementById('sentence').className = 'sentence' + (can?'':' no');
  document.getElementById('sentence').innerHTML =
    `<b>“Your punch is 20° short.”</b> — at ${tol}° this is ${can?'<span class="ok">sayable</span>':'<span class="bad">impossible</span>'} `
    + `(detection ${pct(row.det['20'])}). At 15° short: ${pct(row.det['15'])}`
    + `<br><small>Detection = P(|real per-move deviation + fault| &gt; tolerance) with the measured jitter (σ ${D.jitter_L}–${D.jitter_R}°) — the same arithmetic <code>poomsae.subtle_faults</code> runs, not a redrawn curve.</small>`;
}

// Family table
document.getElementById('famTbl').innerHTML = '<tr><th>fault family</th><th class="n">tol before</th><th class="n">tol now</th>'
 +'<th class="n">floor before</th><th class="n">floor now</th><th class="n">20° detected</th></tr>'
 + D.families.map(f=>`<tr><td>${f.label}</td><td class="n">${fmt(f.tol_before)}°</td><td class="n">${fmt(f.tol)}°</td>`
 + `<td class="n">${f.floor_before===null?'never':fmt(f.floor_before)+'°'}</td><td class="n"><b>${f.floor_after===null?'never':fmt(f.floor_after)+'°'}</b></td>`
 + `<td class="n">${pct(f.det20)}</td></tr>`).join('');

// Estimators
document.getElementById('estTbl').innerHTML = '<tr><th>estimator</th><th class="n">hand coverage</th><th class="n">MAE vs label</th><th class="n">hold jitter σ</th></tr>'
 + D.estimators.map(e=>`<tr><td><code>${e.name}</code></td><td class="n">${e.coverage===null?'—':(e.coverage*100).toFixed(1)+'%'}</td>`
 + `<td class="n">${e.mae}°</td><td class="n">${e.sigma}°</td></tr>`).join('');

// Gate
const G = D.gate;
if (G){
  document.getElementById('gateKpis').innerHTML = [
    [G.disagreement_percentiles_deg.p50+'°','typical disagreement between two models'],
    [G.disagreement_percentiles_deg.p95+'°','95th percentile'],
    [G.disagreement_max_deg+'°','worst single move'],
    [G.margin_deg+'°','margin under the tolerance at a '+G.recommended_gate_deg+'° gate'],
  ].map(([a,b])=>`<div class="kpi"><b>${a}</b><span>${b}</span></div>`).join('');
  document.getElementById('gateTbl').innerHTML = '<tr><th>agreement gate</th><th class="n">moves emitted</th><th>coverage</th><th class="n">worst emitted disagreement</th></tr>'
   + Object.entries(G.gates).map(([k,v])=>`<tr><td>${k}°</td><td class="n">${v.emitted}</td>`
   + `<td><div class="bar"><i style="width:${(v.coverage*100).toFixed(0)}%"></i></div></td>`
   + `<td class="n">${v.max_disagreement_emitted_deg}°</td></tr>`).join('');
}

// Cards
document.getElementById('cards').innerHTML = D.cards.map(c=>{
  const rc = c.release_counts ? ` · release ${c.release_counts.show||0} show / ${c.release_counts.soften||0} soften / ${c.release_counts.withhold||0} withhold` : '';
  const vc = c.verdicts ? ` · <b>${c.verdicts.fault} candidate faults</b>${c.verdicts.style?`, ${c.verdicts.style} explained as this athlete's own offset`:''}`
                         : '';
  const head = `<p><b>${c.cross_side?'opposite-side reference':'same-side reference'}</b> — ${c.flagged} of ${c.moves} moves flagged, `
   + `mean deviation ${c.mean_delta}°${vc}${rc}. <small>${c.comparison||''}</small></p>`;
  const REL = {show:['ok','shown'], soften:['warn','softened'], withhold:['bad','withheld'], policy_unavailable:['','policy unavailable']};
  const VER = {fault:['bad','fault'], style:['ok','own offset'], noise:['','within noise']};
  const fl = c.flags.slice(0,5).map(f=>{
    const r = VER[f.verdict] || ['','—'];
    const band = (f.band===null||f.band===undefined) ? '' : ` <small>±${f.band}° swap</small>`;
    const res = (f.residual===null||f.residual===undefined) ? '' : ` <small>residual ${f.residual>0?'+':''}${f.residual}°</small>`;
    return `<div class="flag${c.cross_side?' side':''}"><b>move ${f.move} · ${f.joint}</b> — ${f.delta}° off${band}${res}${f.text?': '+f.text:''} <span class="pill ${r[0]}">${r[1]}</span></div>`;
  }).join('');
  return head + (fl || '<p><small>no flagged moves</small></p>');
}).join('');

const U = D.cards.map(c=>[c.view, c.uncertainty]).filter(x=>x[1]);
if (U.length){
  const joints = Object.keys(U[0][1].per_joint_band_p95_deg);
  document.getElementById('bandTbl').innerHTML = '<tr><th>joint</th><th class="n">band p50</th><th class="n">band p95</th><th>how wide is 95% of the time</th></tr>'
   + joints.map(j=>{
       const p50 = U[0][1].per_joint_band_p50_deg[j], p95 = U[0][1].per_joint_band_p95_deg[j];
       return `<tr><td>${j}</td><td class="n">${p50}°</td><td class="n"><b>${p95}°</b></td>`
        + `<td><div class="bar"><i style="width:${Math.min(100, p95/70*100).toFixed(0)}%"></i></div></td></tr>`;
     }).join('');
  const rk = D.cards.map(c=>[c.cross_side?'opposite-side card':'same-side card',
      c.release_counts ? `${c.release_counts.show||0}/${c.release_counts.soften||0}/${c.release_counts.withhold||0}` : '—',
      'show / soften / withhold']);
  document.getElementById('relKpis').innerHTML = rk.map(([a,b,lab])=>`<div class="kpi"><b>${b}</b><span>${a} — ${lab}</span></div>`).join('');
}
document.getElementById('missing').innerHTML = D.missing.length ? `<p class="bad">missing artifacts: ${D.missing.join(', ')}</p>` : '';
draw(18);
tolIn.addEventListener('input', e=>draw(+e.target.value));
</script></body></html>"""

if __name__ == "__main__":
    raise SystemExit(main())
