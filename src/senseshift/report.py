"""Generate a self-contained, offline research page from measured result JSON."""

import json
from statistics import mean
from pathlib import Path

from .data import ACTIVITIES


def render(results_path: Path, out: Path):
    results = json.loads(Path(results_path).read_text())
    if not results["runs"]:
        raise ValueError("No completed experiment runs")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    payload = json.dumps(results).replace("<", "\\u003c")
    names = json.dumps(ACTIVITIES)
    page = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="A measured study of activity recognition for new people under phone-to-watch sensor shift.">
<title>SenseShift / Research notes</title><style>
:root{--paper:#f6f0e4;--white:#fffdf8;--ink:#202840;--dim:#5d6577;--line:#d8d4cb;--blue:#263ec6;--red:#d9644e;--lime:#c7df7a}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 system-ui,-apple-system,Segoe UI,sans-serif}
.wrap{max-width:1180px;margin:auto;padding:0 24px}header,footer{background:var(--ink);color:white}nav{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;padding:18px 0;gap:12px}
.brand{font-weight:900;font-size:1.55rem;letter-spacing:-.07em}.brand i{color:var(--lime);font-style:normal}nav a{color:#e1e7fb;text-decoration:none;margin-left:17px;font-size:.9rem}
.hero{display:grid;grid-template-columns:1.3fr .7fr;gap:35px;align-items:center;padding:75px 0}.eyebrow,.kicker{font-size:.78rem;text-transform:uppercase;letter-spacing:.15em;font-weight:800}.eyebrow{color:var(--lime)}.kicker{color:var(--blue)}
h1,h2,h3,p{margin-top:0}h1{font-size:clamp(2.9rem,6vw,5.6rem);line-height:1.02;letter-spacing:-.07em;margin:20px 0 25px}h1 em{font-style:normal;color:var(--lime)}h2{font-size:clamp(2rem,4vw,3.1rem);letter-spacing:-.055em;line-height:1.13;margin:10px 0 15px}
.hero p{color:#d8dff0;font-size:1.13rem}.motif{border:1px solid #647092;background:linear-gradient(130deg,#2a3757,#1c2440);padding:27px;border-radius:16px}.wave{padding:16px 10px;color:var(--lime);border-left:4px solid var(--lime);font:700 .88rem ui-monospace,monospace;letter-spacing:.08em}.wave.alt{color:#f2a28e;border-color:var(--red)}
.section{padding:67px 0;border-bottom:1px solid var(--line)}.lead{color:var(--dim);font-size:1.12rem;max-width:830px}.facts{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:28px 0}.fact,.panel{background:var(--white);border:1px solid var(--line);border-radius:11px;padding:21px}.fact strong{display:block;font-size:2rem;letter-spacing:-.05em}.fact small,.note{color:var(--dim)}.note{font-size:.88rem}
.split{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:25px 0}.split div{background:var(--ink);color:white;border-radius:9px;padding:18px}.split b{display:block;font-size:1.7rem}.split small{color:#cbd4e6}
.controls{display:flex;align-items:center;flex-wrap:wrap;gap:13px;margin:24px 0 19px}select{font:inherit;padding:9px 13px;color:var(--ink);background:white;border:1px solid #aeb5c2;border-radius:7px}.grid{display:grid;grid-template-columns:1.3fr .9fr;gap:18px;align-items:start}
.barline{display:grid;grid-template-columns:135px 1fr 46px;gap:8px;align-items:center;font-size:.88rem;margin:14px 0}.track{height:20px;background:#e6e4de;border-radius:4px;overflow:hidden}.fill{height:100%;background:var(--blue)}.fill.target{background:var(--red)}.legend{display:flex;gap:18px;font-size:.84rem;color:var(--dim);margin-top:17px}.dot{display:inline-block;width:11px;height:11px;background:var(--blue);margin-right:5px}.dot.target{background:var(--red)}
.metric{display:flex;justify-content:space-between;gap:13px;border-bottom:1px solid var(--line);padding:11px 0}.metric:last-child{border:0}.metric b{white-space:nowrap}.callout{background:#fff0e7;border-left:5px solid var(--red);padding:21px;margin-top:21px}
table{border-collapse:collapse;width:100%;font-size:.88rem}td,th{border-bottom:1px solid var(--line);padding:9px 6px;text-align:left}th{text-transform:uppercase;font-size:.75rem;letter-spacing:.07em}.num{text-align:right;font-variant-numeric:tabular-nums}.scroll{overflow:auto}
pre{background:var(--ink);color:#f0eee5;padding:22px;border-radius:8px;white-space:pre-wrap;overflow:auto}code{background:#e5e5e1;padding:2px 4px;border-radius:3px}a{color:var(--blue)}footer{padding:30px 0;font-size:.87rem;color:#d5daea}footer a{color:var(--lime)}
@media(max-width:850px){.hero,.grid{grid-template-columns:1fr}.hero{padding:45px 0}.facts{grid-template-columns:repeat(2,1fr)}}@media(max-width:520px){.facts,.split{grid-template-columns:1fr}.barline{grid-template-columns:105px 1fr 42px}.wrap{padding:0 17px}nav a{margin-left:8px}}
</style></head><body><header><div class="wrap"><nav><div class="brand">Sense<i>Shift</i> / field notes</div><div><a href="#experiment">Experiment</a><a href="#results">Results</a><a href="#limits">Limits</a><a href="#reproduce">Reproduce</a></div></nav>
<div class="hero"><div><div class="eyebrow">Activity recognition / 51 people / 18 activities</div><h1>What happens when <em>the sensor changes?</em></h1><p>A model trained on phone movement meets new people and watch recordings. We measure what survives, what fails, and whether its confidence can still be trusted.</p></div><div class="motif" aria-label="Illustrative signal comparison"><p>ILLUSTRATIVE SENSOR TRACES</p><div class="wave">PHONE / POCKET<br>▁▃▇▂▁▆▂▁▅▃▁▇▃▁▆▁</div><div class="wave alt">WATCH / WRIST<br>▃▇▂▅▁▆▃▇▁▅▂▇▃▁▆▂</div><p class="note" style="color:#bcc5db;margin:20px 0 0">The figures below use measured experiments.</p></div></div></div></header>
<main><section id="experiment" class="section"><div class="wrap"><div class="kicker">01 / The question</div><h2>A new person is the real test.</h2><p class="lead">A random split of ten-second clips can put one person's movements in both training and testing. Here, every recording from each test person is held out together. Phone and watch clocks differ, so their windows are evaluated separately.</p>
<div class="facts"><div class="fact"><strong>51</strong><small>participants</small></div><div class="fact"><strong>18</strong><small>activities</small></div><div class="fact"><strong id="windows">—</strong><small>retained ten-second windows</small></div><div class="fact"><strong>3</strong><small>subject splits</small></div></div><div class="split"><div><b>31</b><small>people for training</small></div><div><b>10</b><small>people for settings</small></div><div><b>10</b><small>new people for testing</small></div></div><p class="note">The follow-up excludes the first 15 seconds after each activity starts and resamples by timestamp to 20 Hz. It reuses the original participants, so this is a sensitivity study rather than a new untouched test.</p></div></section>
<section id="results" class="section"><div class="wrap"><div class="kicker">02 / Measured results</div><h2>Accuracy is only part of the story.</h2><p class="lead">The same phone-trained models are tested on held-out people's phones and watches. A separately trained watch forest is a practical fallback when the phone is unavailable.</p><div class="controls"><label for="seed">Participant split</label><select id="seed"></select><label for="metric">Measure</label><select id="metric"><option value="macro_f1">Macro F1</option><option value="balanced_accuracy">Balanced accuracy</option><option value="accuracy">Accuracy</option></select></div>
<div class="grid"><div class="panel"><h3>Phone training → held-out device</h3><div id="bars"></div><div class="legend"><span><i class="dot"></i>Phone</span><span><i class="dot target"></i>Watch</span></div></div><div><div class="panel"><h3>Transformer vs. forests</h3><div id="contrast"></div><p class="note">Macro-F1 differences with descriptive intervals from resampling ten held-out people. The raw forest sees the same 200 input values as the transformer. This follow-up reuses the test people.</p></div><div class="panel" style="margin-top:18px"><h3>When the phone is absent</h3><div id="fallback"></div><p class="note">Both watch results use the same held-out people, with independent windows.</p></div></div></div><div class="callout" id="interpretation"></div>
<div class="grid" style="margin-top:20px"><div class="panel"><h3>Confidence under shift</h3><p class="note">The confidence cutoff is chosen on source validation for 80% coverage. Actual test coverage can differ.</p><div id="selective"></div></div><div class="panel"><h3>Which activities break?</h3><label for="inspect">Inspect model </label><select id="inspect"></select><div class="scroll" id="activity"></div></div></div></div></section>
<section class="section"><div class="wrap"><div class="kicker">03 / Repeatability</div><h2>Does it depend on who was held out?</h2><p class="lead">The three splits share participants. Their scores show sensitivity, not independent trials or a leaderboard.</p><div class="panel scroll" id="repeats"></div></div></section>
<section id="limits" class="section"><div class="wrap"><div class="kicker">04 / Read with care</div><h2>The boundaries of this result.</h2><div class="grid"><div class="panel"><h3>What this establishes</h3><p>The models were trained and scored on recordings from all 51 subjects, with held-out people, source-only settings and separate device tests. The transition-exclusion and matched-input analyses were added after the first results were reviewed.</p><h3>What it cannot establish</h3><p>This is controlled activity data. Phone versus watch also changes sensor placement. It is not a claim about arbitrary devices, free-living behavior or healthcare use.</p></div><div class="panel"><h3>Where errors can hide</h3><p>Mixed native recording rates, missing or broken segments, similar activities and different movement styles. Confidence calibrated on a phone can fail on a watch.</p><p class="note">See the <a href="AUDIT.md">data audit</a> and <a href="ORIGINAL_RESULTS.md">original results</a> for the study timeline.</p></div></div></div></section>
<section id="reproduce" class="section"><div class="wrap"><div class="kicker">05 / Reproduce</div><h2>Every number has a path back to data.</h2><p>Download the official <a href="https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset">WISDM dataset</a> or let the acquisition command fetch it. Raw data stays outside Git; the dataset is credited under CC BY 4.0.</p>
<pre>python -m venv .venv
# activate .venv for your operating system
pip install -e '.[deep]'
senseshift acquire --download
senseshift prepare
senseshift run
senseshift report</pre><p>Inspect the <a href="results.json">measured results</a>, <a href="PROTOCOL.md">fixed protocol</a> and repository README. The repository contains all source code.</p></div></section></main>
<footer><div class="wrap">SenseShift · 2026 research project · WISDM Lab / Gary Weiss, CC BY 4.0 · <a href="https://doi.org/10.24432/C5HK59">Dataset DOI</a></div></footer>
<script>
const data=__PAYLOAD__,names=__NAMES__;
const display={dummy:'Prior only',linear:'Logistic regression',rf:'Feature forest',rf_raw:'Raw-sample forest',cnn:'Small CNN',transformer:'Patch transformer',watch_rf_fallback:'Watch RF fallback'};
const el=id=>document.getElementById(id),pct=v=>v==null?'—':(100*v).toFixed(1)+'%',pp=v=>(v>=0?'+':'')+(100*v).toFixed(1)+' pp';
el('windows').textContent=data.dataset.windows.toLocaleString();
el('seed').innerHTML=data.runs.map((r,i)=>'<option value="'+i+'">Seed '+r.seed+'</option>').join('');
function draw(){const r=data.runs[Number(el('seed').value)],measure=el('metric').value,models=Object.keys(r.models).filter(m=>m!=='watch_rf_fallback');
el('bars').innerHTML=models.map(m=>{const z=r.models[m].tests;return '<div class="barline"><b>'+display[m]+'</b><div><div class="track"><div class="fill" style="width:'+100*z.phone[measure]+'%"></div></div><div class="track" style="margin-top:4px"><div class="fill target" style="width:'+100*z.watch[measure]+'%"></div></div></div><b>'+pct(z.phone[measure])+'<br>'+pct(z.watch[measure])+'</b></div>'}).join('');
const h=r.hypothesis,matched=r.matched_input_contrast;el('contrast').innerHTML=h.phone?['phone','watch'].map(d=>{const a=h[d],b=matched[d];return '<div class="metric"><span>'+d+' / vs feature forest<br><small class="note">95%: '+pp(a.subject_bootstrap_95_interval[0])+' to '+pp(a.subject_bootstrap_95_interval[1])+'</small></span><b>'+pp(a.difference_transformer_minus_rf)+'</b></div><div class="metric"><span>'+d+' / vs raw forest<br><small class="note">95%: '+pp(b.subject_bootstrap_95_interval[0])+' to '+pp(b.subject_bootstrap_95_interval[1])+'</small></span><b>'+pp(b.difference_transformer_minus_rf)+'</b></div>'}).join(''):'Comparison unavailable';
const f=r.models.watch_rf_fallback?.tests.watch?.macro_f1,baseline=r.models.rf?.tests.watch?.macro_f1;
el('fallback').innerHTML=f==null?'Unavailable':'<div class="metric"><span>Phone forest → watch</span><b>'+pct(baseline)+'</b></div><div class="metric"><span>Watch forest → watch</span><b>'+pct(f)+'</b></div>';
const ci=matched.watch?.subject_bootstrap_95_interval,change=matched.watch?.difference_transformer_minus_rf;
el('interpretation').innerHTML=ci?'<b>Matched input, watch test:</b> The transformer–raw-forest difference is '+pp(change)+'. The subject interval '+(ci[0]<=0&&ci[1]>=0?'crosses zero.':ci[0]>0?'lies above zero.':'lies below zero.')+' This comparison was added after the original test was seen; it is descriptive, not fresh confirmation.':'No matched-input comparison in this run.';
el('selective').innerHTML=models.filter(m=>m!=='dummy').map(m=>{const z=r.models[m].tests;return '<div class="metric"><span><b>'+display[m]+'</b><br><small class="note">Phone: '+pct(z.phone.coverage)+' kept / '+pct(z.phone.accepted_accuracy)+' correct<br>Watch: '+pct(z.watch.coverage)+' kept / '+pct(z.watch.accepted_accuracy)+' correct</small></span></div>'}).join('');
let selected=el('inspect').value;if(!r.models[selected])selected=models.includes('rf')?'rf':models[0];el('inspect').innerHTML=models.map(m=>'<option value="'+m+'" '+(m===selected?'selected':'')+'>'+display[m]+'</option>').join('');drawActivity()}
function drawActivity(){const r=data.runs[Number(el('seed').value)],z=r.models[el('inspect').value].tests;el('activity').innerHTML='<table><thead><tr><th>Activity</th><th>Phone F1</th><th>Watch F1</th><th>Watch n</th></tr></thead><tbody>'+Object.keys(names).map(k=>'<tr><td>'+names[k]+'</td><td class="num">'+pct(z.phone.class_f1[k])+'</td><td class="num">'+pct(z.watch.class_f1[k])+'</td><td class="num">'+z.watch.class_support[k]+'</td></tr>').join('')+'</tbody></table><p class="note">Top watch confusions: '+z.watch.top_confusions.slice(0,3).map(c=>names[c.true]+' → '+names[c.predicted]+' ('+c.count+')').join('; ')+'</p>'}
function drawRepeats(){const rows=[];for(const r of data.runs)for(const [m,z] of Object.entries(r.models))rows.push('<tr><td>'+r.seed+'</td><td>'+display[m]+'</td><td class="num">'+(z.tests.phone?pct(z.tests.phone.macro_f1):'—')+'</td><td class="num">'+pct(z.tests.watch.macro_f1)+'</td><td class="num">'+pct(z.tests.watch.coverage)+'</td></tr>');el('repeats').innerHTML='<table><thead><tr><th>Seed</th><th>Model</th><th>Phone macro F1</th><th>Watch macro F1</th><th>Watch coverage</th></tr></thead><tbody>'+rows.join('')+'</tbody></table>'}
el('seed').addEventListener('change',draw);el('metric').addEventListener('change',draw);el('inspect').addEventListener('change',drawActivity);draw();drawRepeats();
</script></body></html>'''
    out.write_text(page.replace("__PAYLOAD__", payload).replace("__NAMES__", names))
    _write_summary(results, out.parent / "RESULTS.md")
    return out


def _write_summary(data: dict, path: Path):
    runs = data["runs"]
    model_names = {"dummy": "Prior only", "linear": "Logistic regression",
                   "rf": "Feature forest", "rf_raw": "Raw-sample forest", "cnn": "CNN",
                   "transformer": "Patch transformer", "watch_rf_fallback": "Watch RF fallback"}
    def values(model, device, metric):
        return [r["models"][model]["tests"][device][metric] for r in runs]
    def pct(v):
        return f"{mean(v):.3f} ({min(v):.3f}–{max(v):.3f})"
    lines = ["# Results: transition-exclusion follow-up", "",
             "This 15-second transition-exclusion run was designed **after** the original "
             "results were inspected. It reuses the same subject splits; it is a sensitivity "
             "analysis, not fresh confirmation. The [original results](ORIGINAL_RESULTS.md) "
             "and `artifacts/original_results.json` remain available. "
             "The raw-sample forest is a post hoc matched-input comparison.", "",
             "All 51 subjects and all 18 activity classes were used. "
             "The three fixed person-level splits hold out 10 subjects for testing each time. "
             "Scores below are arithmetic means and ranges across these overlapping splits, "
             "not independent confidence intervals.", "",
             "| Training → test | Model | Macro-F1 mean (range) | Accuracy mean (range) | ECE mean | Coverage mean | Accepted accuracy mean |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for model in model_names:
        devices = ("watch",) if model == "watch_rf_fallback" else ("phone", "watch")
        for device in devices:
            origin = "watch" if model == "watch_rf_fallback" else "phone"
            lines.append(f"| {origin} → {device} | {model_names[model]} | "
                         f"{pct(values(model, device, 'macro_f1'))} | "
                         f"{pct(values(model, device, 'accuracy'))} | "
                         f"{mean(values(model, device, 'ece_10')):.3f} | "
                         f"{mean(values(model, device, 'coverage')):.3f} | "
                         f"{mean(values(model, device, 'accepted_accuracy')):.3f} |")
    lines += ["", "Macro-F1 weights the 18 activities equally. ECE is ten-bin expected "
              "calibration error (smaller is better). Coverage is the share of test windows "
              "the model accepts using a confidence cutoff set for 80% coverage on validation. "
              "Accepted accuracy only describes those kept windows; it should be read together "
              "with realized coverage.", "", "## Original contrast, revisited on trimmed data", "",
              "Transformer minus phone-trained feature forest macro-F1. The contrast was "
              "in the original protocol, but these amended data and repeated test people were not. "
              "Intervals resample the 10 held-out **people**, keeping each person's windows together.", "",
              "| Split | Test device | Difference | Descriptive subject-bootstrap 95% interval |",
              "| ---: | --- | ---: | --- |"]
    all_differences = []
    for r in runs:
        for device in ("phone", "watch"):
            h = r["hypothesis"][device]
            d = h["difference_transformer_minus_rf"]
            lo, hi = h["subject_bootstrap_95_interval"]
            all_differences.append(d)
            lines.append(f"| {r['seed']} | {device} | {d:+.3f} | [{lo:+.3f}, {hi:+.3f}] |")
    if all(d < 0 for d in all_differences):
        lines += ["", "The transformer was below the forest on both test devices in all "
                  "three measured splits. The conclusion is limited to these trained models, "
                  "features, people, activities, and recording conditions. It does not show "
                  "that transformers cannot work for activity recognition."]
    lines += ["", "## Matched input: transformer versus raw-sample forest", "",
              "Both models see the same 200 ordered magnitude samples. The forest has "
              "240 trees with the original fixed forest settings. This comparison was "
              "added after viewing the original test results, so its intervals are "
              "descriptive and do not constitute an untouched hypothesis test.", "",
              "| Split | Test device | Transformer minus raw forest F1 | Descriptive subject-bootstrap 95% interval |",
              "| ---: | --- | ---: | --- |"]
    for r in runs:
        for device in ("phone", "watch"):
            h = r["matched_input_contrast"][device]
            lo, hi = h["subject_bootstrap_95_interval"]
            lines.append(f"| {r['seed']} | {device} | {h['difference_transformer_minus_rf']:+.3f} | "
                         f"[{lo:+.3f}, {hi:+.3f}] |")
    lines += ["", "## Practical device finding", "",
              "A forest trained on watch examples from the train people performs better on "
              "held-out watch examples than the forest trained on those people's phone examples. "
              "The mean watch macro-F1 values are "
              f"{mean(values('watch_rf_fallback', 'watch', 'macro_f1')):.3f} and "
              f"{mean(values('rf', 'watch', 'macro_f1')):.3f}, respectively. "
              "This changes both the training device and its placement, so it does not isolate "
              "one physical cause. A **post hoc**, paired subject analysis of this fallback "
              "appears in [AUDIT.md](AUDIT.md); it was not the predeclared model contrast.", "",
              "## Important limitations", "",
              "- A phone in a pocket is less informative for some fine hand activities than "
              "a watch on a wrist. Per-class F1 and confusion matrices are in "
              "[results.json](results.json) and the [report](index.html).",
              "- The original feature forest uses hand-designed magnitude features; the CNN, "
              "transformer and added raw forest use the same magnitude samples. Matching "
              "input values does not equalize model tuning or inductive biases.",
              "- The first 15 seconds were excluded based on a published data concern, after "
              "the original results were known. Retained windows shift in time and are "
              "not paired with the original windows.",
              "- These are controlled scripted activities, not a field deployment or a medical device.",
              "- Three test splits reuse subjects. Treat across-split ranges as a sensitivity "
              "check, not as independent trials.", "",
              "See [PROTOCOL.md](PROTOCOL.md) for the frozen design and "
              "[AUDIT.md](AUDIT.md) for source rate and sample checks.", ""]
    path.write_text("\n".join(lines))
