"""Optional deliverable 6: one self-contained HTML dashboard (no server, no external libraries)."""
import json, pandas as pd
from common import *
pol = pd.read_csv(ROOT / "results" / "policy_summary.csv")
cr = pd.read_csv(ROOT / "results" / "cost_results.csv")
mt = pd.read_csv(ROOT / "results" / "metrics.csv")
t = mt[(mt.eval_set == "test_last") & (mt.truth == "uncapped") & (mt.target.isin(["capped"])) & (mt.model != "naive") & (mt.src == mt.tgt)]
nv = mt[(mt.eval_set == "test_last") & (mt.truth == "uncapped") & (mt.model == "naive") & (mt.src == mt.tgt)]
models = pd.concat([t, nv.assign(feats="n/a")])[["src", "model", "feats", "rmse", "mae", "nasa_score", "overest_rate"]].round(2)
cross = pd.read_csv(ROOT / "results" / "cross_dataset_drop.csv").round(2)
data = dict(pol=pol.round(3).to_dict("records"), cr=cr[cr.policy.isin(["fixed", "rul"])][["scenario", "policy", "cost_ratio", "n_failures", "n_engines", "mean_wasted_rul", "param", "param_k"]].round(2).fillna(0).to_dict("records"),
            models=models.to_dict("records"), cross=cross.to_dict("records"))
html = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Predictive Engine Maintenance Dashboard</title><style>
:root{--bg:#fff;--fg:#1c2430;--mut:#667085;--card:#f5f7fa;--line:#dfe3ea;--rtf:#9aa0a6;--fx:#c0504d;--rul:#1f6fb2;--or:#3a9d5d}
@media(prefers-color-scheme:dark){:root{--bg:#12161c;--fg:#e6e9ef;--mut:#98a1b0;--card:#1b212b;--line:#2c3441}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif}main{max-width:980px;margin:0 auto;padding:20px 16px 48px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 8px}p.s{color:var(--mut);margin:0 0 16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px}.row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
select,input{font:inherit;color:var(--fg);background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:5px 8px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:right}th:first-child,td:first-child{text-align:left}
.kpi{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-top:12px}.kpi div{background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:8px 10px}
.kpi b{display:block;font-size:20px}.kpi span{color:var(--mut);font-size:12px}.note{color:var(--mut);font-size:12px;margin-top:8px}.scroll{overflow-x:auto}
canvas{width:100%;height:260px;display:block}
</style></head><body><main>
<h1>Predictive Engine Maintenance: when should we pull an engine?</h1>
<p class="s">NASA C-MAPSS (simulated turbofan data). All costs are illustrative assumptions; only the comparison between policies is meaningful. Decision support only.</p>
<h2>1 &middot; Cost of each policy vs failure-to-planned cost ratio</h2>
<div class="card"><div class="row"><label>Scenario <select id="sc"></select></label>
<label>Cost ratio r <input id="r" type="range" min="0" max="0" step="1" style="vertical-align:middle"> <b id="rv"></b></label></div>
<canvas id="cv"></canvas><div class="kpi" id="kpi"></div>
<p class="note">Cost per 1000 engine-cycles, planned maintenance = 1. Thresholds are tuned on one half of the engines and scored on the other half (mean of 40 splits). "Tuned on FD001" = threshold chosen on FD001 and moved unchanged to another fleet.</p></div>
<h2>2 &middot; Model comparison (official test engines, final cycle, XGBoost/linear with capped label)</h2><div class="card scroll"><table id="mt"></table></div>
<h2>3 &middot; Cross-dataset drop (FD001-trained XGBoost)</h2><div class="card scroll"><table id="ct"></table></div>
</main><script>
const D=__DATA__;
const scs=[...new Set(D.pol.map(d=>d.scenario))],rs=[...new Set(D.pol.map(d=>d.cost_ratio))].sort((a,b)=>a-b);
const sc=document.getElementById('sc'),rr=document.getElementById('r');scs.forEach(s=>sc.add(new Option(s,s)));rr.max=rs.length-1;rr.value=rs.indexOf(10);
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v)}
function draw(){const s=sc.value,r=rs[+rr.value];document.getElementById('rv').textContent=r;
 const g=D.pol.filter(d=>d.scenario===s&&d.cost_ratio===r)[0];const vals=[['Run-to-failure',g.rtf,'--rtf'],['Best fixed interval',g.fixed,'--fx'],['RUL threshold',g.rul,'--rul'],['Perfect-RUL oracle',g.oracle,'--or']];
 const cv=document.getElementById('cv'),dpr=devicePixelRatio||1,W=cv.clientWidth,H=cv.clientHeight;cv.width=W*dpr;cv.height=H*dpr;const c=cv.getContext('2d');c.scale(dpr,dpr);c.clearRect(0,0,W,H);
 const mx=Math.max(...vals.map(v=>v[1]))*1.1,bw=W/vals.length;c.font='12px system-ui';c.textAlign='center';
 vals.forEach((v,i)=>{const h=(H-50)*v[1]/mx;c.fillStyle=css(v[2]);c.fillRect(i*bw+bw*.2,H-30-h,bw*.6,h);c.fillStyle=css('--fg');c.fillText(v[1].toFixed(1),i*bw+bw/2,H-34-h);c.fillStyle=css('--mut');c.fillText(v[0],i*bw+bw/2,H-10)});
 const f=D.cr.filter(d=>d.scenario===s&&d.cost_ratio===r);const fx=f.find(d=>d.policy==='fixed'),ru=f.find(d=>d.policy==='rul');
 document.getElementById('kpi').innerHTML=`<div><b>${g.gain_pct.toFixed(0)}%</b><span>RUL policy saving vs fixed (negative = worse)</span></div>
 <div><b>${(100*ru.n_failures/ru.n_engines).toFixed(1)}% / ${(100*fx.n_failures/fx.n_engines).toFixed(1)}%</b><span>unplanned failures, RUL / fixed</span></div>
 <div><b>${ru.mean_wasted_rul.toFixed(0)} / ${fx.mean_wasted_rul.toFixed(0)}</b><span>cycles of life wasted per service, RUL / fixed</span></div>
 <div><b>${Math.round(100*g.rul_wins)}%</b><span>of splits where RUL is cheaper</span></div>`}
sc.onchange=rr.oninput=draw;addEventListener('resize',draw);matchMedia('(prefers-color-scheme:dark)').onchange=draw;draw();
function tbl(id,rows,cols){document.getElementById(id).innerHTML='<tr>'+cols.map(c=>'<th>'+c+'</th>').join('')+'</tr>'+rows.map(r=>'<tr>'+cols.map(c=>'<td>'+(r[c]??'')+'</td>').join('')+'</tr>').join('')}
tbl('mt',D.models,['src','model','feats','rmse','mae','nasa_score','overest_rate']);
tbl('ct',D.cross,['test_set','features','rmse_FD001_trained','rmse_in_domain','rmse_increase_pct','overest_FD001_trained','overest_in_domain']);
</script></body></html>"""
out = ROOT / "dashboard"; out.mkdir(exist_ok=True)
(out / "index.html").write_text(html.replace("__DATA__", json.dumps(data)))
print("wrote", out / "index.html")
