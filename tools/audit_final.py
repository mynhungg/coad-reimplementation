"""Full audit from the raw CSVs: every main table, the hypotheses, robustness checks of doubtful points.
Writes audit_final.json (numbers) and prints a readable report. Independent of the analysis notebook."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
sys.stdout.reconfigure(encoding='utf-8')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 200)
STORE = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('CoAD_project')
R = STORE / 'results'
OUT = {}

def load(pattern):
    files = sorted(R.glob(pattern))
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True) if files else pd.DataFrame()
def holm(p):
    p = np.asarray(p, float); out = np.empty(len(p)); run = 0
    for i, j in enumerate(np.argsort(p)):
        run = max(run, (len(p) - i) * p[j]); out[j] = min(1, run)
    return out
def boot(d, n=10000, seed=2026):
    d = np.asarray(d); m = np.random.default_rng(seed).choice(d, (n, len(d))).mean(1)
    return np.percentile(m, [2.5, 97.5])
def wtest(d, alt='two-sided'):
    d = pd.Series(d).dropna()
    return float(wilcoxon(d, alternative=alt).pvalue) if (d != 0).any() else np.nan

K = {'f1_score': 'F1', 'auc_prc': 'AUC-PR', 'R_AUC_PR': 'R-AUC-PR', 'VUS_PR': 'VUS-PR'}
TABLE1 = {'CoAD': [49.13, 43.66, 40.50, 39.83], 'Kmeans': [41.42, 37.26, 37.64, 36.81], 'KshapeAD': [35.51, 32.67, 31.87, 31.27],
          'SAND': [34.74, 31.52, 31.85, 31.05], 'Series2Graph': [33.48, 30.11, 30.13, 29.57], 'MP': [35.10, 27.93, 29.88, 28.94],
          'PCA': [32.49, 27.59, 23.88, 23.85], 'damp': [17.44, 11.59, 12.98, 12.60]}
TABLE2 = {'time_soft': [43.10, 36.87, 37.91, 36.92], 'guide_hard': [44.06, 38.57, 35.31, 34.76], 'freq_time_mean': [42.24, 36.23, 33.71, 33.18]}
METHODS = list(TABLE1)

man = pd.read_csv(STORE / 'eda' / 'manifest_tsbad64.csv'); G = man['group'].tolist(); fam = man.set_index('group')['family']
tsb = load('tsbad64_results*.csv'); tsb = tsb[tsb.status == 'success'].sort_values('run_id').drop_duplicates(['method', 'group'], keep='last')
def M(m, k='VUS_PR'): return 100 * tsb[tsb.method == m].set_index('group')[k].reindex(G)
ri = load('reimpl_results_*.csv'); ri = ri[ri.status == 'success'].sort_values('run_id').drop_duplicates(['mode', 'group', 'seed'], keep='last')
code, paper = ri[ri['mode'] == 'code'], ri[ri['mode'] == 'paper']
def S(df, k='VUS_PR', seeds=None):
    d = df if seeds is None else df[df.seed.isin(seeds)]
    return 100 * d.groupby('group')[k].mean().reindex(G)
def seedstat(df, k='VUS_PR'):
    p = 100 * df.groupby('seed')[k].mean(); return p.mean(), p.std()

def section(t): print('\n' + '=' * 100 + '\n' + t + '\n' + '=' * 100)

# 1. Reproduction vs Table 1
section('1. Reproduction (one run per method) vs Table 1')
rows = []
for m in METHODS:
    for i, (k, n) in enumerate(K.items()):
        ours = M(m, k).mean(); rows.append(dict(method=m, metric=n, ours=ours, paper=TABLE1[m][i], delta=ours - TABLE1[m][i]))
t1 = pd.DataFrame(rows); print(t1.pivot(index='method', columns='metric', values='delta').round(2).loc[METHODS])
OUT['table1'] = t1.round(2).to_dict('records')

# 2. Reimplementation
section('2. Reimplementation: code mode (3 seeds), paper mode, official run, Table 1')
rows = []
for i, (k, n) in enumerate(K.items()):
    cm, cs = seedstat(code, k); pm, ps = seedstat(paper, k)
    d = S(code, k) - M('CoAD', k)
    rows.append(dict(metric=n, table1=TABLE1['CoAD'][i], official=M('CoAD', k).mean(), code=cm, code_sd=cs, paper=pm, paper_sd=ps,
                     code_minus_official_p=wtest(d), code_minus_table1=cm - TABLE1['CoAD'][i], gap_in_seed_sd=(cm - TABLE1['CoAD'][i]) / cs))
t2 = pd.DataFrame(rows); print(t2.round(3).to_string(index=False)); OUT['reimpl'] = t2.round(3).to_dict('records')
within = ((S(code) - M('CoAD')).abs() <= (100 * code.groupby('group')['VUS_PR'].agg(lambda v: v.max() - v.min())).reindex(G) / 2).mean()
lo = 100 * code.groupby('group')['VUS_PR'].min().reindex(G); hi = 100 * code.groupby('group')['VUS_PR'].max().reindex(G)
inside = int(((M('CoAD') >= lo) & (M('CoAD') <= hi)).sum())
print(f'official run inside the range of the 3 code-mode seeds: {inside}/64'); OUT['official_inside_seed_range'] = inside
mg = man.loc[man.family == 'MGAB', 'group']
print('MGAB code / paper:', round(100 * code[code.group.isin(mg)].VUS_PR.mean(), 2), round(100 * paper[paper.group.isin(mg)].VUS_PR.mean(), 2))

# 3. M5
section('3. M5: average ranks (8 methods of Table 1) and CoAD vs each baseline (Holm over 7)')
rk = {}
for k, n in K.items():
    mat = pd.DataFrame({m: M(m, k) for m in METHODS}); rk[n] = mat.rank(axis=1, ascending=False).mean()
rk = pd.DataFrame(rk).sort_values('VUS-PR'); print(rk.round(2)); OUT['ranks'] = rk.round(2).to_dict()
pw = []
for k, n in K.items():
    mat = pd.DataFrame({m: M(m, k) for m in METHODS}); ps = []
    for b in METHODS[1:]:
        d = mat['CoAD'] - mat[b]; lo_, hi_ = boot(d)
        ps.append(dict(metric=n, baseline=b, better=int((d > 0).sum()), worse=int((d < 0).sum()), diff=d.mean(), ci_low=lo_, ci_high=hi_, p=wtest(d)))
    for r, ph in zip(ps, holm([r['p'] for r in ps])): r['p_holm'] = ph; r['significant'] = ph < 0.05
    pw += ps
pw = pd.DataFrame(pw); print(pw[pw.metric == 'VUS-PR'].round(3).to_string(index=False))
print('significant after Holm, by metric:', pw[pw.significant].groupby('metric')['baseline'].apply(list).to_dict())
OUT['m5_pairwise'] = pw.round(4).to_dict('records')

# 4. M6-A
section('4. M6-A: smoothing on/off (code mode, 192 models)')
m6a = load('m6a_smoothing*.csv').drop_duplicates(['group', 'seed'], keep='last')
rows = []
for i, (k, n) in enumerate(K.items()):
    sm, ss = seedstat(code, k); um, us = seedstat(m6a, k); d = S(m6a, k) - S(code, k)
    rows.append(dict(metric=n, table1=TABLE1['CoAD'][i], smoothed=sm, unsmoothed=um, unsm_higher=int((d > 0).sum()), p=wtest(d),
                     closer_to_table1=abs(um - TABLE1['CoAD'][i]) < abs(sm - TABLE1['CoAD'][i])))
t4 = pd.DataFrame(rows); print(t4.round(3).to_string(index=False)); OUT['m6a'] = t4.round(3).to_dict('records')

# 5. M6-C
section('5. M6-C: group ablation on MGAB (9 series x 3 seeds)')
m6c = load('m6c_results*.csv'); m6c = m6c[m6c.status == 'success'].drop_duplicates(['config', 'group', 'seed'], keep='last')
ends = ri[ri.group.isin(mg)].assign(config=lambda d: d['mode'] + ' mode')
allr = pd.concat([ends, m6c]); per = 100 * allr.groupby(['config', 'group'])['VUS_PR'].mean().unstack(0)
rows = []
for c in per.columns:
    start = 'paper mode' if c.startswith('paper +') else 'code mode' if c.startswith('code -') else None
    r = dict(config=c, vus=per[c].mean(), seed_sd=(100 * allr[allr.config == c].groupby('seed')['VUS_PR'].mean()).std())
    if start:
        d = per[c] - per[start]; r.update(change=d.mean(), higher=int((d > 0).sum()), p=wtest(d))
    rows.append(r)
t5 = pd.DataFrame(rows).set_index('config'); print(t5.round(3)); OUT['m6c'] = t5.round(3).reset_index().to_dict('records')

# 6. M6-B
section('6. M6-B: authors\' ablation variants (seed 46) vs CoAD official run and Table 2')
ab = load('ablation_results*.csv'); ab = ab[ab.status == 'success'].drop_duplicates(['variant', 'group'], keep='last')
rows = []
for v, d in ab.groupby('variant'):
    d = d.set_index('group').reindex(G)
    for i, (k, n) in enumerate(K.items()):
        diff = 100 * d[k] - M('CoAD', k)
        rows.append(dict(variant=v, metric=n, ours=100 * d[k].mean(), diff=diff.mean(), lower=int((diff < 0).sum()), p=wtest(diff),
                         paper_diff=TABLE2[v][i] - TABLE1['CoAD'][i], same_direction=np.sign(diff.mean()) == np.sign(TABLE2[v][i] - TABLE1['CoAD'][i])))
t6 = pd.DataFrame(rows); print(t6.round(3).to_string(index=False)); OUT['m6b'] = t6.round(3).to_dict('records')

# 7. M7 / M8
section('7. M7 branch split (seed 46) and M8 reconstruction-only')
m7 = load('m7_branches*.csv'); m8 = load('m8_recon*.csv')
full46 = ri[ri.seed == 46].assign(branch='full')
b = pd.concat([full46[['mode', 'group', 'branch', 'VUS_PR']], m7[['mode', 'group', 'branch', 'VUS_PR']]])
b['family'] = fam.reindex(b.group).values
t7 = 100 * b.pivot_table(index='family', columns=['mode', 'branch'], values='VUS_PR'); t7.loc['all'] = 100 * b.groupby(['mode', 'branch'])['VUS_PR'].mean()
print(t7.round(2)); OUT['m7'] = {f'{a}|{b_}': v for (a, b_), v in t7.round(2).loc['all'].items()}
rec = pd.concat([m7[m7.branch == 'reconstruction only'].drop(columns='branch'), m8]).drop_duplicates(['mode', 'group', 'seed'], keep='last')
rows = []
for i, (k, n) in enumerate(K.items()):
    cm = seedstat(code, k)[0]; rm, rs = seedstat(rec[rec['mode'] == 'code'], k); d = S(rec[rec['mode'] == 'code'], k) - S(code, k)
    rows.append(dict(metric=n, table1=TABLE1['CoAD'][i], coad_code=cm, recon_only_code=rm, diff=d.mean(), higher=int((d > 0).sum()), p=wtest(d)))
t8 = pd.DataFrame(rows); print(t8.round(3).to_string(index=False)); OUT['m8_code'] = t8.round(3).to_dict('records')
pr, cf = S(rec[rec['mode'] == 'paper'], seeds=[47, 48]), S(code, seeds=[47, 48])
OUT['m8_prereg'] = dict(paper_recon=pr.mean(), code=cf.mean(), wins=int((pr > cf).sum()), p=wtest(pr - cf, 'greater'))
print('pre-registered (paper-mode recon vs code, seeds 47-48):', {k: round(v, 4) for k, v in OUT['m8_prereg'].items()})

# 8. M9
section('8. M9: error analysis (8262 events), problem 1, module A, problem 2 stress')
ev = load('m9_events*.csv').drop_duplicates(['group', 'seed', 'event'], keep='last'); mo = load('m9_models*.csv').drop_duplicates(['group', 'seed'], keep='last')
st = load('m9_stress*.csv')
def step(r, th):
    if r.full_peak >= th: return 'found'
    if r.full_unsmoothed_peak >= th: return 'smoothing'
    if max(r.classify_peak, r.reconstruct_peak) >= th: return 'fusion'
    if max(r.oe_peak, r.residual_peak, r.error_peak) >= th: return 'processing'
    return 'all weak'
for th in [0.95, 0.99, 0.999]:
    s = ev.apply(lambda r: step(r, th), axis=1)
    tab = pd.crosstab(ev.family, s, normalize='index')
    miss = ev[ev.oe_peak < th]
    print(f'\nthreshold {th}: share found by family / step shares (all events):'); print(tab.round(3))
    print(f'  OE misses {len(miss)}/{len(ev)}; of those found by raw error {(miss.error_peak >= th).mean():.3f}, '
          f'reconstruct branch {(miss.reconstruct_peak >= th).mean():.3f}, final {(miss.full_peak >= th).mean():.3f}, backup {(miss.backup_peak >= th).mean():.3f}')
    OUT[f'm9_steps_{th}'] = tab.round(3).to_dict()
    OUT[f'm9_rescue_{th}'] = dict(oe_missed=len(miss), raw_error=(miss.error_peak >= th).mean(), reconstruct=(miss.reconstruct_peak >= th).mean(),
                                 final=(miss.full_peak >= th).mean(), backup=(miss.backup_peak >= th).mean())
r = spearmanr(ev.oe_mean, ev.error_mean); OUT['m9_oe_error_rho'] = float(r.statistic); print('Spearman OE vs error', round(r.statistic, 3))
print('alarms:', mo.groupby('family')[['top_k_normal_share', 'normal_alarms_near_share']].mean().round(3).to_dict())
OUT['m9_alarms'] = mo.groupby('family')[['top_k_normal_share', 'normal_alarms_near_share']].mean().round(3).to_dict()
mo_i = mo.set_index(['group', 'seed']); cr = code.set_index(['group', 'seed'])
rows = []
for v in ['calibrated', 'backup', 'calibrated + backup']:
    for i, (k, n) in enumerate(K.items()):
        x = 100 * mo_i[f'{v}: {k}']; c = 100 * cr[k].reindex(x.index)
        d = (x - c).groupby(level='group').mean(); per_seed = (x - c).groupby(level='seed').mean()
        rows.append(dict(variant=v, metric=n, value=x.groupby(level='seed').mean().mean(), coad=c.groupby(level='seed').mean().mean(),
                         diff=d.mean(), higher=int((d > 0).sum()), p_one_sided=wtest(d, 'greater'),
                         gain_seed46=per_seed.get(46), gain_seed47=per_seed.get(47), gain_seed48=per_seed.get(48)))
t10 = pd.DataFrame(rows); print(t10.round(4).to_string(index=False)); OUT['module_a'] = t10.round(4).to_dict('records')
cal = (100 * mo_i['calibrated: VUS_PR']).groupby(level='group').mean().reindex(G); cpb = (100 * mo_i['calibrated + backup: VUS_PR']).groupby(level='group').mean().reindex(G)
print('calibrated + backup vs calibrated: diff %.3f, higher %d/64, two-sided p %.4f' % ((cpb - cal).mean(), ((cpb - cal) > 0).sum(), wtest(cpb - cal)))
OUT['backup_over_calibrated'] = dict(diff=(cpb - cal).mean(), higher=int(((cpb - cal) > 0).sum()), p=wtest(cpb - cal))
print('calibrated - CoAD code by family:', (cal - S(code)).groupby(fam.reindex(G).values).mean().round(2).to_dict())
OUT['calibrated_by_family'] = (cal - S(code)).groupby(fam.reindex(G).values).mean().round(2).to_dict()
mat = pd.DataFrame({'calibrated': cal, **{m: M(m) for m in METHODS}})
rk9 = mat.rank(axis=1, ascending=False).mean().sort_values(); print('ranks with calibrated:', rk9.round(2).to_dict()); OUT['ranks_with_calibrated'] = rk9.round(2).to_dict()
ps = [wtest(mat['calibrated'] - mat[m]) for m in METHODS]
OUT['calibrated_vs_table1'] = [dict(vs=m, better=int((mat['calibrated'] > mat[m]).sum()), diff=float((mat['calibrated'] - mat[m]).mean()), p=p, p_holm=h)
                               for m, p, h in zip(METHODS, ps, holm(ps))]
print(pd.DataFrame(OUT['calibrated_vs_table1']).round(4).to_string(index=False))
u = st[(st.offset == 0)].groupby(['type', 'length_periods'])['full_mean'].mean().unstack(); print('stress full mean pct:'); print(u.round(3))
OUT['stress'] = u.round(3).to_dict()

# 9. M10
section('9. M10: hypotheses of 04.md')
e10 = load('m10_events*.csv').drop_duplicates(['group', 'seed', 'event'], keep='last')
s10 = load('m10_stress*.csv'); m10 = load('m10_models*.csv').drop_duplicates(['group', 'seed'], keep='last'); g10 = load('m10_gradient*.csv')
h = {}
d = e10.oe_middle - e10.oe_edges; h['H1-A'] = dict(value=d.mean(), n=len(d), p_less=wtest(d, 'less'), p_two=wtest(d), predicted='<= -0.05', supported=bool(d.mean() <= -0.05 and wtest(d, 'less') < 0.05))
w = s10.pivot_table(index=['group', 'seed', 'position', 'type', 'amplitude'], columns='boundary', values='oe_plateau'); d = (w.abrupt - w.smooth).dropna()
h['H1-B'] = dict(value=d.mean(), n=len(d), p_greater=wtest(d, 'greater'), p_two=wtest(d), predicted='>= 0.05', supported=bool(d.mean() >= 0.05 and wtest(d, 'greater') < 0.05))
d = m10.auc_hard - m10.auc_soft; h['H2-A'] = dict(value=d.mean(), n=len(d), p_greater=wtest(d, 'greater'), p_two=wtest(d), higher=int((d > 0).sum()), predicted='>= 0.02', supported=bool(d.mean() >= 0.02 and wtest(d, 'greater') < 0.05))
d = (m10.gain_anomalous - m10.gain_normal).dropna(); h['H2-B'] = dict(value=d.mean(), median=d.median(), n=len(d), p_greater=wtest(d, 'greater'), p_two=wtest(d), predicted='> 0', supported=bool(d.mean() > 0 and wtest(d, 'greater') < 0.05))
wv = 100 * g10.pivot_table(index='group', columns='variant', values='VUS_PR'); d = wv.detached - wv.joint
cos = g10[g10.variant == 'joint'].cosine_mean
h['Gradient'] = dict(cosine=cos.mean(), cosine_negative_runs=int((cos < 0).sum()), detached_minus_joint=d.mean(), lower=int((d < 0).sum()), p_two=wtest(d), p_greater=wtest(d, 'greater'), supported=bool(cos.mean() < 0 and d.mean() > 0 and wtest(d, 'greater') < 0.05))
for k_, v in h.items(): print(k_, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})
print('mask level (A_c) on anomalous vs normal patches by family:'); print(m10.groupby('family')[['mask_level_anomalous', 'mask_level_normal', 'auc_soft', 'auc_hard']].mean().round(3))
OUT['m10'] = h

# 10. Doubtful points
section('10. Robustness of doubtful points')
# run-to-run noise: same code, same seed 46, MGAB: section 7 vs the joint run of 15.4
j = wv['joint']; s7 = 100 * code[code.seed == 46].set_index('group')['VUS_PR'].reindex(j.index)
noise = (j - s7)
print('MGAB, same code and seed (joint 15.4 vs section 7): mean diff %.2f, mean |diff| %.2f, max |diff| %.2f' % (noise.mean(), noise.abs().mean(), noise.abs().max()))
print('detached - joint per series:', d.round(2).to_dict())
print('detached - section-7 seed 46: mean %.2f, lower on %d/9' % ((wv.detached - s7).mean(), ((wv.detached - s7) < 0).sum()))
sd3 = (100 * code[code.group.isin(mg)].groupby('group')['VUS_PR'].std()).mean()
print('MGAB per-series SD over the 3 code-mode seeds: mean %.2f' % sd3)
OUT['noise_mgab'] = dict(same_seed_mean_abs_diff=noise.abs().mean(), same_seed_mean_diff=noise.mean(), per_series_seed_sd=sd3,
                         detached_minus_section7=(wv.detached - s7).mean(), detached_lower_than_section7=int(((wv.detached - s7) < 0).sum()))
# M6-B effect vs noise: per-series seed SD of code mode, 64 series
sd64 = (100 * code.groupby('group')['VUS_PR'].std()).mean(); seedmean_sd = seedstat(code)[1]
print('64 series: mean per-series seed SD %.2f; SD of the 64-series mean over seeds %.2f; official run - code 3-seed mean %.2f'
      % (sd64, seedmean_sd, (M('CoAD') - S(code)).mean()))
OUT['noise_64'] = dict(per_series_seed_sd=sd64, seed_sd_of_mean=seedmean_sd)
# M6-B against the 3-seed code mean instead of the single official run
for v, dd in ab.groupby('variant'):
    dd = dd.set_index('group').reindex(G); x = 100 * dd['VUS_PR'] - S(code)
    print(f'M6-B {v}: vs code-mode 3-seed mean {x.mean():+.2f} (p {wtest(x):.3f}); vs code seed 46 {(100 * dd["VUS_PR"] - S(code, seeds=[46])).mean():+.2f}')
    OUT[f'm6b_vs_code_{v}'] = dict(vs_3seed=x.mean(), p=wtest(x), vs_seed46=(100 * dd['VUS_PR'] - S(code, seeds=[46])).mean())
json.dump(OUT, open(Path(__file__).with_name('audit_final.json'), 'w'), default=lambda o: (None if o is None else float(o)), indent=1)
print('\nsaved audit_final.json')
