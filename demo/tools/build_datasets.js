// Builds data/datasets.json for the "Data" screen: per-series metadata of the paper's 64 TSB-AD series and of the
// 60-series external set, plus a KDD21 summary. Reads the raw TSB-AD-U CSVs (label column) and all_length.csv.
// usage: node tools/build_datasets.js <TSB-AD raw dir> <UCR all_length.csv> <m13 external csv dir>
const fs = require('fs'), path = require('path');
const [rawDir, kddFile, m13Dir] = process.argv.slice(2);
const out = path.join(__dirname, '..', 'data', 'datasets.json');
const bench = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'data', 'benchmark.json'), 'utf8'));
const files = fs.readdirSync(rawDir).filter(f => f.endsWith('.csv')).sort();

function seriesMeta(group) {
  const file = files[group - 1];
  const lines = fs.readFileSync(path.join(rawDir, file), 'utf8').trim().split(/\r?\n/).slice(1);
  let anom = 0, events = 0, prev = 0, missing = 0;
  for (const l of lines) {
    const parts = l.split(','), y = Math.round(+parts[parts.length - 1]);
    if (parts.some(p => p.trim() === '')) missing++;
    if (y === 1) { anom++; if (prev === 0) events++; }
    prev = y;
  }
  const bits = file.replace('.csv', '').split('_');
  return { group, file, family: bits[1], length: lines.length, n_train: +bits[bits.length - 3],
           first_anomaly: +bits[bits.length - 1], anomaly_points: anom, anomaly_ratio: anom / lines.length, events,
           rows_with_missing: missing };
}

const DEMO_TAGS = { 237: 'success', 276: 'success', 234: 'representative', 269: 'failure', 295: 'failure' };
const paper = bench.rows.map(r => Object.assign(seriesMeta(r.group), { split: r.split, demo: r.demo, tag: DEMO_TAGS[r.group] || null }))
  .sort((a, b) => a.group - b.group);

const ext = new Map();
for (const f of fs.readdirSync(m13Dir).filter(f => f.startsWith('m13_external_') && f.endsWith('.csv'))) {
  const L = fs.readFileSync(path.join(m13Dir, f), 'utf8').trim().split(/\r?\n/);
  const h = L[0].split(','), gi = h.indexOf('group');
  L.slice(1).forEach(l => { const g = +l.split(',')[gi]; if (g) ext.set(g, true); });
}
const external = [...ext.keys()].sort((a, b) => a - b).map(seriesMeta);

const K = fs.readFileSync(kddFile, 'utf8').trim().split(/\r?\n/).slice(1).map(l => l.split(','));
const num = i => K.map(r => +r[i]).sort((a, b) => a - b);
const stat = a => ({ min: a[0], median: a[Math.floor(a.length / 2)], max: a[a.length - 1] });
const kdd = { series: K.length, train_length: stat(num(1)), test_length: stat(num(2)), anomaly_length: stat(num(3)),
              period: stat(num(4)), events_per_series: 1 };

fs.writeFileSync(out, JSON.stringify({ paper, external, kdd21: kdd, built: new Date().toISOString().slice(0, 16) }, null, 1));
console.log('paper', paper.length, 'external', external.length, 'kdd21', kdd.series, '->', out);
