/* CoAD Lab — research workspace for the CoAD demo (spec: CoAD_Web_Demo_Error_Analysis_Spec.md,
   design: CoAD_Web_UI_Design.md). Reads precomputed data from data/ (precompute.py); event rules in rules.js. */
(function () {
  'use strict';
  const R = window.Rules;
  const $ = (s, el = document) => el.querySelector(s);
  const el = (tag, attrs = {}, html = '') => { const e = document.createElement(tag); Object.assign(e, attrs); if (html) e.innerHTML = html; return e; };
  const fmt = (v, d = 3) => v === null || v === undefined || Number.isNaN(v) ? 'N/A' : (+v).toFixed(d);
  const pct = v => v === null || v === undefined || Number.isNaN(v) ? 'N/A' : (100 * v).toFixed(1) + '%';
  const pts = v => v === null || v === undefined || Number.isNaN(v) ? 'N/A' : (100 * v).toFixed(2);
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const info = t => `<i class="info" title="${esc(t)}">i</i>`;
  const IL = window.Illus;
  // a picture with one short sentence, shown only in "Giải thích" mode
  const ex = (pic, text, cls = '') => `<details class="explain ${cls}"><summary>Cách đọc</summary><div class="exbody">${pic}<div class="txt">${text}</div></div></details>`;
  const exText = text => `<div class="explain">${text}</div>`;
  /* bring a detail box into view (it may sit below the charts on narrow screens) and flash it so the change is visible */
  function reveal(node) {
    if (!node) return;
    const r = node.getBoundingClientRect();
    if (r.top > innerHeight - 120 || r.bottom < 60) node.scrollIntoView({ behavior: 'smooth', block: 'start' });
    node.classList.remove('flash'); void node.offsetWidth; node.classList.add('flash');
  }

  const METHOD = {
    coad: { label: 'CoAD gốc', color: '#2563EB' },
    calibrated: { label: 'Our Improvement', color: '#7C3AED' },
    kmeans: { label: 'KMeansAD', color: '#B45309' },
  };
  const POLICY_LABEL = {
    train_mu3sd: 'Train μ + 3σ (normal calibration)', train_max: 'Max trên phần train',
    interactive: 'Tương tác', self_mu3sd: 'μ + 3σ trên chính score test',
  };
  const STAGES = {
    shared: [
      ['oe', 'A — xác suất OE (max time/freq)', 'xác suất 0–1, gán cho 8 điểm/patch, trung bình các window chồng lấp'],
      ['residual', 'Aʳ — xác suất residual classification', 'xác suất 0–1'],
      ['combined', 'A_c = (A + Aʳ)/2', 'xác suất 0–1'],
      ['rec_error', '|x − x̂| reconstruction error', 'đơn vị của input (min-max theo train)'],
    ],
    coad: [
      ['c_raw', 'Nhánh phân loại: |A_c − mean_train|', 'trước làm mượt'],
      ['r_raw', 'Nhánh tái tạo: | |x − x̂| − mean_train |', 'trước làm mượt'],
      ['coad_pre', 'Gộp (c + r)/2 trước làm mượt', 'thang của CoAD gốc'],
      ['c_smooth', 'Nhánh phân loại sau làm mượt', 'moving average, cửa sổ = 1 chu kỳ'],
      ['r_smooth', 'Nhánh tái tạo sau làm mượt', 'moving average, cửa sổ = 1 chu kỳ'],
      ['coad_final', 'Score cuối CoAD gốc', 'thang của CoAD gốc'],
    ],
    calibrated: [
      ['cal_c', 'Nhánh phân loại / σ_c (train)', 'đơn vị σ của phần train'],
      ['cal_r', 'Nhánh tái tạo / σ_r (train)', 'đơn vị σ của phần train'],
      ['cal_pre', 'Gộp Our Improvement trước làm mượt', 'đơn vị σ'],
      ['cal_final', 'Score cuối Our Improvement', 'đơn vị σ'],
    ],
  };
  const STAGE_LABEL = Object.fromEntries([...STAGES.shared, ...STAGES.coad, ...STAGES.calibrated].map(s => [s[0], s[1]]));

  const store = {
    get(k, d) { try { const v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) { /* storage unavailable */ } },
  };

  const ST = {
    bench: null, cache: {}, group: null, data: null, screen: 'series',
    cfg: { methods: { coad: true, calibrated: true, kmeans: false }, policy: 'train_mu3sd', inter: {} },
    draft: null, view: null, tab: 'events', selEvent: null, inspT: null, inspWin: null, inspPatch: null,
    ea: { filter: 'all', key: null, tab: 'obs' }, res: { metric: 'VUS_PR', family: 'all', split: 'confirmation', sort: 'delta' },
    results: {}, upload: null,
  };
  ST.draft = JSON.parse(JSON.stringify(ST.cfg));

  /* ---------- data ---------- */
  async function loadJSON(url) { const r = await fetch(url); if (!r.ok) throw new Error(url + ': ' + r.status); return r.json(); }

  async function loadSeries(group) {
    if (ST.cache[group]) return ST.cache[group];
    const meta = await loadJSON(`data/series_${group}.json`);
    const buf = await (await fetch(`data/series_${group}.bin`)).arrayBuffer();
    const all = new Float32Array(buf);
    const arr = {};
    for (const [k, f] of Object.entries(meta.fields)) arr[k] = all.subarray(f.offset, f.offset + f.length);
    const d = { meta, arr, n: meta.length, overview: decimate(arr.input, 3000) };
    ST.cache[group] = d;
    return d;
  }

  /* min/max per bucket (display only: keeps outliers visible; metrics never use it) */
  function decimate(y, buckets) {
    const n = y.length, step = Math.max(1, Math.ceil(n / buckets)), xs = [], ys = [];
    for (let s = 0; s < n; s += step) {
      let lo = s, hi = s;
      for (let i = s; i < Math.min(n, s + step); i++) { if (y[i] < y[lo]) lo = i; if (y[i] > y[hi]) hi = i; }
      const a = Math.min(lo, hi), b = Math.max(lo, hi);
      xs.push(a); ys.push(y[a]); if (b !== a) { xs.push(b); ys.push(y[b]); }
    }
    return { x: xs, y: ys, step };
  }

  function methodsAvailable(d) { return Object.keys(METHOD).filter(m => d.meta.methods[m]); }
  function activeMethods() { return methodsAvailable(ST.data).filter(m => ST.cfg.methods[m]); }
  function score(m) { return ST.data.arr[ST.data.meta.methods[m].score]; }

  function thresholdOf(m, cfg = ST.cfg) {
    const t = ST.data.meta.methods[m].thresholds;
    if (cfg.policy === 'interactive' && cfg.inter[m] !== undefined) return { value: cfg.inter[m], policy: 'interactive' };
    if (t[cfg.policy] !== undefined && t[cfg.policy] !== null) return { value: t[cfg.policy], policy: cfg.policy };
    return { value: t.self_mu3sd, policy: 'self_mu3sd' };
  }

  function compute() {
    const d = ST.data, label = d.arr.label, vf = d.meta.valid_from;
    ST.results = {};
    methodsAvailable(d).forEach(m => {
      const th = thresholdOf(m), s = score(m), pred = R.predict(s, th.value, vf);
      ST.results[m] = { th, pred, ev: R.evaluate(label, pred, vf, s) };
    });
  }

  /* ---------- plotting helpers ---------- */
  const BASE = {
    margin: { l: 56, r: 16, t: 8, b: 28 }, paper_bgcolor: '#fff', plot_bgcolor: '#fff', hovermode: 'x',
    font: { family: 'Be Vietnam Pro, Segoe UI, sans-serif', size: 12, color: '#0F172A' },
    legend: { orientation: 'h', y: 1.12, x: 0, font: { size: 12 } }, dragmode: 'zoom',
    modebar: { orientation: 'v', bgcolor: 'rgba(255,255,255,.85)' },   // vertical, so it never sits on the legend row
  };
  const CONF = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ['lasso2d', 'select2d', 'autoScale2d'],
                 toImageButtonOptions: { format: 'png', scale: 2 } };
  const xs = n => { const a = new Float32Array(n); for (let i = 0; i < n; i++) a[i] = i; return a; };
  let linked = [], syncing = false, cursorRAF = null;

  /* Display thinning (spec 12): charts get at most ~2 points per bucket of the visible range, never the full 90k
     points. 'minmax' keeps both extremes of each bucket so outliers stay visible; 'max' keeps a bucket at its maximum
     so a 1-point anomaly or prediction never disappears. Metrics and events always use the full arrays. */
  const MAXB = 2000;
  function visRange(n) {
    if (!ST.view) return [0, n];
    const s = ST.view[1] - ST.view[0] + 1;
    return [Math.max(0, ST.view[0] - s), Math.min(n, ST.view[1] + s + 1)];
  }
  function thin(arr, mode = 'minmax', from = 0) {
    const n = arr.length, [a, b] = visRange(n), step = Math.max(1, Math.ceil((b - a) / MAXB)), x = [], y = [];
    if (step === 1) { for (let i = a; i < b; i++) { x.push(i); y.push(i < from ? null : arr[i]); } return { x, y }; }
    for (let s = a; s < b; s += step) {
      const e = Math.min(b, s + step);
      let lo = -1, hi = -1;
      for (let i = Math.max(s, from); i < e; i++) { if (lo < 0 || arr[i] < arr[lo]) lo = i; if (hi < 0 || arr[i] > arr[hi]) hi = i; }
      if (hi < 0) { x.push(s); y.push(null); continue; }
      if (mode === 'max') { x.push(s, e - 1); y.push(arr[hi], arr[hi]); continue; }
      const p = Math.min(lo, hi), q = Math.max(lo, hi);
      x.push(p); y.push(arr[p]); if (q !== p) { x.push(q); y.push(arr[q]); }
    }
    return { x, y };
  }
  function line(arr, extra, from = 0) { const t = thin(arr, 'minmax', from); return Object.assign({ x: t.x, y: t.y, type: 'scattergl', mode: 'lines' }, extra); }

  function bandTrace(mask, yref, name, color, show = true) {
    // filled step area 0/1 on an overlaying axis
    const t = thin(mask, 'max');
    return { x: t.x, y: t.y.map(v => (v > 0.5 ? 1 : 0)), type: 'scattergl', mode: 'lines',
             line: { width: 0, shape: 'hv' }, fill: 'tozeroy', fillcolor: color, yaxis: yref, name, hoverinfo: 'skip', showlegend: show };
  }

  function linkChart(div) {
    linked.push(div);
    div.on('plotly_relayout', ev => {
      if (syncing) return;
      let r = null;
      if (ev['xaxis.range[0]'] !== undefined) r = [ev['xaxis.range[0]'], ev['xaxis.range[1]']];
      else if (ev['xaxis.range']) r = ev['xaxis.range'];
      else if (ev['xaxis.autorange']) r = null; else return;
      // Plotly can emit relayout after setView's synchronous guard is gone: ignore echoes of the current view,
      // otherwise linked charts relayout each other forever and the page freezes
      const rr = r ? [Math.max(0, Math.round(r[0])), Math.min(ST.data.n - 1, Math.round(r[1]))] : null;
      if ((rr === null && ST.view === null) || (rr && ST.view && rr[0] === ST.view[0] && rr[1] === ST.view[1])) return;
      setView(r ? [Math.max(0, Math.round(r[0])), Math.min(ST.data.n - 1, Math.round(r[1]))] : null, div);
    });
    // synced vertical cursor (design 4.1): a DOM line moved by pixels, so no chart is relayouted on mouse move
    div.on('plotly_hover', ev => {
      const x = ev.points && ev.points[0] ? ev.points[0].x : null;
      if (x === null) return;
      if (cursorRAF) cancelAnimationFrame(cursorRAF);
      cursorRAF = requestAnimationFrame(() => linked.forEach(g => moveCursor(g, x)));
    });
    div.on('plotly_unhover', () => linked.forEach(g => moveCursor(g, null)));
    div.on('plotly_click', ev => { if (ev.points && ev.points[0] && div.dataset.inspect) openInspector(Math.round(ev.points[0].x)); });
  }
  function moveCursor(g, x) {
    if (!g.isConnected || !g._fullLayout || !g._fullLayout.xaxis) return;
    let c = g.querySelector(':scope > .hcursor');
    if (!c) { g.style.position = 'relative'; c = el('div', { className: 'hcursor' }); g.append(c); }
    const xa = g._fullLayout.xaxis, px = x === null ? null : xa.l2p(x) + g._fullLayout._size.l;
    if (px === null || px < g._fullLayout._size.l || px > g._fullLayout._size.l + xa._length) { c.style.display = 'none'; return; }
    c.style.display = 'block'; c.style.left = px + 'px';
  }
  function cursorShapes(g, x) {
    const base = (g._baseShapes || []);
    return x === null ? base : base.concat([{ type: 'line', xref: 'x', yref: 'paper', x0: x, x1: x, y0: 0, y1: 1, line: { color: '#64748B', width: 1, dash: 'dot' } }]);
  }
  function setView(r, source) {
    ST.view = r;
    syncing = true;
    const upd = r ? { 'xaxis.range': [r[0], r[1]], 'xaxis.autorange': false } : { 'xaxis.autorange': true };
    linked.forEach(g => { if (g !== source && g.isConnected) Plotly.relayout(g, upd); });
    syncing = false;
    updateViewing();
    drawOverviewWindow();
    refreshCharts();
  }
  /* after a zoom, rebuild the thinned traces for the new range (full resolution once the range is small) */
  let refreshTimer = null;
  function refreshCharts() {
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(() => linked.forEach(g => {
      if (!g.isConnected || !g._build) return;
      const lay = Object.assign({}, g.layout);
      lay.xaxis = Object.assign({}, g.layout.xaxis, ST.view ? { range: ST.view.slice(), autorange: false } : { autorange: true });
      Plotly.react(g, g._build(), lay);
    }), 150);
  }
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(() => {
    document.querySelectorAll('.js-plotly-plot').forEach(g => { try { Plotly.relayout(g, {}); } catch (e) { /* not drawn yet */ } });
  });
  function plot(div, build, layout, opts = {}) {
    const traces = typeof build === 'function' ? build() : build;
    div._build = typeof build === 'function' ? build : null;
    const lay = Object.assign({}, BASE, layout);
    lay.xaxis = Object.assign({ showspikes: false, zeroline: false }, layout.xaxis || {});
    if (ST.view && opts.link !== false) { lay.xaxis.range = ST.view.slice(); lay.xaxis.autorange = false; }
    div._baseShapes = lay.shapes || [];
    Plotly.newPlot(div, traces, lay, CONF);
    if (opts.link !== false) linkChart(div);
    if (opts.inspect) div.dataset.inspect = '1';
  }

  /* ---------- toolbar ---------- */
  function renderToolbar() {
    const tb = $('#toolbar');
    if (ST.screen === 'results') { tb.innerHTML = `<span class="meta">Kết quả benchmark tính sẵn — <b>${ST.bench ? ST.bench.rows.length : 0}</b> series</span>`; return; }
    if (ST.screen === 'data') { tb.innerHTML = '<span class="meta">Dữ liệu: TSB-AD-U · 64 series của paper và 60 series nhóm chạy thêm (12 family khác)</span>'; return; }
    if (!ST.data) { tb.innerHTML = '<span class="meta">Chưa chọn dữ liệu</span>'; return; }
    const m = ST.data.meta, ms = activeMethods().map(k => `<span class="swatch" style="background:${METHOD[k].color}"></span> ${METHOD[k].label}`).join(' &nbsp; ');
    const pol = ST.cfg.policy === 'interactive' ? '<span class="badge inter">Interactive</span>' : esc(POLICY_LABEL[ST.cfg.policy]);
    tb.innerHTML = `
      <span class="meta" title="${esc(m.file)}">Series <b>${esc(m.family)} · ${m.group}</b></span>
      <span class="meta">${ms || '—'}</span>
      <span class="meta">Ngưỡng: ${pol}</span>
      <span class="badge pre" title="Kết quả tính sẵn từ checkpoint ${esc(m.run.checkpoint_id)}">Tính sẵn</span>
      <span class="meta help" title="${esc(m.run.inference_scope)} · thiết bị: ${esc(m.run.inference_device || '')} · checkpoint ${esc(m.run.checkpoint_id)}"><b>${fmt(m.run.inference_seconds, 1)} s</b></span>
      <span class="spacer" style="flex:1"></span>
      <details class="xmenu"><summary class="btn small">Xuất ▾</summary><div class="xm">
        <button id="tbCsv">Kết quả (CSV + cấu hình)</button><button id="tbJson">Chỉ cấu hình (JSON)</button><button id="tbReload">↻ Tải lại kết quả</button></div></details>`;
    $('#tbReload').onclick = () => { delete ST.cache[ST.group]; selectSeries(ST.group); };
    $('#tbCsv').onclick = exportCSV; $('#tbJson').onclick = exportJSON;
  }

  /* ---------- sidebar ---------- */
  const METHOD_DESC = { coad: 'Bản gốc của paper', calibrated: 'Bản cải tiến (thử nghiệm)', kmeans: 'Baseline đơn giản: phân cụm' };
  const POLICY_SEG = [
    ['train_mu3sd', 'Chuẩn', 'Báo động khi điểm cao hơn mức bình thường (trung bình + 3 độ lệch chuẩn của phần đầu chuỗi). Không dùng nhãn.'],
    ['train_max', 'Chặt', 'Chỉ báo động khi điểm vượt mức cao nhất từng thấy ở phần bình thường: ít báo nhầm hơn, dễ bỏ sót hơn.'],
    ['interactive', 'Tự chỉnh', 'Kéo thanh trượt để thử ngưỡng khác. Chỉ để khám phá, không thay kết quả chính thức.'],
  ];
  function renderSidebar() {
    const box = $('#cfgSeries'), demo = ST.bench.rows.filter(r => r.demo);
    box.innerHTML = demo.map(r => {
      const [icon, name] = IL.family(r.family), tag = DEMO_TAG[r.group], missing = ST.cache[r.group] === null;
      const tagText = tag ? { success: 'Impr. tốt hơn', failure: 'Impr. kém (seed demo)', representative: 'Đại diện' }[tag] : '';
      return `<button class="srow1 ${r.group === ST.group ? 'sel' : ''}" data-g="${r.group}" ${missing ? 'disabled title="Chưa có dữ liệu tính sẵn"' : ''}>
        <span class="nm">${r.family} ${r.group}</span><span class="sub">${name}</span>${tag ? `<i class="tdot ${tag}" title="${tagText}"></i>` : ''}</button>`;
    }).join('') + `<div class="tlegend"><span><i class="tdot success"></i>Impr. tốt hơn</span><span><i class="tdot failure"></i>Impr. kém</span><span><i class="tdot representative"></i>Đại diện</span></div>`;
    box.querySelectorAll('.srow1').forEach(b => b.onclick = () => { if (+b.dataset.g !== ST.group) { b.classList.add('loading'); selectSeries(+b.dataset.g); } });
    if (!ST.data) { $('#cfgMethods').innerHTML = ''; return; }
    const m = ST.data.meta;
    $('#cfgMethods').innerHTML = '<div class="mchips">' + methodsAvailable(ST.data).map(k => `<button class="mchip ${ST.draft.methods[k] ? 'on' : ''}" data-m="${k}" style="--c:${METHOD[k].color}" title="${METHOD_DESC[k]}">${METHOD[k].label}</button>`).join('') + '</div>';
    $('#cfgMethods').querySelectorAll('.mchip').forEach(b => b.onclick = () => { ST.draft.methods[b.dataset.m] = !ST.draft.methods[b.dataset.m]; apply(); renderSidebar(); });
    const c = m.calibration;
    $('#advanced').innerHTML = `
      <dt>Cửa sổ T</dt><dd class="num">${m.window}</dd><dt>Patch P</dt><dd class="num">${m.patch}</dd>
      <dt>Stride suy luận</dt><dd class="num">${m.inference_stride}</dd><dt>Stride train</dt><dd class="num">${m.train_stride}</dd>
      <dt>Làm mượt</dt><dd>MA ${m.period} điểm</dd><dt>Warm-up</dt><dd class="num">${m.valid_from} điểm đầu</dd>
      <dt>σ_r (train)</dt><dd class="num">${fmt(c.sd_reconstruction, 4)}</dd><dt>σ_c (train)</dt><dd class="num">${fmt(c.sd_classification, 4)}</dd>
      <dt>Tiền xử lý</dt><dd>min-max theo train</dd>`;
  }
  function renderSliders() {
    const box = $('#interactiveBox');
    if (!box) return;
    box.classList.toggle('hidden', ST.draft.policy !== 'interactive');
    if (ST.draft.policy !== 'interactive' || !ST.data) return;
    const s = $('#sliders'); s.innerHTML = '';
    methodsAvailable(ST.data).forEach(m => {
      const sc = score(m), vf = ST.data.meta.valid_from;
      let mx = 0; for (let i = vf; i < sc.length; i++) if (sc[i] > mx) mx = sc[i];
      const cur = ST.draft.inter[m] !== undefined ? ST.draft.inter[m] : thresholdOf(m, { policy: 'train_mu3sd', inter: {} }).value;
      const row = el('div', {}, `<div class="note"><span class="swatch" style="background:${METHOD[m].color}"></span> ${METHOD[m].label}: ngưỡng <span class="num" id="sv-${m}">${fmt(cur, 4)}</span></div>`);
      const inp = el('input', { type: 'range', min: 0, max: mx, step: mx / 500, value: cur, style: 'width:100%' });
      inp.oninput = () => { ST.draft.inter[m] = +inp.value; $('#sv-' + m).textContent = fmt(+inp.value, 4); markPending(); };
      ST.draft.inter[m] = cur;
      row.append(inp); s.append(row);
    });
  }
  function markPending() { if (!$('#applyBar')) return; const dirty = JSON.stringify(ST.draft) !== JSON.stringify(ST.cfg); $('#pendingNote').classList.toggle('hidden', !dirty); $('#applyBar').classList.toggle('dirty', dirty); }
  function apply() { ST.cfg = JSON.parse(JSON.stringify(ST.draft)); markPending(); if (ST.data) { compute(); renderAll(); } }

  /* ---------- screen: series ---------- */
  function updateViewing() {
    const v = $('#viewing');
    if (!v || !ST.data) return;
    const r = ST.view || [0, ST.data.n - 1];
    v.querySelector('.range').textContent = `${r[0].toLocaleString('vi')}–${r[1].toLocaleString('vi')}`;
  }

  const STATUS_VI = { hit: 'phát hiện', partial: 'một phần', missed: 'bỏ sót' };
  /* one card per method: the numbers a reader needs first, rules and units in the (i) tooltips */
  function methodCard(m) {
    const ev = ST.results[m].ev, met = ST.data.meta.methods[m].metrics, th = ST.results[m].th;
    const dots = ev.events.map((e, k) => `<i class="${e.status} ${ST.selEvent && ST.selEvent.id === e.id ? 'sel' : ''}" data-k="${k}" title="${e.id}: ${e.a}–${e.b - 1} · ${STATUS_VI[e.status]} · coverage ${pct(e.coverage)}"></i>`).join('');
    return `<div class="mcard" style="--c:${METHOD[m].color}"><div class="hd"><span class="swatch" style="background:${METHOD[m].color}"></span>${METHOD[m].label}
        <span class="th">ngưỡng <span class="num">${fmt(th.value, 4)}</span> ${info(POLICY_LABEL[th.policy])}</span></div>
      <div class="row" title="${esc(R.RULES.hit + ' ' + R.RULES.false_alarm + ' VUS-PR không phụ thuộc ngưỡng (evaluator TSB-AD); AUC-PR ' + pts(met.auc_prc) + '.')}"><div><span class="big">${ev.nHit}<small> / ${ev.events.length}</small></span><span class="lab">bắt được</span></div>
        <div><span class="big">${ev.falseAlarms.length}</span><span class="lab">báo sai</span></div>
        <div><span class="big">${pts(met.VUS_PR)}</span><span class="lab">VUS-PR</span></div></div>
      ${ev.events.length ? `<div class="dots" data-m="${m}">${dots}</div>` : ''}
      <div class="lab" style="margin-top:4px">Gắn cờ ${ev.flagged.toLocaleString('vi')} điểm (${pct(ev.flaggedShare)}) · ${ev.intervals.length} đoạn · P ${pct(ev.point.precision)} · R ${pct(ev.point.recall)} ${info(R.RULES.predicted_interval + ' ' + R.RULES.point)}</div></div>`;
  }
  /* "what am I looking at": the series in plain words + the question the screen answers */
  function contextCard() {
    const m = ST.data.meta, [icon, name, desc] = IL.family(m.family), ms = activeMethods();
    const nEv = ms.length ? ST.results[ms[0]].ev.events.length : 0;
    const anyMiss = ms.some(k => ST.results[k].ev.nMissed > 0);
    return `<div class="context explain" style="margin:0 0 12px"><span class="ico">${icon}</span><div style="flex:1">
      <h3>${name} · chuỗi ${m.group}</h3><p class="q">Mô hình có bắt được ${nEv} đoạn bất thường (nền cam) mà ít báo nhầm không?</p>
      <p>${desc}. ${m.length.toLocaleString('vi')} điểm đo; ${m.n_train.toLocaleString('vi')} điểm đầu là phần bình thường để mô hình học.</p></div></div>`;
  }
  function selectEvent(e, kind = 'gt', method) {
    const ctx = Math.max(ST.data.meta.period * 2, e.b - e.a);
    ST.selEvent = kind === 'gt' ? { a: e.a, b: e.b, id: e.id, kind } : { a: e.a, b: e.b, id: method + '-' + e.id, kind, method };
    setView([Math.max(0, e.a - ctx), Math.min(ST.data.n - 1, e.b + ctx)], null);
    document.querySelectorAll('.dots i').forEach(i => { const ev = ST.results[i.parentNode.dataset.m].ev.events[+i.dataset.k]; i.classList.toggle('sel', ev.id === ST.selEvent.id); });
  }

  function renderSeries() {
    const root = $('#screen-series');
    linked = [];
    if (!ST.data) { root.innerHTML = ST.loadError ? `<div class="panel"><div class="empty">${esc(ST.loadError)}</div></div>` : '<div class="panel"><div class="empty">Chọn một series mẫu ở thanh bên trái để xem kết quả.</div></div>'; return; }
    const ms = activeMethods();
    root.innerHTML = `
      ${ST.data.meta.family === 'CSV' ? journeyHTML(4, ST.story === 4 ? 4 : 3) + (ST.story === 4 ? storyResultsHTML() : '') : ''}${ST.data.meta.family === 'CSV' ? '' : contextCard()}
      <div class="mcards">${ms.map(methodCard).join('') || '<span class="note">Chọn ít nhất một phương pháp.</span>'}</div>
      <div class="split"><div>
      <div class="panel"><div class="ph viewbar" id="viewing" style="margin:0"><h2>Tổng quan</h2>
        <span>Đang xem <b class="range num"></b></span><button class="btn small" id="fullBtn">Toàn chuỗi</button>
        <span class="legend-mini" style="margin-left:auto"><span style="--c:#16A34A">phát hiện</span><span style="--c:#F59E0B">một phần</span><span style="--c:#E11D48">bỏ sót</span></span>
        ${info('Kéo chọn trên dải này để phóng to; ô màu trên thẻ phương pháp = từng sự kiện, bấm để nhảy tới. Click biểu đồ để xem CoAD xử lý điểm đó ở khung bên dưới.')}</div>
        <div class="chart" id="cOverview" style="height:64px"></div></div>
      <div class="panel"><div class="ph"><h2>Input · ground truth · prediction</h2></div>
        <div class="chart" id="cInput" style="height:280px"></div></div>
      <div class="panel"><div class="ph"><h2>Score + ngưỡng</h2>${info('Mỗi phương pháp một trục, ngưỡng theo thang của chính nó; vùng gạch = warm-up (chưa có score hợp lệ).')}<span class="note" style="margin-left:auto">Vạch đứt = ngưỡng báo động: trung bình + 3σ của score trên phần train, không dùng nhãn</span></div>
        <div class="chart" id="cScore" style="height:${Math.max(200, 150 * ms.length)}px"></div></div>
      <div class="panel"><div class="ph"><h2>Input vs reconstruction</h2>${info('Reconstruction của CoAD (lượt tái tạo cuối, code mode), cùng thang với input.')}</div>
        <div class="chart" id="cRecon" style="height:200px"></div></div>
      <div class="panel" id="inspPanel"><div class="ph"><h2>Inspector · CoAD xử lý một cửa sổ như thế nào</h2>${info('Click một điểm trên biểu đồ phía trên (hoặc một ô sự kiện ở thẻ phương pháp) để xem cửa sổ chứa điểm đó đi qua từng bước của CoAD.')}</div>
        <div class="pb" id="inspBox"></div></div>
      </div></div>`;
    $('#fullBtn').onclick = () => setView(null, null);
    bindJourney(root);
    root.querySelectorAll('.dots i').forEach(i => i.onclick = () => {
      const e = ST.results[i.parentNode.dataset.m].ev.events[+i.dataset.k];
      selectEvent(e);
      ST.inspT = R.peakIn(ST.data.arr.combined, e.a, e.b).index; ST.inspWin = null; ST.inspPatch = null; renderSeriesTab();
    });
    drawOverview(); drawInput($('#cInput'), ms); drawScores($('#cScore'), ms); drawRecon($('#cRecon'));
    if (ST.inspT === null || ST.inspT === undefined || ST.inspT >= ST.data.n) {      // start on the first missed (else first) event
      const evs = ms.length ? ST.results[ms[0]].ev.events : [], e = evs.find(x => x.status === 'missed') || evs[0];
      ST.inspT = e ? R.peakIn(ST.data.arr.combined, e.a, e.b).index : Math.floor(ST.data.n / 2); ST.inspWin = null; ST.inspPatch = null;
    }
    updateViewing(); renderSeriesTab();
    if ($('#upGo')) bindUploadPanel();
  }

  function drawOverview() {
    const d = ST.data, div = $('#cOverview');
    const gt = R.segments(d.arr.label, 0).map(([a, b]) => ({ type: 'rect', xref: 'x', yref: 'paper', x0: a, x1: b, y0: 0, y1: 1, fillcolor: 'rgba(251,146,60,.35)', line: { width: 0 } }));
    Plotly.newPlot(div, [{ x: d.overview.x, y: d.overview.y, type: 'scatter', mode: 'lines', line: { color: '#334155', width: 1 }, hoverinfo: 'x' }],
      Object.assign({}, BASE, { margin: { l: 56, r: 16, t: 4, b: 18 }, dragmode: 'select', selectdirection: 'h', showlegend: false,
        xaxis: { fixedrange: true }, yaxis: { fixedrange: true, showticklabels: false }, shapes: gt }), { displayModeBar: false, responsive: true });
    div._gtShapes = gt;
    div.on('plotly_selected', ev => { if (ev && ev.range) setView([Math.max(0, Math.round(ev.range.x[0])), Math.min(d.n - 1, Math.round(ev.range.x[1]))], null); });
    drawOverviewWindow();
  }
  function drawOverviewWindow() {
    const div = $('#cOverview');
    if (!div || !div._gtShapes) return;
    const shapes = div._gtShapes.slice();
    if (ST.view) shapes.push({ type: 'rect', xref: 'x', yref: 'paper', x0: ST.view[0], x1: ST.view[1], y0: 0, y1: 1, fillcolor: 'rgba(37,99,235,.12)', line: { color: '#2563EB', width: 1 } });
    Plotly.relayout(div, { shapes });
  }

  function predTracks(ms, d, yref) {
    return ms.map((m, k) => {
      const t = thin(ST.results[m].pred, 'max'), y = t.y.map(v => (v ? k + 1 : null));
      return { x: t.x, y, type: 'scattergl', mode: 'lines', line: { color: METHOD[m].color, width: 7 }, yaxis: yref,
               name: 'Prediction ' + METHOD[m].label, connectgaps: false, hovertemplate: `${METHOD[m].label}: cảnh báo<extra></extra>` };
    });
  }

  function drawInput(div, ms, opts = {}) {
    const d = ST.data, n = d.n, train = new Float32Array(n).fill(0);
    for (let i = 0; i < d.meta.n_train; i++) train[i] = 1;
    const traces = () => [
      bandTrace(train, 'y2', 'Phần train', 'rgba(148,163,184,.13)'),
      bandTrace(d.arr.label, 'y2', 'Ground truth', 'rgba(251,146,60,.35)'),
      line(d.arr.input, { line: { color: '#334155', width: 1 }, name: 'Input', hovertemplate: 't=%{x}<br>input=%{y:.4f}<extra></extra>' }),
      ...predTracks(ms, d, 'y3'),
    ];
    plot(div, traces, Object.assign({
      margin: Object.assign({}, BASE.margin, { l: 92 }),
      yaxis: { domain: [0.28, 1], title: { text: 'input', font: { size: 11 } } },
      yaxis2: { domain: [0.28, 1], overlaying: 'y', range: [0, 1], visible: false, fixedrange: true },
      yaxis3: { domain: [0, 0.2], range: [0.4, ms.length + 0.6], tickvals: ms.map((_, k) => k + 1), ticktext: ms.map(m => METHOD[m].label), fixedrange: true, showgrid: false, automargin: true },
    }, opts.layout || {}), { inspect: !opts.noInspect, link: opts.link });
  }

  function drawScores(div, ms, opts = {}) {
    const d = ST.data, n = d.n, vf = d.meta.valid_from, layout = { showlegend: true, shapes: [] };
    const gap = 0.06, h = (1 - gap * (ms.length - 1)) / Math.max(1, ms.length);
    const traces = () => ms.flatMap((m, k) => {
      const ya = k === 0 ? 'y' : 'y' + (k + 1), out = [];
      out.push(line(score(m), { line: { color: METHOD[m].color, width: 1.2 }, yaxis: ya, name: 'Score ' + METHOD[m].label,
                                hovertemplate: `t=%{x}<br>${METHOD[m].label} score=%{y:.4f}<extra></extra>` }, vf));
      const pre = d.meta.methods[m].pre;
      if (opts.pre && pre) out.push(line(d.arr[pre], { line: { color: METHOD[m].color, width: 1, dash: 'dot' }, opacity: 0.55, yaxis: ya,
        name: 'Trước làm mượt ' + METHOD[m].label, hovertemplate: `t=%{x}<br>trước làm mượt=%{y:.4f}<extra></extra>` }, vf));
      return out;
    });
    ms.forEach((m, k) => {
      const ya = k === 0 ? 'y' : 'y' + (k + 1), key = k === 0 ? 'yaxis' : 'yaxis' + (k + 1);
      const top = 1 - k * (h + gap), dom = [Math.max(0, top - h), top];
      layout[key] = { domain: dom, title: { text: METHOD[m].label, font: { size: 11, color: METHOD[m].color } }, zeroline: false, automargin: true };
      const th = ST.results[m].th.value;
      layout.shapes.push({ type: 'line', xref: 'paper', x0: 0, x1: 1, yref: ya, y0: th, y1: th, line: { color: '#64748B', width: 1, dash: 'dash' } });
      layout.shapes.push({ type: 'rect', xref: 'x', x0: 0, x1: vf, yref: ya + ' domain', y0: 0, y1: 1, fillcolor: 'rgba(100,116,139,.12)', line: { width: 0 } });
    });
    layout.xaxis = { anchor: ms.length > 1 ? 'y' + ms.length : 'y' };     // x ticks under the bottom subplot
    layout.margin = Object.assign({}, BASE.margin, { l: 64 });
    if (opts.layout) Object.assign(layout, opts.layout);
    plot(div, traces, layout, { inspect: !opts.noInspect, link: opts.link });
  }

  function drawRecon(div, opts = {}) {
    const d = ST.data, n = d.n;
    const traces = () => [
      bandTrace(d.arr.label, 'y2', 'Ground truth', 'rgba(251,146,60,.30)', false),
      line(d.arr.input, { line: { color: '#334155', width: 1 }, name: 'Input', hovertemplate: 't=%{x}<br>input=%{y:.4f}<extra></extra>' }),
      line(d.arr.reconstruction, { line: { color: '#0F766E', width: 1.2, dash: 'dash' }, name: 'Reconstruction', hovertemplate: 'reconstruction=%{y:.4f}<extra></extra>' }),
      ...(opts.error ? [line(d.arr.rec_error, { line: { color: '#0F766E', width: 1 }, yaxis: 'y3', name: '|x − x̂|', hovertemplate: '|x − x̂|=%{y:.4f}<extra></extra>' })] : []),
    ];
    const layout = { yaxis: { domain: opts.error ? [0.38, 1] : [0, 1] }, yaxis2: { domain: opts.error ? [0.38, 1] : [0, 1], overlaying: 'y', range: [0, 1], visible: false, fixedrange: true } };
    if (opts.error) layout.yaxis3 = { domain: [0, 0.3], title: { text: '|x − x̂|', font: { size: 11 } } };
    if (opts.layout) Object.assign(layout, opts.layout);
    plot(div, traces, layout, { inspect: !opts.noInspect, link: opts.link });
  }

  function renderSeriesTab() {
    const box = $('#inspBox');
    if (box) renderInspector(box);
  }

  /* trace tab: event navigator + evidence bars */
  function renderTraceTab(box) {
    const ms = activeMethods(), evs = ms.length ? ST.results[ms[0]].ev.events : [];
    if (!ST.selEvent) { const e = evs.find(x => x.status === 'missed') || evs[0]; if (e) selectEvent(e); }
    if (!ST.selEvent) { box.innerHTML = '<div class="empty">Chuỗi không có sự kiện.</div>'; return; }
    const S = ST.selEvent, k = S.kind === 'gt' ? evs.findIndex(e => e.id === S.id) : -1;
    const canCompare = S.kind === 'gt' && ST.results.coad && ST.results.calibrated;
    box.innerHTML = `<div class="evnav">
        <button class="btn small" id="evPrev" ${k <= 0 ? 'disabled' : ''} aria-label="Sự kiện trước">‹</button>
        <div class="who"><b>${esc(S.kind === 'gt' ? S.id : 'Báo sai ' + S.id)}</b><small>${S.a.toLocaleString('vi')}–${(S.b - 1).toLocaleString('vi')} · ${S.b - S.a} điểm</small></div>
        <button class="btn small" id="evNext" ${k < 0 || k >= evs.length - 1 ? 'disabled' : ''} aria-label="Sự kiện sau">›</button>
        ${canCompare ? '<button class="btn small primary" id="toCase">So sánh →</button>' : ''}</div>` + traceHTML(S);
    const step = d => { selectEvent(evs[k + d]); renderTraceTab(box); };
    if (k > 0) $('#evPrev').onclick = () => step(-1);
    if (k >= 0 && k < evs.length - 1) $('#evNext').onclick = () => step(1);
    if (canCompare) $('#toCase').onclick = () => { ST.ea.filter = 'all'; ST.ea.key = 'gt-' + S.id; go('errors'); };
  }

  /* config tab: readable groups; the raw JSON stays one click away */
  function configHTML() {
    const m = ST.data.meta, ms = activeMethods(), row = (k, v) => `<dt>${k}</dt><dd>${v}</dd>`;
    return `<div class="cfgcards">
      <div class="cfgc"><h4>Dữ liệu</h4><dl class="kv">${row('Series', `${m.family} · ${m.group}`)}${row('Số điểm', m.length.toLocaleString('vi'))}
        ${row('Phần học', '0–' + (m.n_train - 1).toLocaleString('vi'))}${row('Tiền xử lý', 'min–max theo phần học')}</dl></div>
      <div class="cfgc"><h4>Mô hình</h4><dl class="kv">${row('Checkpoint', `<code>${esc(m.run.checkpoint_id)}</code>`)}${row('Kết quả', 'tính sẵn')}
        ${row('Cửa sổ / patch', m.window + ' / ' + m.patch)}${row('Làm mượt', 'trung bình trượt ' + m.period + ' điểm')}${row('Thời gian suy luận', fmt(m.run.inference_seconds, 1) + ' s')}</dl></div>
      <div class="cfgc"><h4>Ngưỡng</h4><dl class="kv">${ms.map(k => row(`<span class="swatch" style="background:${METHOD[k].color}"></span> ${METHOD[k].label}`,
        `${fmt(ST.results[k].th.value, 4)}<br><span class="note">${esc(POLICY_LABEL[ST.results[k].th.policy])}</span>`)).join('')}</dl></div>
      <div class="cfgc"><h4>Metric (không phụ thuộc ngưỡng)</h4><dl class="kv">${ms.map(k => row(METHOD[k].label, `VUS-PR ${pts(m.methods[k].metrics.VUS_PR)}<br><span class="note">AUC-PR ${pts(m.methods[k].metrics.auc_prc)}</span>`)).join('')}</dl></div></div>
      <details class="more"><summary>Quy tắc sự kiện</summary><ul class="note" style="padding-left:18px;margin:4px 0">${Object.values(R.RULES).map(r => `<li>${esc(r)}</li>`).join('')}</ul></details>
      <details class="more"><summary>JSON đầy đủ</summary><pre style="white-space:pre-wrap;font-size:11.5px">${esc(JSON.stringify(configPayload(), null, 2))}</pre></details>
      <button class="btn small" id="cfgExport" style="margin-top:10px">Xuất cấu hình (JSON)</button>`;
  }

  function statusCell(e) {
    const t = { hit: '✓ phát hiện', partial: '◐ một phần', missed: '✗ bỏ sót' }[e.status];
    return `<span class="st ${e.status}">${t}</span>`;
  }

  const ST_ICON = { hit: '✓', partial: '◐', missed: '✗' };
  function pill(m, x) {
    return `<span class="spill ${x.status}" title="${METHOD[m].label}: ${STATUS_VI[x.status]} · coverage ${pct(x.coverage)} · peak ${fmt(x.peak.value, 4)} · delay ${x.delay === null ? 'N/A' : x.delay}">${ST_ICON[x.status]} ${x.status === 'partial' ? Math.round(100 * x.coverage) + '%' : STATUS_VI[x.status]}</span>`;
  }
  /* events tab: one compact row per event, a status pill per method; false alarms in their own view */
  function eventsHTML() {
    const ms = activeMethods();
    if (!ms.length) return '<div class="empty">Chọn ít nhất một phương pháp.</div>';
    const V = ST.evView || (ST.evView = { kind: 'gt', filter: ms.some(m => ST.results[m].ev.nMissed > 0) ? 'missed' : 'all' });
    const ev0 = ST.results[ms[0]].ev.events, nFa = ms.reduce((s, m) => s + ST.results[m].ev.falseAlarms.length, 0);
    const any = (k, st) => ms.some(m => ST.results[m].ev.events[k].status === st);
    const cnt = { all: ev0.length, missed: ev0.filter((_, k) => any(k, 'missed')).length, partial: ev0.filter((_, k) => any(k, 'partial')).length };
    const cols = `grid-template-columns:1fr ${ms.map(() => '92px').join(' ')}`;
    let h = `<div class="seg seg2" id="evKind"><button data-k="gt" class="${V.kind === 'gt' ? 'on' : ''}">Sự kiện thật · ${ev0.length}</button><button data-k="fa" class="${V.kind === 'fa' ? 'on' : ''}">Báo sai · ${nFa}</button></div>`;
    if (V.kind === 'gt') {
      h += `<div class="fchips" id="evFilter">${[['all', 'Tất cả'], ['missed', 'Có bỏ sót'], ['partial', 'Một phần']].map(([k, l]) =>
        `<button data-f="${k}" class="${V.filter === k ? 'on' : ''}" ${cnt[k] === 0 && k !== 'all' ? 'disabled' : ''}>${l} <b>${cnt[k]}</b></button>`).join('')}
        <span style="margin-left:auto">${info(R.RULES.hit + ' ' + R.RULES.coverage + ' Rê chuột lên trạng thái để xem coverage, peak, delay.')}</span></div>
        <div class="evlist"><div class="evhead" style="${cols}"><span>Sự kiện</span>${ms.map(m => `<span style="color:${METHOD[m].color}">${METHOD[m].label}</span>`).join('')}</div>`;
      ev0.forEach((e, k) => {
        if (V.filter !== 'all' && !any(k, V.filter)) return;
        h += `<div class="evrow ${ST.selEvent && ST.selEvent.id === e.id ? 'sel' : ''}" style="${cols}" data-kind="gt" data-id="${e.id}" data-a="${e.a}" data-b="${e.b}">
          <span><b>${e.id}</b><small>${e.a.toLocaleString('vi')}–${(e.b - 1).toLocaleString('vi')} · ${e.length} điểm</small></span>${ms.map(m => pill(m, ST.results[m].ev.events[k])).join('')}</div>`;
      });
      h += '</div>';
    } else {
      h += `<div class="note" style="margin:6px 2px">Đoạn mô hình báo động nhưng không trùng sự kiện thật nào ${info(R.RULES.false_alarm)}</div><div class="evlist">`;
      ms.forEach(m => {
        const fa = ST.results[m].ev.falseAlarms;
        h += `<div class="evgroup"><span class="swatch" style="background:${METHOD[m].color}"></span> ${METHOD[m].label}<span class="note" style="margin-left:auto">${fa.length} đoạn</span></div>`;
        if (!fa.length) h += '<div class="note" style="padding:6px 10px">Không có đoạn báo sai.</div>';
        fa.slice(0, 300).forEach(f => { h += `<div class="evrow fa ${ST.selEvent && ST.selEvent.id === m + '-' + f.id ? 'sel' : ''}" data-kind="fa" data-m="${m}" data-fid="${f.id}" data-a="${f.a}" data-b="${f.b}">
          <span><b>${f.id}</b><small>${f.a.toLocaleString('vi')}–${(f.b - 1).toLocaleString('vi')} · ${f.length} điểm</small></span><span class="note">peak ${fmt(f.peak.value, 3)}</span></div>`; });
        if (fa.length > 300) h += `<div class="note" style="padding:6px 10px">… còn ${fa.length - 300} đoạn (xem đủ bằng Xuất CSV)</div>`;
      });
      h += '</div>';
    }
    return h;
  }
  function bindEvents(box) {
    box.querySelectorAll('#evKind button').forEach(b => b.onclick = () => { ST.evView.kind = b.dataset.k; renderSeriesTab(); });
    box.querySelectorAll('#evFilter button').forEach(b => b.onclick = () => { ST.evView.filter = b.dataset.f; renderSeriesTab(); });
    box.querySelectorAll('.evrow').forEach(r => r.onclick = () => {
      selectEvent({ a: +r.dataset.a, b: +r.dataset.b, id: r.dataset.kind === 'gt' ? r.dataset.id : r.dataset.fid }, r.dataset.kind, r.dataset.m);
      box.querySelectorAll('.evrow').forEach(x => x.classList.toggle('sel', x === r));
    });
  }

  /* evidence trace: per stage peak inside [a, b) against the stage's own label-free reference threshold */
  function traceRows(a, b, keys) {
    const d = ST.data;
    return keys.map(([k, label, unit]) => {
      const s = d.arr[k], ref = d.meta.stage_thresholds[k];
      if (!s || !ref) return null;
      const p = R.peakIn(s, Math.max(a, d.meta.valid_from > a && k.match(/raw|smooth|pre|final|cal_/) ? d.meta.valid_from : a), b);
      return { k, label, unit, peak: p.value, at: p.index, ref: ref.train_mu3sd, over: p.value >= ref.train_mu3sd };
    }).filter(Boolean);
  }
  /* each stage as a bar: peak inside the segment relative to its own reference threshold (the black tick = threshold) */
  // plain names for the stages; the formula stays in the row tooltip
  const STAGE_SHORT = {
    oe: 'Phân loại (A)', residual: 'Phân loại residual (Aʳ)', combined: 'Kết hợp A_c', rec_error: 'Lỗi tái tạo',
    c_raw: 'Nhánh phân loại', r_raw: 'Nhánh tái tạo', coad_pre: 'Gộp, trước làm mượt', c_smooth: 'Phân loại, sau làm mượt',
    r_smooth: 'Tái tạo, sau làm mượt', coad_final: 'Score cuối', cal_c: 'Nhánh phân loại ÷ σ', cal_r: 'Nhánh tái tạo ÷ σ',
    cal_pre: 'Gộp, trước làm mượt', cal_final: 'Score cuối',
  };
  /* evidence bars on a log scale (1x = the stage's own reference threshold, at the black tick; 20x fills the bar) */
  const RMAX = 20, pos = r => Math.min(1, Math.log10(1 + Math.max(0, r)) / Math.log10(1 + RMAX));
  const ratioOf = r => (r.ref > 0 ? r.peak / r.ref : (r.peak > 0 ? Infinity : 0));
  const rtxt = q => (Number.isFinite(q) ? (q >= 10 ? q.toFixed(0) : q.toFixed(1)) + '×' : '∞');
  function traceTable(rows) {
    return rows.map(r => {
      const q = ratioOf(r);
      return `<div class="ev-row" title="${esc(r.label)} — ${esc(r.unit)} · peak ${fmt(r.peak, 4)} tại ${r.at} · ngưỡng tham chiếu (train μ+3σ) ${fmt(r.ref, 4)}">
        <span class="nm">${esc(STAGE_SHORT[r.k] || r.label)}</span><span class="bar"><b class="${r.over ? 'over' : ''}" style="width:${(100 * pos(q)).toFixed(1)}%"></b><u style="left:${(100 * pos(1)).toFixed(1)}%"></u></span>
        <span class="r ${r.over ? 'up' : ''}">${rtxt(q)}</span></div>`;
    }).join('');
  }
  /* where does the signal get lost? one chain per method, each step ✓ (above its reference) or ✗ */
  const CHAIN = {
    coad: [[['combined', 'Phân loại'], ['rec_error', 'Tái tạo']], [['c_raw', 'Nhánh PL'], ['r_raw', 'Nhánh TT']], [['coad_pre', 'Gộp']], [['coad_final', 'Làm mượt → score']]],
    calibrated: [[['combined', 'Phân loại'], ['rec_error', 'Tái tạo']], [['cal_c', 'Nhánh PL ÷σ'], ['cal_r', 'Nhánh TT ÷σ']], [['cal_pre', 'Gộp']], [['cal_final', 'Làm mượt → score']]],
  };
  function chainHTML(m, a, b) {
    const th = ST.results[m].th.value, pk = R.peakIn(score(m), Math.max(a, ST.data.meta.valid_from), b).value, ok = pk >= th;
    const chip = ([k, l]) => { const r = traceRows(a, b, [[k, STAGE_LABEL[k] || k, '']])[0]; if (!r) return '';
      return `<span class="cchip ${r.over ? 'ok' : 'no'}" title="${esc(STAGE_LABEL[k] || k)}: peak ${fmt(r.peak, 4)} so với mức bình thường ${fmt(r.ref, 4)}">${r.over ? '✓' : '✗'} ${l}<b>${rtxt(ratioOf(r))}</b></span>`; };
    return `<div class="chain"><div class="verdict ${ok ? 'ok' : 'no'}"><span class="swatch" style="background:${METHOD[m].color}"></span><b>${METHOD[m].label}</b>
        <span class="vtag">${ok ? '✓ báo động' : '✗ không báo'}</span><span class="note">điểm ${fmt(pk, 3)} ${ok ? '≥' : '<'} ngưỡng ${fmt(th, 3)}</span></div>
      <div class="steps2">${CHAIN[m].map(col => `<div class="col">${col.map(chip).join('')}</div>`).join('<span class="arr">→</span>')}</div></div>`;
  }
  function traceHTML(sel) {
    const { a, b } = sel, ms = activeMethods().filter(m => m !== 'kmeans');
    const lostAtSmoothing = ms.some(m => { const p = traceRows(a, b, [[ST.data.meta.methods[m].pre, '', '']])[0], f = traceRows(a, b, [[ST.data.meta.methods[m].score, '', '']])[0]; return p && f && p.over && !f.over; });
    let h = lostAtSmoothing
      ? ex(IL.smoothing(), 'Ở đoạn này, tín hiệu cao hơn mức bình thường <b>trước</b> làm mượt nhưng không còn <b>sau</b> làm mượt (hình minh họa cách đọc). Mỗi chip là một bước tính: <b>✓</b> cao hơn mức bình thường, <b>✗</b> không.')
      : exText('Mỗi chip là một <b>bước tính</b> của mô hình: <b>✓</b> = tín hiệu ở bước đó cao hơn mức bình thường, <b>✗</b> = không. Đọc từ trái sang phải.');
    h += ms.map(m => chainHTML(m, a, b)).join('');
    h += `<details class="more" style="margin-top:6px"><summary>Xem từng tín hiệu (${STAGES.shared.length + ms.reduce((s, m) => s + STAGES[m].length, 0)} thanh)</summary>
      <div class="legend-mini" style="margin:6px 0"><span style="--c:#E11D48">cao hơn mức bình thường</span><span style="--c:#94A3B8">không</span><span>vạch đen = mức bình thường (1×), thang log</span>
        ${info(`Đoạn ${sel.id}: ${a}–${b - 1} (${b - a} điểm). Mỗi tín hiệu so với ngưỡng tham chiếu của chính nó, lấy từ phần học (μ+3σ, không dùng nhãn). Patch probability được gán cho 8 điểm của patch rồi lấy trung bình qua các window chồng lấp. Rê chuột lên từng dòng để xem giá trị.`)}</div>
      <div class="ev-h">Tín hiệu mô hình</div>${traceTable(traceRows(a, b, STAGES.shared))}
      ${ms.map(m => `<div class="ev-h" style="color:${METHOD[m].color}">${METHOD[m].label}</div>${traceTable(traceRows(a, b, STAGES[m]))}`).join('')}</details>`;
    h += observations(a, b);
    return h;
  }
  function observations(a, b) {
    const get = k => traceRows(a, b, [[k, k, '']])[0], lines = [];
    const ac = get('combined'), re = get('rec_error');
    if (ac && re) lines.push([ac.over || re.over, `Phân loại ${ac.over ? 'có' : 'không'} phản ứng · tái tạo ${re.over ? 'có' : 'không'} phản ứng`]);
    ['coad', 'calibrated'].forEach(m => {
      if (!ST.results[m]) return;
      const pre = get(ST.data.meta.methods[m].pre), fin = get(ST.data.meta.methods[m].score);
      if (pre && fin && pre.over && !fin.over) lines.push([false, `${METHOD[m].label}: vượt trước làm mượt, mất sau làm mượt`]);
    });
    if (!lines.length) return '';
    return `<div class="obs">${lines.map(([ok, t]) => `<span class="${ok ? 'ok' : 'no'}">${ok ? '●' : '○'} ${esc(t)}</span>`).join('')}
      <span class="note">Quan sát, không phải kết luận nguyên nhân ${info('Các nhóm có thể chồng lấp. Để khẳng định tác động của một bước cần đối chứng: thay đúng bước đó, giữ nguyên các bước khác (ví dụ thí nghiệm M18 trong báo cáo).')}</span></div>`;
  }

  /* ---------- Model Inspector ---------- */
  function openInspector(t) {
    ST.inspT = t; ST.inspWin = null; ST.inspPatch = null;
    if (ST.screen === 'series') { renderSeriesTab(); reveal($('#inspPanel')); }
    else { ST.ea.tab = 'patch'; renderCaseTab(); reveal($('#caseTabsPanel')); }
  }
  function windowsAt(t) {
    const w = ST.data.meta.windows, T = ST.data.meta.window, out = [];
    w.starts.forEach((s, i) => { if (t >= s && t < s + T) out.push(i); });
    return out;
  }
  /* how CoAD computes the patch values shown in the heatmap: numbered steps = numbered heatmap rows */
  const INSP_FLOW = `<div class="explain pic wide">${IL.pipeline()}</div>
    <div class="explain pic">${IL.patches()}<div class="txt">Heatmap bên dưới có 7 hàng = các bước ①–④ trong sơ đồ, mỗi cột là một <b>patch</b> 8 điểm. Màu sáng = mô hình nghi patch đó bất thường, nên "che" nó nhiều hơn khi vẽ lại.</div></div>`;
  const INSP_STEPS = `<ol class="isteps"><li><b>Chọn thời điểm</b>: click một điểm trên biểu đồ (hoặc dùng điểm peak của case).</li>
    <li><b>Chọn window</b>: điểm đó nằm trong vài window chồng lấp (W…); mặc định window có tâm gần nhất.</li>
    <li><b>Đọc heatmap</b>: mỗi cột là một patch 8 điểm, mỗi hàng là một bước ①–⑦ ở sơ đồ trên; màu sáng = xác suất bất thường cao.</li>
    <li><b>Click một ô</b>: xem giá trị từng bước của patch đó và tỷ trọng trộn embedding ở hai lượt tái tạo.</li></ol>`;
  /* Inspector as a vertical flow: one small chart per step of CoAD, all on the same time axis of the chosen window.
     Patch values are per window (not the aggregated timestamp score). Click a column to inspect that patch. */
  const FLOW_STEPS = [
    { id: 'in', n: '①', t: 'Dữ liệu vào', s: 'Một cửa sổ (window) của chuỗi; mô hình xét từng patch 8 điểm.' },
    { id: 'A', n: '②', t: 'Phân loại: patch nào trông lạ?', s: 'Cột cao = mô hình nghi patch đó bất thường (A = max của nhánh thời gian và nhánh tần số).', key: 'patch_p_mask', color: '#2563EB' },
    { id: 'rec', n: '③', t: 'Che mềm rồi vẽ lại', s: 'Patch bị nghi được "che" nhiều hơn khi vẽ lại; chỗ bản vẽ lại (nét đứt) lệch dữ liệu thật được tô đỏ.' },
    { id: 'R', n: '④', t: 'Kiểm tra phần dư (Aʳ)', s: 'So dữ liệu với bản vẽ lại: phần dư lớn → Aʳ cao, xác nhận patch lạ.', key: 'patch_r_fused', color: '#0F766E' },
    { id: 'C', n: '⑤', t: 'Kết luận cho từng patch: A_c = (A + Aʳ)/2', s: 'Mức nghi ngờ cuối của patch; đi tiếp vào score theo thời điểm (tab Dấu vết).', key: 'patch_p_comb', color: '#7C3AED' },
  ];
  function renderInspector(box) {
    const d = ST.data;
    if (!d.arr.patch_p_time) { box.innerHTML = '<div class="warnbox">Thiếu diagnostics cấp patch cho series này; score cuối vẫn khả dụng.</div>'; return; }
    if (ST.inspT === null) {
      box.innerHTML = `<div class="empty" style="padding:24px 10px">Click một điểm trên biểu đồ để xem mô hình xử lý đoạn đó qua từng bước</div>
        <details class="more"><summary>Inspector hiển thị gì?</summary>${INSP_FLOW}</details>`;
      return;
    }
    const wins = windowsAt(ST.inspT), T = d.meta.window, N = d.meta.windows.n_patches, P = d.meta.patch;
    if (!wins.length) { box.innerHTML = '<div class="note">Không có window nào chứa điểm này.</div>'; return; }
    if (ST.inspWin === null || !wins.includes(ST.inspWin)) {
      ST.inspWin = wins.reduce((best, i) => Math.abs(d.meta.windows.starts[i] + T / 2 - ST.inspT) < Math.abs(d.meta.windows.starts[best] + T / 2 - ST.inspT) ? i : best, wins[0]);
    }
    const wi = ST.inspWin, s0 = d.meta.windows.starts[wi];
    box.innerHTML = `<div class="note">t = <b>${ST.inspT}</b> · window ${wins.map(i => `<button class="btn small" data-w="${i}" ${i === wi ? 'style="font-weight:700;border-color:#1D4ED8"' : ''}>W${i}</button>`).join(' ')}
      · điểm ${Math.max(0, s0)}–${s0 + T - 1} ${info('Một thời điểm nằm trong nhiều window chồng lấp; mỗi window được xử lý riêng. Giá trị ở đây là của patch trong window đang chọn, không phải score đã tổng hợp theo thời điểm.')}</div>
      <div class="flow2">${FLOW_STEPS.map((st, k) => `${k ? '<div class="fdown">↓</div>' : ''}<div class="fstep2"><div class="fh2"><span class="fn2">${st.n}</span><b>${st.t}</b></div>
        <div class="fs2">${st.s}</div><div id="fl-${st.id}" style="height:${st.id === 'in' || st.id === 'rec' ? 110 : 80}px"></div></div>`).join('')}</div>
      <div class="patchbox" id="patchBox" style="margin-top:10px">Click một cột ở bước ②, ④ hoặc ⑤ để xem chi tiết patch đó.</div>`;
    box.querySelectorAll('button[data-w]').forEach(b => b.onclick = () => { ST.inspWin = +b.dataset.w; ST.inspPatch = null; renderInspector(box); });
    const a = Math.max(0, s0), b = s0 + T, idx = [], inp = [], rec = [];
    for (let i = a; i < b; i++) { idx.push(i); inp.push(d.arr.input[i]); rec.push(d.arr.reconstruction[i]); }
    const centers = Array.from({ length: N }, (_, j) => s0 + j * P + (P - 1) / 2), cfg = { displaylogo: false, responsive: true, displayModeBar: false };
    const lay = extra => Object.assign({}, BASE, { margin: { l: 34, r: 8, t: 4, b: 18 }, showlegend: false, hovermode: 'closest',
      xaxis: { range: [s0 - 0.5, s0 + T - 0.5], showgrid: false }, shapes: flowShapes() }, extra);
    Plotly.newPlot($('#fl-in'), [{ x: idx, y: inp, mode: 'lines', line: { color: '#334155', width: 1.2 }, hovertemplate: 't=%{x}<br>%{y:.3f}<extra></extra>' }], lay({}), cfg);
    // reconstruction step: fill between input and reconstruction where they differ
    Plotly.newPlot($('#fl-rec'), [
      { x: idx, y: rec, mode: 'lines', line: { color: '#0F766E', width: 0 }, hoverinfo: 'skip' },
      { x: idx, y: inp, mode: 'lines', line: { color: '#334155', width: 1.2 }, fill: 'tonexty', fillcolor: 'rgba(225,29,72,.22)', hovertemplate: 'dữ liệu %{y:.3f}<extra></extra>' },
      { x: idx, y: rec, mode: 'lines', line: { color: '#0F766E', width: 1.4, dash: 'dash' }, hovertemplate: 'vẽ lại %{y:.3f}<extra></extra>' }], lay({}), cfg);
    FLOW_STEPS.filter(st => st.key).forEach(st => {
      const vals = Array.from(d.arr[st.key].subarray(wi * N, wi * N + N));
      const div = $('#fl-' + st.id);
      Plotly.newPlot(div, [{ type: 'bar', x: centers, y: vals, width: P * 0.9, marker: { color: vals.map(v => (v >= 0.5 ? st.color : '#CBD5E1')) },
        customdata: vals.map((_, j) => j), hovertemplate: 'patch %{customdata} · %{y:.2f}<extra></extra>' }],
        lay({ yaxis: { range: [0, 1], tickvals: [0, 0.5, 1], fixedrange: true }, bargap: 0 }), cfg);
      div.on('plotly_click', ev => { ST.inspPatch = ev.points[0].customdata; patchDetail(); });
    });
    if (ST.inspPatch !== null) patchDetail();
  }
  // the clicked time point and the selected patch, drawn on every flow chart
  function flowShapes() {
    const sh = [{ type: 'line', x0: ST.inspT, x1: ST.inspT, yref: 'paper', y0: 0, y1: 1, line: { color: '#64748B', dash: 'dot', width: 1 } }];
    if (ST.inspPatch !== null) {
      const P = ST.data.meta.patch, a0 = ST.data.meta.windows.starts[ST.inspWin] + ST.inspPatch * P;
      sh.push({ type: 'rect', x0: a0 - 0.5, x1: a0 + P - 0.5, yref: 'paper', y0: 0, y1: 1, fillcolor: 'rgba(245,158,11,.25)', line: { color: '#F59E0B', width: 1 } });
    }
    return sh;
  }
  function patchDetail() {
    const d = ST.data, N = d.meta.windows.n_patches, P = d.meta.patch, wi = ST.inspWin, j = ST.inspPatch, s0 = d.meta.windows.starts[wi];
    const v = k => d.arr[k][wi * N + j];
    const m1 = v('patch_p_mask'), m2 = v('patch_p_comb');
    const mix = (m, lab) => `<div style="margin-top:8px"><b>${lab}</b><div class="mixbar"><span style="width:${(100 * m).toFixed(1)}%"></span><span style="width:${(100 * (1 - m)).toFixed(1)}%"></span></div>
      <span class="note">"che" (mask) ${(100 * m).toFixed(0)}% · giữ input ${(100 * (1 - m)).toFixed(0)}%</span></div>`;
    $('#patchBox').innerHTML = `<div><b>Patch ${j}</b> · W${wi} · điểm ${s0 + j * P}–${s0 + j * P + P - 1}${s0 + j * P < 0 ? ' (padding)' : ''}</div>
      <div class="pflow"><span>② A <b>${fmt(m1, 2)}</b><small>thời gian ${fmt(v('patch_p_time'), 2)} · tần số ${fmt(v('patch_p_freq'), 2)}</small></span><i>→</i>
        <span>④ Aʳ <b>${fmt(v('patch_r_fused'), 2)}</b><small>thời gian ${fmt(v('patch_r_time'), 2)} · tần số ${fmt(v('patch_r_freq'), 2)}</small></span><i>→</i>
        <span class="last">⑤ A_c <b>${fmt(m2, 2)}</b><small>(A + Aʳ) / 2</small></span></div>
      ${mix(m1, 'Lượt vẽ lại 1 (theo A)')}${mix(m2, 'Lượt vẽ lại 2 (theo A_c)')}
      <div class="note" style="margin-top:6px">Đây là tỷ trọng trộn embedding, không phải tỷ lệ thời điểm bị xóa. Phần input giữ lại không mặc định là bình thường.</div>`;
    ['in', 'A', 'rec', 'R', 'C'].forEach(id => { const g = $('#fl-' + id); if (g && g.data) Plotly.relayout(g, { shapes: flowShapes() }); });
  }

  /* ---------- screen: errors ---------- */
  function errorCases() {
    if (!ST.results.coad || !ST.results.calibrated) return null;
    const cmp = R.compare(ST.results.coad.ev, ST.results.calibrated.ev);
    const gt = cmp.map(c => ({ key: 'gt-' + c.event.id, type: 'gt', id: c.event.id, a: c.event.a, b: c.event.b, cmp: c }));
    const fa = ['coad', 'calibrated'].flatMap(m => ST.results[m].ev.falseAlarms.map(f => ({ key: `fa-${m}-${f.id}`, type: 'fa', id: f.id, method: m, a: f.a, b: f.b })));
    return { gt, fa };
  }
  const KIND_LABEL = { recovered: 'Được cứu', lost: 'Mới bỏ sót', both_missed: 'Cả hai bỏ sót', both_hit: 'Cả hai phát hiện' };
  const FILTERS = [['all', 'Tất cả'], ['missed', 'Bỏ sót (CoAD gốc)'], ['partial', 'Một phần (CoAD gốc)'], ['fa', 'Báo nhầm'], ['recovered', 'Được cứu'], ['lost', 'Mới bỏ sót'], ['both_missed', 'Cả hai bỏ sót']];
  function filtered(c) {
    const f = ST.ea.filter;
    const gt = c.gt.filter(x => f === 'all' || (f === 'missed' && !x.cmp.original.hit) || (f === 'partial' && x.cmp.original.status === 'partial') || x.cmp.kind === f);
    const fa = f === 'all' || f === 'fa' ? c.fa : [];
    return { gt: f === 'fa' ? [] : gt, fa };
  }
  function renderErrors() {
    const root = $('#screen-errors');
    linked = [];
    if (!ST.data) { root.innerHTML = '<div class="panel"><div class="empty">Chọn một chuỗi ở thanh bên trái.</div></div>'; return; }
    const c = errorCases();
    if (!c) { root.innerHTML = '<div class="panel"><div class="empty">Cần bật cả CoAD gốc và Our Improvement ở thanh bên trái để so sánh.</div></div>'; return; }
    const by = k => c.gt.filter(x => x.cmp.kind === k).length;
    const cnt = { missed: c.gt.filter(x => !x.cmp.original.hit).length, partial: c.gt.filter(x => x.cmp.original.status === 'partial').length, fa: c.fa.length,
                  recovered: by('recovered'), lost: by('lost'), both_missed: by('both_missed'), both_hit: by('both_hit') };
    const fc = { coad: ST.results.coad.ev.falseAlarms.length, calibrated: ST.results.calibrated.ev.falseAlarms.length }, dfa = fc.calibrated - fc.coad;
    if (ST.ea.filter === 'fa' || ST.ea.filter === 'missed' || ST.ea.filter === 'partial') ST.ea.filter = 'all';
    const list = filtered(c), flat = list.gt;
    if (!flat.find(x => x.key === ST.ea.key)) { const pick = flat.find(x => x.type === 'gt' && x.cmp.kind !== 'both_hit') || flat[0]; ST.ea.key = pick ? pick.key : null; }
    const tile = (f, n, label, sub, cls) => `<button class="mtile ${cls} ${ST.ea.filter === f ? 'on' : ''}" data-f="${f}" ${n === 0 ? 'disabled' : ''}><b>${n}</b><span>${label}</span>${sub ? `<small>${sub}</small>` : ''}</button>`;
    root.innerHTML = `
      <div class="ecmp one">
        <div class="panel"><div class="ph"><h2>${c.gt.length} sự kiện thật</h2>${info(R.RULES.comparison + ' Ngưỡng: ' + POLICY_LABEL[ST.cfg.policy] + '.')}</div>
          <div class="pb"><div class="mgrid">
            <span></span><span class="mh" style="color:${METHOD.calibrated.color}">Our Improvement ✓ bắt</span><span class="mh" style="color:${METHOD.calibrated.color}">Our Improvement ✗ sót</span>
            <span class="mh rh" style="color:${METHOD.coad.color}">CoAD ✓ bắt</span>${tile('both_hit', cnt.both_hit, 'cả hai bắt', '', 'neutral')}${tile('lost', cnt.lost, 'mới bỏ sót', 'cải tiến làm kém đi', 'bad')}
            <span class="mh rh" style="color:${METHOD.coad.color}">CoAD ✗ sót</span>${tile('recovered', cnt.recovered, 'được cứu', 'cải tiến sửa được', 'good')}${tile('both_missed', cnt.both_missed, 'cả hai sót', '', 'neutral')}
          </div></div></div>
      </div>
      <div id="caseDetail"></div>`;
    root.querySelectorAll('[data-f]').forEach(b => b.onclick = () => { ST.ea.filter = ST.ea.filter === b.dataset.f && b.dataset.f !== 'all' ? 'all' : b.dataset.f; renderErrors(); });
    ST.ea.flat = flat;
    renderCaseDetail();
  }
  function renderCaseDetail() {
    const box = $('#caseDetail'), flat = ST.ea.flat, k = flat.findIndex(x => x.key === ST.ea.key);
    if (k < 0) { box.innerHTML = ''; return; }
    const x = flat[k], d = ST.data, ctx = Math.max(d.meta.period * 2, x.b - x.a);
    ST.view = [Math.max(0, x.a - ctx), Math.min(d.n - 1, x.b + ctx)];
    const head = x.type === 'gt'
      ? `<span class="badge">${x.a.toLocaleString('vi')}–${(x.b - 1).toLocaleString('vi')} · ${x.b - x.a} điểm</span> CoAD ${pill('coad', x.cmp.original)} Our Improvement ${pill('calibrated', x.cmp.improved)}`
      : `<span class="badge">${x.a.toLocaleString('vi')}–${(x.b - 1).toLocaleString('vi')} · ${x.b - x.a} điểm</span> <span class="note">mô hình ${METHOD[x.method].label} báo động, không có bất thường thật</span>`;
    box.innerHTML = `<div class="panel"><div class="ph"><h2>${x.type === 'gt' ? 'Event ' + x.id + ' · ' + KIND_LABEL[x.cmp.kind] : 'Báo nhầm ' + x.id + ' · ' + METHOD[x.method].label}</h2>
        <span class="spacer" style="flex:1"></span><button class="btn small" id="prevCase" ${k === 0 ? 'disabled' : ''}>Trước</button><button class="btn small" id="nextCase" ${k === flat.length - 1 ? 'disabled' : ''}>Sau</button></div>
        <div class="pb">${head} ${info(`Ngữ cảnh hiển thị: ${ST.view[0]}–${ST.view[1]}; vùng ngoài event chỉ để tham khảo, peak lấy trong ranh giới đoạn (khung cam).`)}</div></div>
      <div class="panel"><div class="ph"><h2>Input · GT · prediction</h2></div><div id="eInput" style="height:240px"></div></div>
      <div class="panel"><div class="ph"><h2>Reconstruction & error</h2></div><div id="eRecon" style="height:220px"></div></div>
      <div class="panel"><div class="ph"><h2>Score trước / sau làm mượt</h2><span class="legend-mini"><span style="--c:#94A3B8">chấm = trước làm mượt</span><span style="--c:#334155">liền = score cuối</span><span style="--c:#CBD5E1">đứt = ngưỡng</span></span></div><div id="eScore" style="height:300px"></div></div>
      <div class="panel" id="caseTabsPanel"><div class="tabs" id="caseTabs"><button data-t="patch">Chi tiết patch</button><button data-t="note">Ghi chú</button></div><div class="pb" id="caseTab"></div></div>`;
    $('#prevCase').onclick = () => { ST.ea.key = flat[k - 1].key; renderErrors(); };
    $('#nextCase').onclick = () => { ST.ea.key = flat[k + 1].key; renderErrors(); };
    const band = { type: 'rect', xref: 'x', yref: 'paper', x0: x.a, x1: x.b, y0: 0, y1: 1, line: { color: '#EA580C', width: 1, dash: 'dot' }, fillcolor: 'rgba(0,0,0,0)' };
    drawInput($('#eInput'), ['coad', 'calibrated'], { layout: { shapes: [band] } });
    drawRecon($('#eRecon'), { error: true, layout: { shapes: [band] } });
    drawScores($('#eScore'), ['coad', 'calibrated'], { pre: true, layout: {} });
    $('#caseTabs').querySelectorAll('button').forEach(b => b.onclick = () => { ST.ea.tab = b.dataset.t; renderCaseTab(); reveal($('#caseTabsPanel')); });
    ST.inspT = R.peakIn(d.arr.combined, x.a, x.b).index; ST.inspWin = null; ST.inspPatch = null;
    if (ST.ea.tab !== 'patch' && ST.ea.tab !== 'note') ST.ea.tab = 'patch';
    renderCaseTab();
  }
  function renderCaseTab() {
    document.querySelectorAll('#caseTabs button').forEach(b => b.classList.toggle('active', b.dataset.t === ST.ea.tab));
    const box = $('#caseTab'), x = ST.ea.flat.find(c => c.key === ST.ea.key);
    if (!box || !x) return;
    if (ST.ea.tab === 'obs') box.innerHTML = traceHTML({ a: x.a, b: x.b, id: x.id });
    if (ST.ea.tab === 'patch') { renderInspector(box); return; }
    if (ST.ea.tab === 'note') {
      const k = `coad-note-${ST.group}-${x.key}`;
      box.innerHTML = `<label class="note">Ghi chú của người xem (chỉ lưu trên trình duyệt này; là metadata review, không thay ground truth)</label>
        <textarea id="noteText">${esc(store.get(k, ''))}</textarea>
        <div style="margin-top:6px"><label class="note">Nhãn review: </label><select id="noteTag">${[['unreviewed', 'Chưa xem xét'], ['confirmed anomaly', 'Xác nhận là bất thường'], ['suspected false alarm', 'Nghi là báo nhầm']].map(([t, l]) => `<option value="${t}" ${store.get(k + '-tag', 'unreviewed') === t ? 'selected' : ''}>${l}</option>`).join('')}</select></div>`;
      $('#noteText').oninput = e => store.set(k, e.target.value);
      $('#noteTag').onchange = e => store.set(k + '-tag', e.target.value);
    }
  }

  /* ---------- screen: results ---------- */
  function renderResults() {
    const root = $('#screen-results'), B = ST.bench, f = ST.res;
    const fams = [...new Set(B.rows.map(r => r.family))].sort();
    const k = f.metric, tol = 0.005;
    const all = B.rows.filter(r => (f.family === 'all' || r.family === f.family) && (f.split === 'all' || r.split === f.split));
    // a value that was never computed is N/A, never 0 (spec 2): such series are left out of counts, means and bars
    const has = r => r[k + '_cal'] !== null && r[k + '_cal'] !== undefined && r[k + '_coad'] !== null && r[k + '_coad'] !== undefined;
    const missing = all.filter(r => !has(r));
    let rows = all.filter(has).map(r => Object.assign({ delta: r[k + '_cal'] - r[k + '_coad'] }, r));
    rows.sort(f.sort === 'delta' ? (a, b) => a.delta - b.delta : (a, b) => (a.family + a.group).localeCompare(b.family + b.group));
    const up = rows.filter(r => r.delta > tol).length, down = rows.filter(r => r.delta < -tol).length, tie = rows.length - up - down;
    const mean = a => a.length ? a.reduce((s, v) => s + v, 0) / a.length : NaN;
    const famRows = fams.map(fm => { const rr = rows.filter(r => r.family === fm); return { fm, n: rr.length, coad: mean(rr.map(r => r[k + '_coad'])), cal: mean(rr.map(r => r[k + '_cal'])) }; }).filter(x => x.n);
    const mlabel = { VUS_PR: 'VUS-PR', auc_prc: 'AUC-PR', R_AUC_PR: 'R-AUC-PR', f1_score: 'F1' };
    const mC = mean(rows.map(r => r[k + '_coad'])), mK = mean(rows.map(r => r[k + '_cal'])), mD = mean(rows.map(r => r.delta)), nn = Math.max(1, rows.length);
    root.innerHTML = `
      <div class="panel"><div class="ph"><h2>Our Improvement so với CoAD gốc</h2>${info(B.aggregation)}</div>
        <div class="pb"><div class="filters">
          <div><label>Family</label><select id="fFam"><option value="all">Tất cả</option>${fams.map(x => `<option ${f.family === x ? 'selected' : ''}>${x}</option>`).join('')}</select></div>
          <div><label>Tập</label><select id="fSplit"><option value="all">Tất cả</option><option value="design" ${f.split === 'design' ? 'selected' : ''}>Thiết kế (30)</option><option value="confirmation" ${f.split === 'confirmation' ? 'selected' : ''}>Xác nhận (34)</option></select></div>
          <div><label>Metric</label><select id="fMetric">${Object.entries(mlabel).map(([a, b]) => `<option value="${a}" ${f.metric === a ? 'selected' : ''}>${b}</option>`).join('')}</select></div>
          <div><label>Sắp xếp</label><select id="fSort"><option value="delta">Theo delta</option><option value="family" ${f.sort === 'family' ? 'selected' : ''}>Theo family</option></select></div></div>
          ${missing.length ? `<div class="warnbox" style="margin-top:10px">${missing.length} series N/A (không tính) ${info(`Không có ${mlabel[k]} của Our Improvement: ` + (missing.every(r => r.split === 'design') ? 'tập thiết kế chỉ được chấm nhanh bằng AUC-PR riêng của M11 lúc chọn phương án, không chạy evaluator đầy đủ.' : 'thiếu kết quả.') + ' Chọn Tập = Xác nhận để xem phần kiểm tra độc lập.')}</div>` : ''}
          <div class="note" style="margin-top:8px">Trung bình 3 seed. Trên 60 series của 12 family khác, Our Improvement không hơn CoAD ổn định (báo cáo mục 9). ${info(B.source.coad + '; ' + B.source.calibrated + '. ' + B.source.split + '.')}</div></div></div>
      <div class="kpis">
        <div class="kpi"><div class="v">${rows.length}</div><div class="l">series có ${mlabel[k]} · ${f.split === 'confirmation' ? 'tập xác nhận' : f.split === 'design' ? 'tập thiết kế' : 'mọi tập'}</div></div>
        <div class="kpi accent" style="--c:${METHOD.coad.color}"><div class="v">${pts(mC)}</div><div class="l">CoAD gốc (trung bình)</div></div>
        <div class="kpi accent" style="--c:${METHOD.calibrated.color}"><div class="v">${pts(mK)}</div><div class="l">Our Improvement (trung bình)</div></div>
        <div class="kpi ${mD > tol ? 'good' : mD < -tol ? 'bad' : ''}"><div class="v">${mD >= 0 ? '+' : ''}${pts(mD)}</div><div class="l">Δ trung bình (điểm)</div></div>
        <div class="kpi"><div class="v" style="font-size:18px"><span style="color:var(--ok)">▲${up}</span> · <span style="color:var(--bad)">▼${down}</span> · =${tie}</div>
          <div class="updown"><span style="width:${100 * up / nn}%;background:#16A34A"></span><span style="width:${100 * tie / nn}%;background:#CBD5E1"></span><span style="width:${100 * down / nn}%;background:#E11D48"></span></div>
          <div class="l">tăng · giảm · hòa ${info('Hòa khi |Δ| ≤ ' + (tol * 100).toFixed(1) + ' điểm')}</div></div></div>
      <div class="panel"><div class="ph"><h2>Δ ${mlabel[k]} = Our Improvement − CoAD gốc (điểm)</h2></div>
        <div id="cDelta" style="height:${Math.max(260, rows.length * 16 + 60)}px"></div></div>
      <div class="panel"><div class="ph"><h2>Theo family (trung bình các series)</h2></div><div class="pb"><table><thead><tr><th>Family</th><th class="r">Series</th><th class="r">CoAD gốc</th><th class="r">Our Improvement</th><th class="r">Δ</th></tr></thead><tbody>
        ${famRows.map(x => `<tr><td>${x.fm}</td><td class="r">${x.n}</td><td class="r">${pts(x.coad)}</td><td class="r">${pts(x.cal)}</td><td class="r">${(x.cal - x.coad >= 0 ? '+' : '')}${pts(x.cal - x.coad)}</td></tr>`).join('')}</tbody></table></div></div>
      <details class="panel" style="padding:0"><summary class="ph" style="cursor:pointer"><h2>Bảng chi tiết (${rows.length} series) ▾</h2><span class="sub">click dòng có dữ liệu demo để mở phân tích chuỗi</span></summary><div class="pb tablewrap" style="max-height:480px"><table><thead><tr>
        <th>Series</th><th>Family</th><th>Tập</th><th class="r">Seed</th><th class="r">CoAD (± SD)</th><th class="r">Our Improvement (± SD)</th><th class="r">Δ</th><th class="r">KMeansAD VUS-PR</th><th>Demo</th></tr></thead><tbody>
        ${rows.map(r => `<tr data-g="${r.group}" ${r.demo ? '' : 'style="cursor:default"'}><td>${r.group}</td><td>${r.family}</td><td>${r.split === 'design' ? 'thiết kế' : 'xác nhận'}</td><td class="r">${r.seeds.join(',')}</td>
          <td class="r">${pts(r[k + '_coad'])}${r[k + '_coad_sd'] !== null ? ' ± ' + pts(r[k + '_coad_sd']) : ''}</td><td class="r">${pts(r[k + '_cal'])}${r[k + '_cal_sd'] !== null ? ' ± ' + pts(r[k + '_cal_sd']) : ''}</td>
          <td class="r">${r.delta >= 0 ? '+' : ''}${pts(r.delta)}</td><td class="r">${pts(r.VUS_PR_kmeans)}</td><td>${r.demo ? 'có' : '—'}</td></tr>`).join('')}</tbody></table></div></details>`;
    $('#fFam').onchange = e => { f.family = e.target.value; renderResults(); };
    $('#fSplit').onchange = e => { f.split = e.target.value; renderResults(); };
    $('#fMetric').onchange = e => { f.metric = e.target.value; renderResults(); };
    $('#fSort').onchange = e => { f.sort = e.target.value; renderResults(); };
    root.querySelectorAll('tr[data-g], .gal').forEach(tr => tr.onclick = () => { const r = B.rows.find(x => x.group === +tr.dataset.g); if (r && r.demo) { selectSeries(r.group); go('series'); } });
    const ys = rows.map(r => `${r.group} ${r.family}${r.split === 'design' ? ' (TK)' : ''}`);
    Plotly.newPlot($('#cDelta'), [{ type: 'bar', orientation: 'h', y: ys, x: rows.map(r => 100 * r.delta), text: rows.map(r => (r.delta >= 0 ? '+' : '') + (100 * r.delta).toFixed(2)),
      textposition: 'outside', marker: { color: rows.map(r => (r.delta > tol ? '#16A34A' : r.delta < -tol ? '#E11D48' : '#94A3B8')), opacity: rows.map(r => (r.split === 'design' ? 0.45 : 1)) }, hovertemplate: '%{y}: Δ=%{x:.2f}<extra></extra>' }],
      Object.assign({}, BASE, { margin: { l: 120, r: 50, t: 8, b: 30 }, xaxis: { zeroline: true, zerolinecolor: '#0F172A', title: { text: 'Δ (điểm)', font: { size: 11 } } },
        yaxis: { automargin: true, categoryorder: 'array', categoryarray: ys }, showlegend: false }), { displaylogo: false, responsive: true });
  }

  /* ---------- screen: data ----------
     dataset cards on top (pick one to explore), family tiles, a length × anomaly-ratio scatter, a compact series list */
  const TAG_LABEL = { success: 'Our Improvement tốt hơn', failure: 'Kém hơn ở checkpoint demo', representative: 'Đại diện' };
  const DEMO_TAG = { 237: 'success', 276: 'success', 234: 'representative', 269: 'failure', 295: 'failure' };
  const FAM_COLORS = ['#B45309', '#DB2777', '#65A30D', '#0891B2', '#64748B', '#CA8A04', '#EA580C', '#BE185D', '#4D7C0F', '#0E7490', '#A16207', '#475569', '#9F1239', '#15803D', '#7C2D12', '#334155', '#C2410C'];
  async function renderData() {
    const root = $('#screen-data');
    if (!ST.datasets) {
      root.innerHTML = '<div class="panel"><div class="loading">Đang tải metadata dữ liệu…</div></div>';
      try { ST.datasets = await loadJSON('data/datasets.json'); }
      catch (e) { root.innerHTML = `<div class="panel"><div class="empty">Không đọc được data/datasets.json (${esc(e.message)}). Chạy tools/build_datasets.js.</div></div>`; return; }
    }
    const D = ST.datasets, f = ST.dataFilter || (ST.dataFilter = { set: 'paper', family: 'all' });
    if (f.set !== 'paper' && f.set !== 'external') f.set = 'paper';
    const sumEv = l => l.reduce((s, r) => s + r.events, 0).toLocaleString('vi'), nFam = l => new Set(l.map(r => r.family)).size;
    const card = (key, color, icon, title, big, sub, role) => `<button class="dcard ${f.set === key ? 'sel' : ''}" data-set="${key}" style="--c:${color}">
        <span class="dic">${icon}</span><span class="dt">${title}</span><span class="dv">${big}</span><span class="ds">${sub}</span><span class="dr">${role}</span></button>`;
    const k = D.kdd21;
    let body = '';
    if (f.set === 'upload') body = uploadAnalysisHTML();
    else if (f.set === 'kdd21') body = kddHTML(k);
    else body = setHTML(f.set === 'paper' ? D.paper : D.external, f);
    root.innerHTML = `
      <div class="dcards">
        ${card('paper', '#2563EB', '', 'TSB-AD · 64 series của paper', D.paper.length, `${nFam(D.paper)} family · ${sumEv(D.paper)} sự kiện`, 'Tái lập Table 1 · thử cải tiến')}
        ${card('external', '#B45309', '', 'Series nhóm chạy thêm (ngoài paper)', D.external.length, `${nFam(D.external)} family khác · ${sumEv(D.external)} sự kiện`, 'Kiểm tra cải tiến có tổng quát không')}
      </div>
      ${body}
      <details class="more" style="margin:4px 2px 16px"><summary>Cách tiền xử lý và chia tập</summary><ul class="note" style="margin:6px 0 0 18px;padding:0">
        <li>Mọi series đơn biến, có nhãn từng điểm. Phần học là đoạn đầu chuỗi (độ dài ghi trong tên file); loader của tác giả chấm điểm toàn chuỗi.</li>
        <li>Min–max theo phần học. Không tự nội suy hay lấy mẫu lại.</li>
        <li>Thiết kế / xác nhận: 30 / 34 series, <code>default_rng(2026)</code>. Series chạy thêm: 5 series mỗi family, <code>default_rng(2027)</code>, bỏ file trùng.</li></ul></details>`;
    root.querySelectorAll('.dcard').forEach(b => b.onclick = () => {
      if (b.dataset.set === 'upload' && !ST.upload) { $('#csvFile').click(); return; }
      f.set = b.dataset.set; f.family = 'all'; renderData();
    });
    if (f.set === 'upload') bindUploadAnalysis();
    bindSet(f.set === 'paper' ? D.paper : D.external, f);
  }
  function famStats(list) {
    const by = {};
    list.forEach(r => { const b = by[r.family] || (by[r.family] = { n: 0, ev: 0, pts: 0, anom: 0 }); b.n++; b.ev += r.events; b.pts += r.length; b.anom += r.anomaly_points; });
    return Object.entries(by).sort();
  }
  function setHTML(list, f) {
    const rows = list, paper = f.set === 'paper';
    return `<div class="panel"><div class="ph"><h2>Series</h2><span class="note">${rows.length}</span></div>
          <div class="evlist" style="max-height:520px;border:0;border-radius:0">${rows.map(r => `<div class="evrow srow ${r.demo && paper ? 'demo' : ''}" data-g="${r.group}">
            <span><b>${r.family} · ${r.group}</b><small>${r.length.toLocaleString('vi')} điểm · ${r.events} sự kiện · ${(100 * r.anomaly_ratio).toFixed(1)}% bất thường${paper ? ' · tập ' + (r.split === 'design' ? 'thiết kế' : 'xác nhận') : ''}</small></span>
            ${r.demo && paper ? '<span class="spill hit" title="Có checkpoint: bấm để mở Phân tích chuỗi">mở</span>' : ''}</div>`).join('')}</div></div>`;
  }
  function bindSet(list, f) {
    $('#screen-data').querySelectorAll('.srow.demo').forEach(r => r.onclick = () => { selectSeries(+r.dataset.g); go('series'); });
  }
  function kddHTML(k) {
    const t = (label, s, unit = '') => `<div class="kstat"><span class="l">${label}</span><div class="kline"><span>${s.min.toLocaleString('vi')}</span><i></i><b>${s.median.toLocaleString('vi')}</b><i></i><span>${s.max.toLocaleString('vi')}</span></div><span class="u">min · trung vị · max${unit}</span></div>`;
    return `<div class="panel"><div class="ph"><h2>KDD21 · UCR Anomaly Archive 2021</h2><span class="badge">${k.series} series</span></div><div class="pb">
      ${exText('Mỗi series có <b>đúng một</b> đoạn bất thường, nằm trong phần sau của chuỗi. Bài toán: tìm đúng đoạn đó.')}
      <div class="kstats">${t('Độ dài phần học', k.train_length, ' (điểm)')}${t('Độ dài phần kiểm tra', k.test_length, ' (điểm)')}${t('Độ dài đoạn bất thường', k.anomaly_length, ' (điểm)')}${t('Chu kỳ', k.period, ' (điểm)')}</div>
      <div class="note" style="margin-top:10px">Dựng lại từ archive gốc (giảm mẫu 10 lần ở series 239–241, nới nhãn 1 điểm thành 3 điểm ở 079, 108, 187). Demo không có checkpoint KDD21; kết quả xem trong báo cáo mục 6.5.</div></div></div>`;
  }

  /* ---------- upload (validation + preview only) ---------- */
  /* one entry point for a CSV: file picker, drag and drop, or a "Dùng thử" sample */
  function loadCsv(name, text) {
    const lines = text.split(/\r?\n/).filter(l => l.trim().length);
    if (!lines.length) return;
    const delim = [',', ';', '\t'].sort((a, b) => lines[0].split(b).length - lines[0].split(a).length)[0];
    const header = lines[0].split(delim).map(s => s.trim());
    const rows = lines.slice(1).map(l => l.split(delim));
    const isTime = h => /^(time|timestamp|date|datetime|index|ts)$/i.test(h), isLab = h => /^(label|is_anomaly|anomaly)$/i.test(h);
    const val = header.find(h => !isTime(h) && !isLab(h)) || header[0];
    ST.upRun = null;
    ST.upload = { name, header, rows, delim, ts: header.find(isTime) || '__index__', val, lab: header.find(isLab) || '' };
    if (ST.screen === 'data') (ST.dataFilter || (ST.dataFilter = { set: 'paper', family: 'all' })).set = 'upload';
    renderUploadSide(); go(ST.screen === 'data' ? 'data' : 'series');
  }
  const readFile = file => { if (!file) return; const r = new FileReader(); r.onload = () => loadCsv(file.name, String(r.result)); r.readAsText(file); };
  if ($('#csvFile')) $('#csvFile').onchange = e => readFile(e.target.files[0]);
  const dz = $('#dropZone') || document.createElement('div');
  ['dragenter', 'dragover'].forEach(t => dz.addEventListener(t, e => { e.preventDefault(); dz.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(t => dz.addEventListener(t, e => { e.preventDefault(); dz.classList.remove('over'); }));
  dz.addEventListener('drop', e => readFile(e.dataTransfer.files[0]));
  document.querySelectorAll('[data-sample]').forEach(b => b.onclick = async () => {
    const f = b.dataset.sample;
    try { const r = await fetch('samples/' + f); if (!r.ok) throw new Error(r.status); loadCsv(f, await r.text()); }
    catch (err) { $('#uploadBox').innerHTML = `<div class="warnbox" style="margin-top:6px">Không tải được ${esc(f)} (${esc(err.message)}).</div>`; }
  });
  function renderUploadSide() {
    const u = ST.upload, box = $('#uploadBox');
    if (!u) { box.innerHTML = ''; return; }
    const opts = (sel, extra) => extra + u.header.map(h => `<option ${sel === h ? 'selected' : ''}>${esc(h)}</option>`).join('');
    box.innerHTML = `<div class="note"><b>${esc(u.name)}</b> · ${u.rows.length} dòng</div>
      <label>Cột thời gian</label><select id="uTs">${opts(u.ts, `<option value="__index__" ${u.ts === '__index__' ? 'selected' : ''}>(dùng index)</option>`)}</select>
      <label>Cột giá trị</label><select id="uVal">${opts(u.val, '')}</select>
      <label>Cột label (tùy chọn)</label><select id="uLab">${opts(u.lab, '<option value="">(không có)</option>')}</select>
      <button class="btn small" id="uClose" style="margin-top:6px">Đóng</button>`;
    ['uTs', 'uVal', 'uLab'].forEach(id => $('#' + id).onchange = e => { u[{ uTs: 'ts', uVal: 'val', uLab: 'lab' }[id]] = e.target.value; ST.upRun = null; renderAll(); });
    $('#uClose').onclick = () => { ST.upload = null; $('#csvFile').value = ''; renderUploadSide(); renderAll(); };
  }
  function validateUpload() {
    const u = ST.upload, col = n => u.header.indexOf(n);
    const vi = col(u.val), li = u.lab ? col(u.lab) : -1, ti = u.ts === '__index__' ? -1 : col(u.ts);
    const vals = [], issues = [];
    let missing = 0, nonnum = 0, dup = 0, nonInc = 0, prev = null;
    const seen = new Set();
    u.rows.forEach(r => {
      const raw = (r[vi] || '').trim();
      if (raw === '') { missing++; vals.push(null); } else if (Number.isNaN(+raw)) { nonnum++; vals.push(null); } else vals.push(+raw);
      if (ti >= 0) {
        const t = (r[ti] || '').trim(), tv = Number.isNaN(+t) ? Date.parse(t) : +t;
        if (seen.has(t)) dup++; seen.add(t);
        if (prev !== null && !(tv > prev)) nonInc++; prev = tv;
      }
    });
    if (missing) issues.push(`${missing} giá trị trống (không tự nội suy).`);
    if (nonnum) issues.push(`${nonnum} giá trị không phải số.`);
    if (dup) issues.push(`${dup} timestamp trùng.`);
    if (nonInc) issues.push(`${nonInc} vị trí timestamp không tăng.`);
    if (u.rows.length < 256) issues.push('Chuỗi ngắn hơn 256 điểm — quá ngắn cho một window CoAD (4 chu kỳ, bội số của 8).');
    let labInfo = '';
    if (li >= 0) { const l = u.rows.map(r => +(r[li] || '').trim()); const bad = l.filter(v => v !== 0 && v !== 1).length; labInfo = bad ? `Label có ${bad} giá trị khác 0/1.` : `Label: ${l.filter(v => v === 1).length} điểm bất thường.`; }
    return { vals, issues, labInfo, labels: li >= 0 ? u.rows.map(r => +(r[li] || '').trim() === 1 ? 1 : 0) : null };
  }
  /* ---------- descriptive analysis of the uploaded CSV ("Dữ liệu" screen). Label-free statistics; no model is run. ---------- */
  // TSB-AD find_length: argmax over local maxima of the ACF (lags 3..400) on the first 20 000 points; 125 if none in [3, 300]
  function findLength(v) {
    const x = v.slice(0, 20000), n = x.length;
    if (n < 50) return { period: null, acf: [], fallback: true };
    const m = x.reduce((s, a) => s + a, 0) / n;
    let c0 = 0; for (const a of x) c0 += (a - m) ** 2;
    const L = Math.min(400, n - 1), acf = [1];
    for (let k = 1; k <= L; k++) { let s = 0; for (let i = 0; i + k < n; i++) s += (x[i] - m) * (x[i + k] - m); acf.push(c0 ? s / c0 : 0); }
    const base = 3, a = acf.slice(base);
    let best = -1;
    for (let i = 1; i < a.length - 1; i++) if (a[i] > a[i - 1] && a[i] > a[i + 1] && (best < 0 || a[i] > a[best])) best = i;
    const fallback = best < 3 || best > 300;
    return { period: fallback ? 125 : best + base, acf, fallback, peakLag: best < 0 ? null : best + base };
  }
  function analyzeUpload(v) {
    const u = ST.upload, num = v.vals.filter(x => x !== null), n = num.length;
    const mean = n ? num.reduce((s, a) => s + a, 0) / n : NaN, sd = n ? Math.sqrt(num.reduce((s, a) => s + (a - mean) ** 2, 0) / n) : NaN;
    let mn = Infinity, mx = -Infinity; for (const a of num) { if (a < mn) mn = a; if (a > mx) mx = a; }
    let samp = null;
    const ti = u.ts === '__index__' ? -1 : u.header.indexOf(u.ts);
    if (ti >= 0 && u.rows.length > 1) {
      const isDate = Number.isNaN(+(u.rows[0][ti] || '').trim());
      const t = u.rows.map(r => { const s = (r[ti] || '').trim(); return isDate ? Date.parse(s.replace(' ', 'T')) : +s; });
      const d = []; for (let i = 1; i < t.length; i++) if (Number.isFinite(t[i]) && Number.isFinite(t[i - 1])) d.push(t[i] - t[i - 1]);
      const sorted = d.slice().sort((a, b) => a - b), med = sorted.length ? sorted[Math.floor(sorted.length / 2)] : NaN;
      samp = { med, isDate, gaps: d.filter(x => x > 1.5 * med).length, irregular: d.filter(x => x !== med).length };
    }
    let lab = null;
    if (v.labels) {
      const seg = R.segments(v.labels, 0), len = seg.map(([a, b]) => b - a).sort((a, b) => a - b);
      lab = { events: seg.length, points: v.labels.reduce((s, a) => s + a, 0), first: seg.length ? seg[0][0] : null, len };
    }
    return Object.assign({ n: v.vals.length, valid: n, mean, sd, mn, mx, samp, lab }, findLength(num));
  }
  function fmtStep(s) {
    if (!s || !Number.isFinite(s.med)) return 'N/A';
    if (!s.isDate) return 'Δ = ' + s.med;
    const sec = s.med / 1000;
    return sec < 60 ? `${sec} giây` : sec < 3600 ? `${+(sec / 60).toFixed(1)} phút` : sec < 86400 ? `${+(sec / 3600).toFixed(1)} giờ` : `${+(sec / 86400).toFixed(1)} ngày`;
  }
  /* ---------- detection on an uploaded CSV, in the browser ----------
     No CoAD checkpoint fits a new series (each is trained on one series), so the page runs a KMeans subsequence
     detector in the spirit of TSB-AD's KMeansAD: windows of one period, k = 20 centres fitted on up to 4 000 windows,
     window score = distance to the nearest centre, point score = mean over the windows covering it. Threshold:
     mean + 3 SD of its own score (no training part, like KMeansAD in the benchmark). Labels are only used to evaluate. */
  function kmeansLite(x, w, k = 20) {
    const n = x.length, nw = n - w + 1;
    if (nw < k * 2) return null;
    let rnd = 12345; const rand = () => (rnd = (rnd * 1103515245 + 12345) % 2147483648) / 2147483648;
    const m = Math.min(nw, 4000), fit = Array.from({ length: m }, (_, j) => Math.floor(j * nw / m));
    const d2 = (i, c) => { let s = 0; for (let t = 0; t < w; t++) { const e = x[i + t] - c[t]; s += e * e; } return s; };
    let C = Array.from({ length: k }, () => { const i = fit[Math.floor(rand() * m)]; return Float64Array.from(x.subarray(i, i + w)); });
    const asg = new Int32Array(m);
    for (let it = 0; it < 15; it++) {
      for (let j = 0; j < m; j++) { let b = 0, bd = Infinity; for (let c = 0; c < k; c++) { const v = d2(fit[j], C[c]); if (v < bd) { bd = v; b = c; } } asg[j] = b; }
      const S = C.map(() => new Float64Array(w)), cnt = new Int32Array(k);
      for (let j = 0; j < m; j++) { const s = S[asg[j]], i = fit[j]; for (let t = 0; t < w; t++) s[t] += x[i + t]; cnt[asg[j]]++; }
      C = C.map((c, i) => (cnt[i] ? S[i].map(v => v / cnt[i]) : c));
    }
    const pre = new Float64Array(nw + 1);
    for (let i = 0; i < nw; i++) { let bd = Infinity; for (let c = 0; c < k; c++) { const v = d2(i, C[c]); if (v < bd) bd = v; } pre[i + 1] = pre[i] + Math.sqrt(bd); }
    const sc = new Float32Array(n);
    for (let t = 0; t < n; t++) { const lo = Math.max(0, t - w + 1), hi = Math.min(nw - 1, t); sc[t] = (pre[hi + 1] - pre[lo]) / (hi - lo + 1); }
    return sc;
  }
  function runUpload() {
    const { v, A } = ST.upA, box = $('#uaRun');
    if (v.vals.some(a => a === null)) { box.innerHTML = '<div class="warnbox">File có ô trống hoặc giá trị không phải số. Sửa file trước khi chạy (demo không tự nội suy).</div>'; return; }
    box.innerHTML = '<div class="loading">Đang chạy KMeansAD trên ' + v.vals.length.toLocaleString('vi') + ' điểm…</div>';
    setTimeout(() => {
      const t0 = performance.now(), raw = Float64Array.from(v.vals);
      const mu = raw.reduce((s, a) => s + a, 0) / raw.length, sd = Math.sqrt(raw.reduce((s, a) => s + (a - mu) ** 2, 0) / raw.length) || 1;
      const x = raw.map(a => (a - mu) / sd), w = Math.max(8, A.period || 125);
      const score = kmeansLite(x, w);
      if (!score) { box.innerHTML = '<div class="warnbox">Chuỗi quá ngắn so với cửa sổ ' + w + ' điểm.</div>'; return; }
      let m = 0; for (const s of score) m += s; m /= score.length; let s2 = 0; for (const s of score) s2 += (s - m) ** 2;
      const th = m + 3 * Math.sqrt(s2 / score.length), pred = R.predict(score, th, 0);
      const ev = v.labels ? R.evaluate(Float32Array.from(v.labels), pred, 0, score) : null, flagged = pred.reduce((s, a) => s + a, 0);
      ST.upRun = { score, th, pred, ev, w, ms: performance.now() - t0, flagged, intervals: R.segments(pred, 0).length };
      renderUploadRun();
    }, 30);
  }
  function renderUploadRun() {
    const r = ST.upRun, box = $('#uaRun'), { v } = ST.upA;
    if (!box || !r) return;
    const stat = (big, lab, cls = '') => `<div class="kpi ${cls}"><div class="v">${big}</div><div class="l">${lab}</div></div>`;
    box.innerHTML = `<div class="kpis" style="margin:8px 0">
        ${r.ev ? stat(`${r.ev.nHit}<small> / ${r.ev.events.length}</small>`, 'đoạn bất thường bắt được', r.ev.nHit === r.ev.events.length ? 'good' : '') + stat(r.ev.falseAlarms.length, 'đoạn báo sai', r.ev.falseAlarms.length ? 'bad' : 'good')
               + stat(`${pct(r.ev.point.precision)} · ${pct(r.ev.point.recall)}`, 'precision · recall (theo điểm)') : stat('—', 'không có nhãn: chỉ xem chỗ bị gắn cờ')}
        ${stat(`${r.intervals}`, `đoạn cảnh báo · ${(100 * r.flagged / v.vals.length).toFixed(1)}% điểm`)}
        ${stat(`${(r.ms / 1000).toFixed(1)} s`, `cửa sổ ${r.w} điểm · ngưỡng μ+3σ`)}</div>
      <div id="uaRunChart" style="height:300px"></div>
      <div class="note">KMeansAD chạy trong trình duyệt (baseline của bảng so sánh), <b>không phải CoAD</b>. Kết quả không dùng nhãn; nhãn chỉ để chấm. Metric chuẩn (VUS-PR) cần evaluator của TSB-AD nên không tính ở đây.</div>`;
    const y = v.vals, dec = decimate(Float64Array.from(y), 3000), ds = decimate(r.score, 3000), step = dec.step, n = y.length;
    const band = (arr, yref, color, name) => { const bx = [], by = []; for (let s = 0; s < n; s += step) { let mm = 0; for (let i = s; i < Math.min(n, s + step); i++) if (arr[i]) { mm = 1; break; } bx.push(s, Math.min(n, s + step) - 1); by.push(mm, mm); }
      return { x: bx, y: by, type: 'scattergl', mode: 'lines', line: { width: 0, shape: 'hv' }, fill: 'tozeroy', fillcolor: color, yaxis: yref, name, hoverinfo: 'skip' }; };
    const tr = [];
    if (v.labels) tr.push(band(v.labels, 'y3', 'rgba(251,146,60,.35)', 'Bất thường thật'));
    tr.push({ x: dec.x, y: dec.y, type: 'scattergl', mode: 'lines', line: { color: '#334155', width: 1 }, name: ST.upload.val, hovertemplate: 'i=%{x}<br>%{y}<extra></extra>' });
    tr.push(band(r.pred, 'y4', 'rgba(180,83,9,.85)', 'KMeansAD báo động'));
    tr.push({ x: ds.x, y: ds.y, type: 'scattergl', mode: 'lines', line: { color: METHOD.kmeans.color, width: 1.2 }, yaxis: 'y2', name: 'Điểm bất thường', hovertemplate: 'i=%{x}<br>score=%{y:.3f}<extra></extra>' });
    Plotly.newPlot($('#uaRunChart'), tr, Object.assign({}, BASE, {
      yaxis: { domain: [0.5, 1] }, yaxis3: { domain: [0.5, 1], overlaying: 'y', range: [0, 1], visible: false, fixedrange: true },
      yaxis2: { domain: [0.08, 0.42], title: { text: 'score', font: { size: 11 } } }, yaxis4: { domain: [0, 0.05], range: [0, 1], visible: false, fixedrange: true },
      shapes: [{ type: 'line', xref: 'paper', x0: 0, x1: 1, yref: 'y2', y0: r.th, y1: r.th, line: { color: '#64748B', dash: 'dash', width: 1 } }] }), { displaylogo: false, responsive: true });
  }

  /* ---------- CoAD on an uploaded CSV: trained by the local server (serve_coad.py) ---------- */
  async function serverInfo() {
    if (ST.server !== undefined) return ST.server;
    try { const r = await fetch('api/health'); ST.server = r.ok ? await r.json() : null; } catch (e) { ST.server = null; }
    return ST.server;
  }
  async function loadUserSeries() {
    if (!(await serverInfo())) return;
    try { ST.userSeries = await (await fetch('api/uploads')).json(); renderSidebar(); } catch (e) { /* server without uploads */ }
  }
  async function coadPanel() {
    const box = $('#uaCoad');
    if (!box) return;
    const srv = await serverInfo(), { v, A } = ST.upA;
    if (!srv) {
      box.innerHTML = `<div class="rh"><b>CoAD trên file này</b><span class="note">cần server train chạy trên máy</span></div>
        <div class="note" style="margin-top:4px">CoAD phải được train riêng cho chuỗi mới. Tắt server hiện tại rồi chạy (trong thư mục <code>demo</code>):</div>
        <pre class="cmd">.venv\\Scripts\\python serve_coad.py</pre><div class="note">Sau đó mở lại http://localhost:8765 và tải lại file.</div>`;
      return;
    }
    const n = v.vals.length, first = A.lab && A.lab.first !== null ? A.lab.first : null;
    const def = Math.max(64, Math.min(Math.floor(n * 0.6), first !== null && first > 64 ? first - (A.period || 0) : Math.floor(n * 0.2)));
    box.innerHTML = `<div class="rh"><b>Train CoAD trên file này</b><span class="note">trên ${esc(srv.device)} của máy bạn ${info('Server train CoAD (code mode, seed 46) trên phần học ở đầu file, coi là bình thường; ngưỡng tính từ phần học, không dùng nhãn. Nhãn (nếu có) chỉ dùng để chấm. Chế độ nhanh giữ nguyên mô hình, loss và cách tính score, chỉ học trên tối đa 128 cửa sổ mỗi epoch; 300 epoch đầy đủ là cấu hình của paper (cần GPU).')}</span></div>
      <div class="trainform">
        <label>Phần học (điểm đầu, bình thường)<input type="number" id="tnTrain" min="64" max="${n - 1}" value="${def}"></label>
        <label>Chế độ<select id="tnMode"><option value="fast">Nhanh trên CPU (vài phút)</option><option value="full">Đầy đủ như code tác giả (rất lâu trên CPU)</option></select></label>
        <label>Số epoch<select id="tnEpochs"><option value="30">30</option><option value="50" selected>50</option><option value="100">100</option><option value="300">300 · như paper</option></select></label>
        <button class="btn primary" id="tnGo">Train CoAD</button></div>
      <div class="note">${first !== null ? `Bất thường đầu tiên ở điểm ${first.toLocaleString('vi')}; phần học mặc định dừng trước đó.` : 'File không có nhãn: phần học mặc định là 20% đầu chuỗi.'}</div>
      <div id="tnStatus"></div>`;
    $('#tnGo').onclick = () => startTrain(+$('#tnTrain').value, +$('#tnEpochs').value, $('#tnMode').value === 'fast');
    if (ST.trainJob) pollTrain();
  }
  async function startTrain(nTrain, epochs, fast) {
    const { v } = ST.upA, st = $('#tnStatus');
    if (v.vals.some(a => a === null)) { st.innerHTML = '<div class="warnbox">File có ô trống hoặc giá trị không phải số; sửa file trước khi train.</div>'; return; }
    st.innerHTML = '<div class="loading">Đang gửi dữ liệu…</div>';
    try {
      const r = await fetch('api/train', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: ST.upload.name, values: v.vals, labels: v.labels, n_train: nTrain, epochs, fast }) });
      const j = await r.json();
      if (!r.ok) throw new Error(j.error || r.status);
      ST.trainJob = j.job; pollTrain();
    } catch (e) { st.innerHTML = `<div class="warnbox">Không bắt đầu được: ${esc(e.message)}</div>`; }
  }
  async function pollTrain() {
    const job = ST.trainJob;
    if (!job) return;
    let j;
    try { j = await (await fetch('api/job/' + job)).json(); } catch (e) { j = { status: 'error', error: e.message }; }
    const st = $('#tnStatus');
    if (st) {
      const p = j.epochs ? j.epoch / j.epochs : 0;
      st.innerHTML = j.status === 'error' ? `<div class="warnbox">Lỗi khi train: ${esc(j.error || '')}</div>`
        : j.status === 'done' ? `<div class="okbox">✓ Xong sau ${j.seconds} s — đang mở chuỗi…</div>`
        : `<div class="tprog"><div class="bar"><i style="width:${(100 * (j.status === 'scoring' ? 1 : p)).toFixed(1)}%"></i></div>
           <span>${j.status === 'scoring' ? 'Đang tính score…' : j.status === 'queued' ? 'Đang chuẩn bị…' : `Epoch ${j.epoch}/${j.epochs}`}${j.loss !== undefined ? ` · loss ${j.loss.toFixed(4)}` : ''}${j.eta !== undefined && j.status === 'training' ? ` · còn ~${Math.ceil(j.eta)} s` : ''}</span></div>
           ${j.window ? `<div class="note">chu kỳ ${j.period} · cửa sổ ${j.window} điểm</div>` : ''}`;
    }
    if (j.status === 'done') {
      ST.trainJob = null;
      ST.userSeries = (ST.userSeries || []).filter(x => x.group !== j.group).concat([{ group: j.group, name: ST.upload.name, epochs: j.epochs }]);
      ST.upTrained = j.group; await storyStep3(j.group); return;
    }
    if (j.status === 'error' || j.status === 'unknown') { ST.trainJob = null; return; }
    setTimeout(pollTrain, 1000);
  }

  /* ---------- the journey of an uploaded CSV: ① data & insights → ② train CoAD → ③ CoAD step by step → ④ results ---------- */
  function stepDuration(samp, k) {
    if (!samp || !samp.isDate || !Number.isFinite(samp.med)) return '';
    const h = samp.med * k / 3.6e6;
    return h < 1 ? ` ≈ ${Math.round(h * 60)} phút` : h < 48 ? ` ≈ ${+h.toFixed(1)} giờ` : ` ≈ ${+(h / 24).toFixed(1)} ngày`;
  }
  /* plain-language findings computed from the file itself (labels only describe, never tune anything) */
  function uploadInsights(v, A) {
    const out = [], T = A.period ? Math.floor(4 * A.period / 8) * 8 : null;
    if (A.samp) out.push(A.samp.gaps ? ['', `Có <b>${A.samp.gaps}</b> chỗ dữ liệu bị hụt (khoảng cách lớn hơn 1,5 lần nhịp đo ${fmtStep(A.samp)}).`, 'warn']
      : ['', `Đo đều, mỗi <b>${fmtStep(A.samp)}</b> một điểm, không bị hụt.`, 'ok']);
    if (A.period && !A.fallback) out.push(['', `Có nhịp lặp lại rõ: chu kỳ khoảng <b>${A.period} điểm</b>${stepDuration(A.samp, A.period)}. CoAD xem mỗi lần ${T} điểm (4 chu kỳ).`, 'ok']);
    else if (A.peakLag) out.push(['', `Nhịp lặp mạnh nhất dài <b>${A.peakLag} điểm</b>${stepDuration(A.samp, A.peakLag)}, vượt giới hạn 300 điểm của TSB-AD, nên CoAD dùng chu kỳ mặc định 125 (cửa sổ ${T} điểm).`, 'warn']);
    if (A.lab && A.lab.events) {
      const L = A.lab.len;
      out.push(['', `<b>${A.lab.events}</b> đoạn bất thường được gán nhãn, chiếm <b>${(100 * A.lab.points / A.n).toFixed(1)}%</b> số điểm; mỗi đoạn dài ${L[0] === L[L.length - 1] ? L[0] : L[0] + '–' + L[L.length - 1]} điểm.`, '']);
      const first = A.lab.first;
      if (T) out.push(first >= 2 * T ? ['', `Đoạn đầu chuỗi bình thường dài <b>${first.toLocaleString('vi')} điểm</b>, đủ để CoAD học (cần ít nhất ${2 * T}).`, 'ok']
        : ['', `Đoạn bình thường ở đầu chỉ có <b>${first} điểm</b>, ngắn hơn 2 cửa sổ (${2 * T}); CoAD sẽ học trên rất ít dữ liệu.`, 'warn']);
      // are the anomalies simple spikes? share of labelled points beyond 3 SD of the whole series
      let far = 0, tot = 0;
      v.vals.forEach((x, i) => { if (v.labels[i] && x !== null) { tot++; if (Math.abs(x - A.mean) > 3 * A.sd) far++; } });
      if (tot) out.push(far / tot < 0.2
        ? ['', `Chỉ <b>${(100 * far / tot).toFixed(0)}%</b> điểm bất thường có giá trị vượt 3σ: bất thường ở đây là <b>hình dạng lạ</b>, không phải đột biến, nên ngưỡng giá trị đơn giản sẽ bỏ sót; cần mô hình xét ngữ cảnh như CoAD.`, '']
        : ['', `<b>${(100 * far / tot).toFixed(0)}%</b> điểm bất thường vượt 3σ: nhiều đoạn là đột biến giá trị, dễ thấy bằng mắt.`, '']);
    } else out.push(['', 'File không có nhãn: kết quả sẽ chỉ cho biết chỗ mô hình báo động, không chấm đúng/sai được.', 'warn']);
    if (v.issues.length) out.push(['', 'Lỗi định dạng: ' + v.issues.map(esc).join(' · '), 'warn']);
    return `<div class="insights">${out.map(([i, t, c]) => `<div class="ins ${c}"><span class="ii">${i}</span><span>${t}</span></div>`).join('')}</div>`;
  }
  /* 4-step bar; stage: 1 data, 2 training, 3 trained */
  function journeyHTML(stage, here) {
    const steps = [['①', 'Dữ liệu & insight'], ['②', 'Train CoAD'], ['③', 'CoAD từng bước'], ['④', 'Kết quả']];
    return `<div class="journey">${steps.map(([n, t], k) => {
      const st = k + 1 === here ? 'on' : k + 1 < stage || (k + 1 === 2 && stage >= 3) ? 'done' : k + 1 <= stage ? 'ready' : 'off';
      return `<button class="jstep ${st}" data-j="${k + 1}" ${st === 'off' ? 'disabled' : ''}><span class="jn">${st === 'done' ? '✓' : n}</span>${t}</button>`;
    }).join('<span class="jarr">→</span>')}</div>`;
  }
  function bindJourney(root) {
    root.querySelectorAll('.jstep').forEach(b => b.onclick = () => {
      const k = +b.dataset.j, g = ST.upTrained;
      if (k === 1) { ST.dataFilter = Object.assign(ST.dataFilter || {}, { set: 'upload' }); go('data'); }
      if (k === 2) { ST.dataFilter = Object.assign(ST.dataFilter || {}, { set: 'upload' }); go('data'); setTimeout(() => reveal($('#uaCoad')), 50); }
      if (k === 3 && g) storyStep3(g);
      if (k === 4 && g) storyStep4(g);
    });
  }
  /* ③ open the series on a labelled event (prefer one CoAD missed) with the step-by-step Inspector */
  async function storyStep3(g) {
    if (ST.group !== g) await selectSeries(g);
    ST.story = 3; go('series');
    const evs = ST.results.coad.ev.events, e = evs.find(x => x.status === 'missed') || evs[0];
    if (e) { selectEvent(e); ST.inspT = R.peakIn(ST.data.arr.combined, e.a, e.b).index; }
    else ST.inspT = R.peakIn(score('coad'), ST.data.meta.valid_from, ST.data.n).index;   // no labels: the highest score
    ST.inspWin = null; ST.inspPatch = null; ST.tab = 'insp'; renderSeriesTab(); reveal($('#dock'));
  }
  /* ④ results: CoAD, Our Improvement and the in-browser KMeansAD on the same file and the same event rules */
  async function storyStep4(g) {
    if (ST.group !== g) await selectSeries(g);
    ST.story = 4; go('series');
  }
  function storyResultsHTML() {
    const d = ST.data, ms = ['coad', 'calibrated'].filter(m => ST.results[m]);
    if (!ST.kmUp || ST.kmUp.group !== ST.group) {
      const x = Float64Array.from(d.arr.input), sc = kmeansLite(x, Math.max(8, d.meta.period));
      if (sc) { let m = 0; sc.forEach(s => m += s); m /= sc.length; let s2 = 0; sc.forEach(s => s2 += (s - m) ** 2);
        const th = m + 3 * Math.sqrt(s2 / sc.length), pred = R.predict(sc, th, d.meta.valid_from);
        ST.kmUp = { group: ST.group, ev: R.evaluate(d.arr.label, pred, d.meta.valid_from, sc) }; }
    }
    const rows = ms.map(m => [METHOD[m].label, METHOD[m].color, ST.results[m].ev]).concat(ST.kmUp && ST.kmUp.group === ST.group ? [['KMeansAD (baseline)', METHOD.kmeans.color, ST.kmUp.ev]] : []);
    const hasLab = d.meta.has_label;
    return `<div class="panel storyres"><div class="ph"><h2>④ Kết quả trên file của bạn</h2><span class="note">cùng quy tắc sự kiện · ngưỡng từ phần học, không dùng nhãn</span></div><div class="pb">
      ${hasLab ? `<table class="restab"><thead><tr><th>Phương pháp</th><th class="r">Bắt được</th><th class="r">Báo sai</th><th class="r">Precision</th><th class="r">Recall</th><th class="r">VUS-PR</th></tr></thead><tbody>
        ${rows.map(([l, c, ev], i) => `<tr><td><span class="swatch" style="background:${c}"></span> ${l}</td><td class="r"><b>${ev.nHit}</b> / ${ev.events.length}</td><td class="r">${ev.falseAlarms.length}</td>
          <td class="r">${pct(ev.point.precision)}</td><td class="r">${pct(ev.point.recall)}</td><td class="r">${i < ms.length ? pts(d.meta.methods[ms[i]].metrics.VUS_PR) : 'N/A'}</td></tr>`).join('')}</tbody></table>
        <div class="note" style="margin-top:6px">KMeansAD dùng ngưỡng μ+3σ của chính điểm số (không có phần học). VUS-PR cần evaluator của TSB-AD: N/A nếu server không chạy với <code>--repo</code>.</div>`
      : '<div class="note">File không có nhãn: chỉ xem được chỗ mỗi phương pháp báo động (biểu đồ bên dưới), không chấm đúng/sai.</div>'}
      <div class="note" style="margin-top:6px">CoAD train ${d.meta.run.epochs} epoch trên ${d.meta.n_train.toLocaleString('vi')} điểm đầu, ${esc(d.meta.run.device)}, ${d.meta.run.train_seconds} s${d.meta.run.training_mode ? ' · chế độ ' + esc(d.meta.run.training_mode) : ''}.</div></div></div>`;
  }

  function uploadAnalysisHTML() {
    const u = ST.upload, v = validateUpload(), A = analyzeUpload(v);
    ST.upA = { v, A };
    const T = A.period ? Math.floor(4 * A.period / 8) * 8 : null, q = a => a[Math.floor(a.length / 2)];
    return `<div class="panel"><div class="ph"><h2>${esc(u.name)}</h2><span class="badge">Chỉ mô tả dữ liệu</span>
        ${info('Thống kê mô tả trên chính file, không chạy mô hình và không dùng nhãn để tính gì ngoài phần "Nhãn". Chu kỳ ước lượng theo find_length của TSB-AD (ACF, lag 3–400, 20 000 điểm đầu), đúng cách loader của tác giả chọn chu kỳ. Không tự nội suy hay lấy mẫu lại.')}
        <span class="spacer" style="flex:1"></span><span class="note">cột giá trị: <b>${esc(u.val)}</b></span></div>
      <div class="pb">${journeyHTML(ST.upTrained ? 3 : 1, 1)}${uploadInsights(v, A)}<div class="kpis" style="margin-bottom:10px">
        <div class="kpi"><div class="v">${A.n.toLocaleString('vi')}</div><div class="l">điểm · ${A.valid.toLocaleString('vi')} giá trị số hợp lệ</div></div>
        <div class="kpi ${v.issues.length ? 'bad' : 'good'}"><div class="v">${v.issues.length ? v.issues.length + ' lỗi' : 'OK'}</div><div class="l">kiểm tra định dạng</div></div>
        <div class="kpi"><div class="v" style="font-size:20px">${A.samp ? fmtStep(A.samp) : 'theo index'}</div><div class="l">nhịp lấy mẫu${A.samp ? ` · ${A.samp.gaps} chỗ hụt` : ''}</div></div>
        <div class="kpi"><div class="v">${A.period ?? 'N/A'}${A.fallback && A.period ? '<small> (mặc định)</small>' : ''}</div><div class="l">chu kỳ ước lượng (điểm)${T ? ` · cửa sổ CoAD T = ${T}` : ''}${A.fallback && A.peakLag ? ` ${info(`Đỉnh ACF mạnh nhất ở lag ${A.peakLag}, ngoài khoảng 3–300 mà find_length của TSB-AD chấp nhận, nên loader dùng giá trị mặc định 125 (giống cách loader của tác giả xử lý).`)}` : ''}</div></div>
        <div class="kpi"><div class="v">${A.lab ? A.lab.events : '—'}</div><div class="l">${A.lab ? `đoạn bất thường · ${(100 * A.lab.points / A.n).toFixed(2)}% số điểm` : 'không có cột nhãn'}</div></div></div>
        ${v.issues.length ? `<div class="warnbox" style="margin-bottom:10px">${v.issues.map(esc).join(' · ')}</div>` : ''}
        <div class="runbox"><div class="rh"><button class="btn primary" id="uaRunBtn">Chạy phát hiện bất thường</button><span class="note">KMeansAD trong trình duyệt · vài giây ${info("CoAD không chạy được trên file mới: mỗi checkpoint CoAD được train riêng cho một chuỗi (PyTorch, vài phút trên GPU). Muốn có CoAD cho file này cần train riêng ngoài web. KMeansAD là baseline không cần train trước: cắt chuỗi thành đoạn dài một chu kỳ, gom 20 cụm, điểm bất thường = khoảng cách tới tâm cụm gần nhất.")}</span></div><div id="uaRun"></div></div>
        <div class="runbox coadbox" id="uaCoad"></div>
        <div id="uaSeries" style="height:220px"></div>
        <div class="split3" style="margin-top:8px"><div><div class="ev-h">Phân bố giá trị</div><div id="uaHist" style="height:200px"></div></div>
          <div><div class="ev-h">Tự tương quan (ACF) ${info('Đỉnh ACF cao nhất trong lag 3–300 là chu kỳ chính; vạch đỏ = chu kỳ được chọn.')}</div><div id="uaAcf" style="height:200px"></div></div>
          <div><div class="ev-h">Chi tiết</div><dl class="kv">
            <dt>Min / max</dt><dd class="num">${fmt(A.mn, 3)} / ${fmt(A.mx, 3)}</dd><dt>Trung bình ± SD</dt><dd class="num">${fmt(A.mean, 3)} ± ${fmt(A.sd, 3)}</dd>
            ${A.samp ? `<dt>Bước không đều</dt><dd class="num">${A.samp.irregular.toLocaleString('vi')}</dd>` : ''}
            ${A.lab ? `<dt>Độ dài đoạn (min / trung vị / max)</dt><dd class="num">${A.lab.len.length ? `${A.lab.len[0]} / ${q(A.lab.len)} / ${A.lab.len[A.lab.len.length - 1]}` : '—'}</dd>
              <dt>Điểm bất thường đầu tiên</dt><dd class="num">${A.lab.first === null ? '—' : A.lab.first.toLocaleString('vi') + ` (${(100 * A.lab.first / A.n).toFixed(1)}% chuỗi)`}</dd>` : ''}
            ${T ? `<dt>Số cửa sổ CoAD (stride = chu kỳ)</dt><dd class="num">${Math.max(0, Math.floor((A.n - T) / A.period) + 1).toLocaleString('vi')}</dd>` : ''}
            ${T && A.n < T ? '<dt>Tương thích</dt><dd style="color:var(--bad)">chuỗi ngắn hơn 1 cửa sổ</dd>' : ''}</dl>
            ${A.lab && A.lab.first !== null ? `<div class="note" style="margin-top:6px">CoAD cần đoạn đầu bình thường để train: trước điểm bất thường đầu tiên có ${A.lab.first.toLocaleString('vi')} điểm.</div>` : ''}</div></div></div></div>`;
  }
  function bindUploadAnalysis() {
    if (!ST.upA || !$('#uaSeries')) return;
    $('#uaRunBtn').onclick = runUpload;
    bindJourney($('#screen-data'));
    coadPanel();
    if (ST.upRun) renderUploadRun();
    const { v, A } = ST.upA, y = v.vals.map(a => (a === null ? NaN : a)), dec = decimate(y, 3000), tr = [];
    if (v.labels) {
      const step = dec.step, lx = [], ly = [];
      for (let s = 0; s < v.labels.length; s += step) { let m = 0; for (let i = s; i < Math.min(v.labels.length, s + step); i++) if (v.labels[i]) { m = 1; break; } lx.push(s, Math.min(v.labels.length, s + step) - 1); ly.push(m, m); }
      tr.push({ x: lx, y: ly, type: 'scattergl', mode: 'lines', line: { width: 0, shape: 'hv' }, fill: 'tozeroy', fillcolor: 'rgba(251,146,60,.35)', yaxis: 'y2', name: 'Nhãn bất thường', hoverinfo: 'skip' });
    }
    tr.push({ x: dec.x, y: dec.y, type: 'scattergl', mode: 'lines', line: { color: '#334155', width: 1 }, name: ST.upload.val, hovertemplate: 'i=%{x}<br>%{y}<extra></extra>' });
    const cfg = { displaylogo: false, responsive: true };
    Plotly.newPlot($('#uaSeries'), tr, Object.assign({}, BASE, { yaxis2: { overlaying: 'y', range: [0, 1], visible: false, fixedrange: true } }), cfg);
    Plotly.newPlot($('#uaHist'), [{ x: y.filter(Number.isFinite), type: 'histogram', nbinsx: 50, marker: { color: '#64748B' }, hovertemplate: '%{x}: %{y}<extra></extra>' }],
      Object.assign({}, BASE, { margin: { l: 48, r: 8, t: 4, b: 28 }, showlegend: false, bargap: 0.02 }), cfg);
    Plotly.newPlot($('#uaAcf'), [{ x: A.acf.map((_, i) => i), y: A.acf, type: 'scatter', mode: 'lines', line: { color: '#2563EB', width: 1.2 }, hovertemplate: 'lag %{x}: %{y:.3f}<extra></extra>' }],
      Object.assign({}, BASE, { margin: { l: 40, r: 8, t: 4, b: 28 }, showlegend: false, xaxis: { title: { text: 'lag', font: { size: 11 } } },
        shapes: A.period ? [{ type: 'line', x0: A.period, x1: A.period, yref: 'paper', y0: 0, y1: 1, line: { color: '#E11D48', dash: 'dot' } }] : [] }), cfg);
  }
  /* series screen: one line about the uploaded file, pointing to its journey (analysis, CoAD training, results) */
  function uploadTrainedGroup() {
    return (ST.userSeries || []).filter(u => u.name === ST.upload.name).map(u => u.group).pop() || null;
  }
  function uploadPanelHTML() {
    const v = validateUpload(), g = uploadTrainedGroup();
    return `<div class="upbanner"><span><b>${esc(ST.upload.name)}</b> · ${v.vals.length.toLocaleString('vi')} điểm${v.labels ? ' · có nhãn' : ''}${v.issues.length ? ` · <span style="color:var(--bad)">${v.issues.length} lỗi định dạng</span>` : ''}</span>
      <span class="note">${g ? 'đã train CoAD trên file này' : 'chưa train CoAD'}</span>
      <button class="btn small primary" id="upGo">${g ? 'Xem CoAD trên file này →' : 'Phân tích & train CoAD →'}</button></div>`;
  }
  function bindUploadPanel() {
    $('#upGo').onclick = () => {
      const g = uploadTrainedGroup();
      if (g) { ST.upTrained = g; storyStep4(g); return; }
      ST.dataFilter = Object.assign(ST.dataFilter || { family: 'all' }, { set: 'upload' }); go('data');
    };
  }

  /* ---------- export ---------- */
  function download(name, text, type) {
    const a = el('a', { href: URL.createObjectURL(new Blob([text], { type })), download: name });
    document.body.append(a); a.click(); a.remove();
  }
  function configPayload() {
    const d = ST.data, m = d.meta;
    return {
      series: { group: m.group, file: m.file, family: m.family, length: m.length, n_train: m.n_train },
      run: m.run, result_mode: 'precomputed', calibration: m.calibration,
      preprocessing: "min-max on the training part (TSB-AD loader of the authors' code)",
      window: { T: m.window, patch: m.patch, inference_stride: m.inference_stride, train_stride: m.train_stride, valid_from: m.valid_from },
      smoothing: { type: 'moving average (np.convolve same)', width: m.period },
      methods: Object.fromEntries(activeMethods().map(k => [k, { threshold: ST.results[k].th, metrics_threshold_free: m.methods[k].metrics,
        evaluator: 'evaluation/evaluator.py of the CoAD repository (VUS from TSB-AD)' }])),
      threshold_policy: ST.cfg.policy, interactive: ST.cfg.policy === 'interactive', event_rules: Object.fromEntries(Object.entries(R.RULES).map(([k, v]) => [k, window.I18N ? I18N.tr(v) : v])),
      runtime_scope: m.run.inference_scope,
    };
  }
  function exportCSV() {
    const d = ST.data, ms = activeMethods(), n = d.n, vf = d.meta.valid_from;
    const head = ['index', 'input', 'label', 'valid', 'reconstruction', ...ms.flatMap(m => [`${m}_score_pre_smoothing`, `${m}_score`, `${m}_threshold`, `${m}_prediction`])];
    const lines = [head.join(',')];
    for (let i = 0; i < n; i++) {
      const row = [i, d.arr.input[i], d.arr.label[i], i >= vf ? 1 : 0, d.arr.reconstruction[i]];
      ms.forEach(m => { const pre = d.meta.methods[m].pre; row.push(pre ? d.arr[pre][i] : '', score(m)[i], ST.results[m].th.value, i >= vf ? ST.results[m].pred[i] : ''); });
      lines.push(row.join(','));
    }
    download(`coad_series_${ST.group}_${ST.cfg.policy}.csv`, lines.join('\n'), 'text/csv');
    exportJSON();
  }
  function exportJSON() { download(`coad_series_${ST.group}_config.json`, JSON.stringify(configPayload(), null, 2), 'application/json'); }

  /* ---------- about ---------- */
  function renderAbout() {
    $('#aboutBox').innerHTML = `<h2>Phạm vi và tái lập</h2>
      <p><b>Đây là demo nghiên cứu với kết quả tính sẵn.</b> Không có training/inference trên web, không có model tổng quát cho CSV mới (upload chỉ kiểm tra và xem trước).</p>
      <h3>Mô hình và dữ liệu</h3><ul>
        <li>CoAD được cài đặt lại (code mode, khớp code tác giả, T4 &lt; 1e-5). Mỗi series demo dùng đúng checkpoint GPU của benchmark (runs/reimpl, seed 46, 300 epoch); score tính lại khớp score đã lưu của benchmark (sai khác tối đa ghi trong Cấu hình lần chạy). Mỗi checkpoint được train riêng cho một series.</li>
        <li><b>Our Improvement</b> (M11, B1): mỗi nhánh đã xử lý được chia cho độ lệch chuẩn của nó trên phần train trước khi gộp. Không dùng nhãn. Là phương án thử nghiệm: tăng trên tập xác nhận nhưng không tổng quát trên 12 family ngoài.</li>
        <li><b>KMeansAD</b>: baseline_compare.py của tác giả; score tính trên chính chuỗi test, nên ngưỡng mặc định là μ+3σ của chính score đó (không có phần train).</li>
        <li>Dữ liệu: TSB-AD-U; phần train là đoạn đầu của chuỗi (nền xám), toàn chuỗi được chấm điểm như loader của tác giả.</li></ul>
      <h3>Ngưỡng</h3><ul><li>Train μ + 3σ và Max trên phần train: tính trên score của phần train (không dùng nhãn), riêng cho từng phương pháp theo thang của nó.</li>
        <li>Ngưỡng tương tác chỉ để khám phá, gắn nhãn Interactive và không ghi đè benchmark.</li><li>Không có ngưỡng nào lấy từ nhãn test.</li></ul>
      <h3>Quy tắc event và metric</h3><ul>${Object.values(R.RULES).map(r => `<li>${esc(r)}</li>`).join('')}</ul>
      <p>VUS-PR, AUC-PR, R-AUC-PR, F1 (best) là metric không phụ thuộc ngưỡng, tính trước bằng evaluator của repo tác giả trên dữ liệu đầy đủ. Precision/recall điểm và event recall phụ thuộc ngưỡng, tính bằng <code>rules.js</code> — module duy nhất, mọi màn hình dùng chung.
      Khác đặc tả: quy tắc event chạy ở trình duyệt (một module dùng chung) thay vì backend, để ngưỡng tương tác dùng đúng cùng quy tắc.</p>
      <h3>Tái lập</h3><p><code>python precompute.py --repo &lt;repo CoAD&gt; --groups ... --epochs 300 --results &lt;results&gt; --m11 &lt;m11&gt;</code>; xem demo/README.md. Mỗi kết quả kèm checkpoint ID, seed, số epoch, thiết bị và thời gian.</p>
      <p><button class="btn" id="closeAbout">Đóng</button></p>`;
    $('#closeAbout').onclick = () => $('#about').classList.remove('open');
  }

  /* ---------- navigation ---------- */
  function go(screen) {
    ST.screen = screen;
    document.querySelectorAll('header nav button').forEach(b => b.classList.toggle('active', b.dataset.screen === screen));
    ['data', 'series', 'errors', 'results'].forEach(s => $('#screen-' + s).classList.toggle('hidden', s !== screen));
    renderAll();
  }
  function renderAll() {
    renderToolbar();
    if (ST.screen === 'data') renderData();
    if (ST.screen === 'series') renderSeries();
    if (ST.screen === 'errors') renderErrors();
    if (ST.screen === 'results') renderResults();
  }
  async function selectSeries(group) {
    ST.group = group; ST.view = null; ST.selEvent = null; ST.inspT = null; ST.ea.key = null; ST.loadError = null;
    $('#screen-series').innerHTML = '<div class="panel"><div class="loading">Đang tải dữ liệu series ' + group + ' (metadata, tín hiệu, diagnostics patch)…</div></div>';
    try {
      ST.data = await loadSeries(group);
      if (ST.data.meta.family === 'CSV') ST.upTrained = group; else ST.story = null;
      ST.cfg.inter = {}; ST.draft = JSON.parse(JSON.stringify(ST.cfg));
      compute();
    } catch (err) {
      ST.data = null; ST.cache[group] = null;
      ST.loadError = `Chưa có dữ liệu tính sẵn cho series ${group} (${err.message}). Chạy precompute.py cho series này rồi bấm Tải kết quả.`;
    }
    renderSidebar(); renderAll();
  }

  document.querySelectorAll('header nav button').forEach(b => b.onclick = () => go(b.dataset.screen));
  $('#toggleSidebar').onclick = () => { const w = $('#workspace'); w.classList.toggle(innerWidth <= 860 ? 'open' : 'collapsed'); };
  $('#workspace').addEventListener('click', e => { const w = $('#workspace'); if (w.classList.contains('open') && (e.target === w || e.target.closest('.srow1'))) w.classList.remove('open'); });
  $('#openAbout').onclick = () => { renderAbout(); $('#about').classList.add('open'); };
  $('#about').onclick = e => { if (e.target.id === 'about') $('#about').classList.remove('open'); };

  /* novice mode: plain-language pictures on/off (remembered per browser) */
  function setExplain(on) {
    document.body.classList.toggle('noexplain', !on);
    $('#toggleExplain').textContent = 'Giải thích: ' + (on ? 'Bật' : 'Tắt');
    store.set('coad-explain', on);
  }
  $('#toggleExplain').onclick = () => setExplain(document.body.classList.contains('noexplain'));
  setExplain(store.get('coad-explain', true));

  /* guided tour: dims the page and points at one area at a time */
  const TOUR = [
    { screen: 'series', sel: '#sidebar', t: '1. Chọn dữ liệu', p: 'Chọn một chuỗi mẫu và phương pháp ở đây. Mỗi chuỗi là dữ liệu đo liên tục theo thời gian, ví dụ điện tâm đồ hay số liệu một dịch vụ web.' },
    { screen: 'series', sel: '.context', t: '2. Bạn đang xem gì', p: 'Dòng này cho biết chuỗi là dữ liệu gì, có bao nhiêu đoạn bất thường thật, và câu hỏi cần trả lời.' },
    { screen: 'series', sel: '.mcards', t: '3. Kết quả từng phương pháp', p: 'Số lớn là số đoạn bất thường bắt được trên tổng số. Mỗi ô vuông là một đoạn: xanh = bắt được, đỏ = bỏ sót. Bấm một ô để phóng to tới đoạn đó.' },
    { screen: 'series', sel: '#cInput', t: '4. Dữ liệu và chỗ mô hình báo động', p: 'Đường xám là dữ liệu. Nền cam là bất thường thật. Thanh màu bên dưới là lúc mô hình báo động; trùng nền cam là bắt đúng.' },
    { screen: 'series', sel: '#cScore', t: '5. Điểm bất thường', p: 'Mô hình chấm điểm cho từng thời điểm. Điểm vượt vạch đứt (ngưỡng) thì báo động.' },
    { screen: 'series', sel: '#inspPanel', t: '6. CoAD chạy thế nào', p: 'Inspector cho thấy một cửa sổ dữ liệu đi qua từng bước của CoAD: phân loại patch, che rồi vẽ lại, kiểm tra phần dư, kết luận. Click một điểm trên biểu đồ để đổi cửa sổ.' },
    { screen: 'series', sel: 'header nav', t: '7. Các màn hình khác', p: 'Phân tích lỗi: so CoAD với bản cải tiến trên từng đoạn. Kết quả thực nghiệm: tổng hợp trên nhiều chuỗi. Dữ liệu: mô tả các bộ dữ liệu và phân tích file CSV của bạn.' },
  ];
  let tourI = -1, hole = null, pop = null;
  function tourShow() {
    const s = TOUR[tourI];
    if (ST.screen !== s.screen) go(s.screen);
    const node = $(s.sel);
    if (!node) { tourNext(1); return; }
    const tall = node.getBoundingClientRect().height > innerHeight * 0.6;
    if (s.sel === '#sidebar' || s.sel === 'header nav') scrollTo(0, 0); else node.scrollIntoView({ block: tall ? 'start' : 'center' });
    requestAnimationFrame(() => {
      const r = node.getBoundingClientRect(), pad = 6;
      Object.assign(hole.style, { left: r.left - pad + 'px', top: r.top - pad + 'px', width: r.width + 2 * pad + 'px', height: Math.min(r.height, innerHeight - 40) + 2 * pad + 'px' });
      pop.innerHTML = `<h4>${s.t}</h4><p>${s.p}</p><div class="row"><span class="n">${tourI + 1}/${TOUR.length}</span>
        <button class="btn small" id="tPrev" ${tourI === 0 ? 'disabled' : ''}>Trước</button>
        <button class="btn small primary" id="tNext">${tourI === TOUR.length - 1 ? 'Xong' : 'Tiếp'}</button><button class="btn small" id="tEnd">Bỏ qua</button></div>`;
      const below = r.bottom + 12 + 170 < innerHeight;
      Object.assign(pop.style, { left: Math.max(16, Math.min(innerWidth - 356, r.left)) + 'px', top: (below ? r.bottom + 12 : Math.max(16, r.top - 182)) + 'px' });
      $('#tPrev').onclick = () => tourNext(-1); $('#tNext').onclick = () => tourNext(1); $('#tEnd').onclick = tourEnd;
    });
  }
  function tourNext(d) { tourI += d; if (tourI >= TOUR.length || tourI < 0) { tourEnd(); return; } tourShow(); }
  function tourStart() {
    if (!ST.data) return;
    hole = el('div', { className: 'tour-hole' }); pop = el('div', { className: 'tour-pop' });
    document.body.append(hole, pop); tourI = 0; tourShow();
  }
  function tourEnd() { if (hole) hole.remove(); if (pop) pop.remove(); hole = pop = null; tourI = -1; store.set('coad-tour-seen', true); }
  $('#openTour').onclick = () => { if (tourI < 0) tourStart(); };
  document.addEventListener('keydown', e => { if (tourI >= 0 && e.key === 'Escape') tourEnd(); });


  (async function init() {
    try { ST.bench = await loadJSON('data/benchmark.json'); }
    catch (e) { $('#screen-series').innerHTML = `<div class="panel"><div class="empty">Không đọc được data/benchmark.json (${esc(e.message)}). Mở demo qua một web server (README.md).</div></div>`; return; }
    const demo = ST.bench.rows.filter(r => r.demo).map(r => r.group);
    renderSidebar();
    let first = null;
    const order = [237, ...demo.filter(g => g !== 237)];
    for (const g of order) { if (!demo.includes(g)) continue; try { await loadJSON(`data/series_${g}.json`); first = g; break; } catch (e) { ST.cache[g] = null; } }
    // chart legends are laid out with the font loaded at draw time: wait (briefly) for the web font first
    if (document.fonts && document.fonts.load) await Promise.race([document.fonts.load('12px "Be Vietnam Pro"'), new Promise(r => setTimeout(r, 1500))]).catch(() => {});
    if (first !== null) {
      await selectSeries(first);
      if (!store.get('coad-tour-seen', false)) setTimeout(tourStart, 800);   // first visit: guided tour
      return;
    }
    renderSidebar(); renderAll();
  })();
})();
