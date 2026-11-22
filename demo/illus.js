/* Small explanatory illustrations (inline SVG, light SMIL motion) used instead of long text.
   Every picture is schematic: it shows how to read a chart, never real results. Exported as window.Illus. */
(function () {
  'use strict';
  const C = { sig: '#334155', gt: 'rgba(251,146,60,.38)', coad: '#2563EB', cal: '#7C3AED', ok: '#16A34A', bad: '#E11D48',
              rec: '#0F766E', th: '#64748B', muted: '#64748B' };
  const T = (x, y, s, o = {}) => `<text x="${x}" y="${y}" font-size="${o.size || 9.5}" fill="${o.fill || C.muted}" text-anchor="${o.anchor || 'middle'}" font-weight="${o.w || 400}">${s}</text>`;
  const svg = (w, h, body, label) => `<svg class="illus" viewBox="0 0 ${w} ${h}" width="${w}" height="${h}" role="img" aria-label="${label}">${body}</svg>`;
  // a path through f(x) for x in [x0, x1]
  const path = (x0, x1, f, step = 2) => { let d = ''; for (let x = x0; x <= x1; x += step) d += (d ? 'L' : 'M') + x.toFixed(1) + ',' + f(x).toFixed(1); return d; };
  const scan = (x0, x1, y0, y1, dur = 4) => `<line x1="${x0}" x2="${x0}" y1="${y0}" y2="${y1}" stroke="${C.th}" stroke-width="1" stroke-dasharray="2 2" class="motion">
      <animate attributeName="x1" values="${x0};${x1}" dur="${dur}s" repeatCount="indefinite"/><animate attributeName="x2" values="${x0};${x1}" dur="${dur}s" repeatCount="indefinite"/></line>`;

  /* how to read input + ground truth + prediction */
  function readChart() {
    const f = x => 38 + 12 * Math.sin(x / 7) + (x > 72 && x < 108 ? -22 * Math.exp(-((x - 90) ** 2) / 60) : 0) + (x > 172 && x < 198 ? 8 * Math.sin(x / 1.5) : 0);
    return svg(270, 112, `
      <rect x="70" y="4" width="40" height="64" fill="${C.gt}"/><rect x="170" y="4" width="30" height="64" fill="${C.gt}"/>
      <path d="${path(4, 266, f)}" fill="none" stroke="${C.sig}" stroke-width="1.4"/>
      <rect x="74" y="76" width="34" height="7" rx="2" fill="${C.coad}"/><rect x="232" y="76" width="18" height="7" rx="2" fill="${C.coad}"/>
      ${T(90, 98, '✓ bắt đúng', { fill: C.ok, w: 600 })}${T(185, 98, '✗ bỏ sót', { fill: C.bad, w: 600 })}${T(241, 98, 'báo sai', { fill: '#B45309', w: 600 })}
      ${T(30, 81, 'mô hình báo →', { size: 8.5, anchor: 'middle' })}${T(135, 110, 'nền cam = bất thường thật (người gán nhãn)', { size: 8.5 })}
      ${scan(4, 266, 2, 86)}`, 'Cách đọc: nền cam là bất thường thật, thanh màu là chỗ mô hình báo động');
  }

  /* score crosses the dashed threshold -> alarm */
  function threshold() {
    const f = x => 70 - 6 * Math.abs(Math.sin(x / 5)) - (x > 110 && x < 160 ? 52 * Math.exp(-((x - 135) ** 2) / 120) : 0);
    return svg(270, 96, `
      <defs><clipPath id="ilAbove"><rect x="0" y="0" width="270" height="34"/></clipPath></defs>
      <path d="${path(4, 266, f)} L266,80 L4,80 Z" fill="rgba(37,99,235,.08)"/>
      <path d="${path(4, 266, f)} L266,34 L4,34 Z" fill="rgba(225,29,72,.35)" clip-path="url(#ilAbove)"/>
      <path d="${path(4, 266, f)}" fill="none" stroke="${C.coad}" stroke-width="1.5"/>
      <line x1="4" x2="266" y1="34" y2="34" stroke="${C.th}" stroke-dasharray="5 3"/>
      ${T(262, 30, 'ngưỡng', { anchor: 'end' })}<text x="135" y="12" font-size="13" text-anchor="middle" class="pulse"></text>
      ${T(135, 93, 'điểm bất thường vượt ngưỡng → báo động', { size: 9 })}${scan(4, 266, 4, 80)}`, 'Score vượt ngưỡng thì báo động');
  }

  /* reconstruction follows normal parts and departs at the anomaly */
  function recon() {
    const base = x => 40 + 16 * Math.sin(x / 9);
    const real = x => base(x) + (x > 140 && x < 190 ? 20 * Math.sin((x - 140) / 50 * Math.PI) * Math.sin(x / 2) : 0);
    let gap = ''; for (let x = 140; x <= 190; x += 2) gap += (gap ? 'L' : 'M') + x + ',' + real(x).toFixed(1);
    for (let x = 190; x >= 140; x -= 2) gap += 'L' + x + ',' + base(x).toFixed(1);
    return svg(270, 96, `
      <path d="${gap}Z" fill="rgba(225,29,72,.18)"/>
      <path d="${path(4, 266, real)}" fill="none" stroke="${C.sig}" stroke-width="1.4"/>
      <path d="${path(4, 266, base)}" fill="none" stroke="${C.rec}" stroke-width="1.6" stroke-dasharray="5 3" class="motion">
        <animate attributeName="stroke-dashoffset" values="0;-16" dur="1.2s" repeatCount="indefinite"/></path>
      ${T(165, 12, 'lệch xa = bất thường', { fill: C.bad, w: 600 })}
      <line x1="10" x2="24" y1="88" y2="88" stroke="${C.sig}" stroke-width="1.4"/>${T(28, 91, 'dữ liệu thật', { anchor: 'start' })}
      <line x1="100" x2="114" y1="88" y2="88" stroke="${C.rec}" stroke-width="1.6" stroke-dasharray="4 2"/>${T(118, 91, 'mô hình vẽ lại phần "bình thường"', { anchor: 'start' })}`,
      'Mô hình vẽ lại dữ liệu; chỗ lệch xa là bất thường');
  }

  /* smoothing pulls a short spike below the threshold (main error-analysis finding) */
  function smoothing() {
    const N = 64, xs = Array.from({ length: N }, (_, i) => 6 + i * 4);
    const raw = xs.map((x, i) => 70 - (i === 30 || i === 31 ? 50 : 0) - 4 * Math.abs(Math.sin(i)));
    const k = 9, sm = raw.map((_, i) => { let s = 0, c = 0; for (let j = i - 4; j <= i + 4; j++) if (j >= 0 && j < N) { s += raw[j]; c++; } return s / c; });
    const d = ys => ys.map((y, i) => (i ? 'L' : 'M') + xs[i] + ',' + y.toFixed(1)).join('');
    return svg(270, 100, `
      <line x1="4" x2="266" y1="40" y2="40" stroke="${C.th}" stroke-dasharray="5 3"/>${T(262, 36, 'ngưỡng', { anchor: 'end' })}
      <path d="${d(raw)}" fill="none" stroke="${C.coad}" stroke-width="1.6" class="motion">
        <animate attributeName="d" values="${d(raw)};${d(raw)};${d(sm)};${d(sm)};${d(raw)}" keyTimes="0;0.2;0.5;0.85;1" dur="5s" repeatCount="indefinite"/></path>
      ${T(130, 96, 'đỉnh ngắn vượt ngưỡng → sau khi làm mượt (trung bình trượt) bị kéo xuống dưới ngưỡng', { size: 8.5 })}`,
      'Làm mượt kéo đỉnh ngắn xuống dưới ngưỡng');
  }

  /* CoAD pipeline: two parallel score branches, a dot travels along the flow */
  function pipeline() {
    const box = (x, y, w, t1, t2, fill = '#F8FAFC', stroke = '#CBD5E1') => `<rect x="${x}" y="${y}" width="${w}" height="38" rx="7" fill="${fill}" stroke="${stroke}"/>
      ${T(x + w / 2, y + 16, t1, { size: 10, fill: '#0F172A', w: 600 })}${T(x + w / 2, y + 29, t2, { size: 8.5 })}`;
    const arrow = (x1, y1, x2, y2) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#94A3B8" stroke-width="1.4" marker-end="url(#ilArr)"/>`;
    const flow = 'M40,29 L520,29 M455,48 L455,96 L560,96';
    return svg(640, 150, `
      <defs><marker id="ilArr" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="#94A3B8"/></marker></defs>
      ${box(4, 10, 70, 'Input', 'cửa sổ T điểm')}${arrow(74, 29, 92, 29)}
      ${box(94, 10, 112, '① Phân loại', 'time + frequency → A')}${arrow(206, 29, 224, 29)}
      ${box(226, 10, 96, '② Che mềm', 'trộn theo A')}${arrow(322, 29, 340, 29)}
      ${box(342, 10, 86, '③ Tái tạo', 'x̂ ≈ x')}${arrow(428, 29, 446, 29)}
      ${box(448, 10, 130, '④ Phân loại residual', 'x − x̂ → Aʳ, A_c')}
      ${arrow(512, 48, 512, 78)}${arrow(385, 48, 385, 78)}
      ${box(320, 80, 130, 'Nhánh tái tạo', '|x − x̂|', '#F0FDFA', '#99F6E4')}${box(462, 80, 116, 'Nhánh phân loại', 'A_c', '#EFF6FF', '#BFDBFE')}
      ${arrow(320, 99, 262, 99)}${arrow(462, 112, 262, 120)}
      ${box(150, 92, 110, 'Gộp + làm mượt', '→ score cuối', '#F5F3FF', '#DDD6FE')}${arrow(150, 111, 122, 111)}
      ${box(28, 92, 92, 'So ngưỡng', 'báo động', '#FFF7ED', '#FED7AA')}
      ${T(470, 145, 'hai nhánh tính song song rồi mới gộp', { size: 8.5 })}
      <circle r="4.5" fill="${C.cal}" class="motion"><animateMotion dur="6s" repeatCount="indefinite"
        path="M40,29 L512,29 L512,99 L262,111 L74,111"/></circle>`,
      'Luồng tính của CoAD');
  }

  /* window split into patches; soft mask mixes embeddings */
  function patches() {
    const p = [0.08, 0.12, 0.1, 0.55, 0.92, 0.6, 0.15, 0.09], col = v => `hsl(${260 - 200 * v},70%,${60 - 15 * v}%)`;
    return svg(270, 100, `
      ${p.map((v, i) => `<rect x="${8 + i * 32}" y="8" width="30" height="26" rx="3" fill="${col(v)}" ${i === 4 ? 'stroke="#0F172A" stroke-width="2" class="pulse"' : ''}/>
        ${T(23 + i * 32, 46, (v * 100).toFixed(0) + '%', { size: 8.5 })}`).join('')}
      ${T(135, 60, '1 window = 8 patch · màu = xác suất bất thường của patch', { size: 8.5 })}
      <rect x="40" y="70" width="190" height="11" rx="3" fill="#CBD5E1"/>
      <rect x="40" y="70" width="175" height="11" rx="3" fill="#A78BFA" class="motion"><animate attributeName="width" values="15;175;175;15" keyTimes="0;0.4;0.8;1" dur="5s" repeatCount="indefinite"/></rect>
      ${T(135, 95, 'patch nghi bất thường → trộn nhiều "mask" (tím), ít input (xám)', { size: 8.5 })}`,
      'Window chia thành patch; soft mask trộn embedding');
  }

  /* recovered / newly missed on the same event */
  function compare() {
    const ev = (x, a, b, name, col) => `<rect x="${x}" y="6" width="56" height="50" fill="${C.gt}"/>
      ${T(x - 6, 24, 'CoAD', { anchor: 'end', fill: C.coad, w: 600 })}${T(x - 6, 46, 'Cal.', { anchor: 'end', fill: C.cal, w: 600 })}
      ${a ? `<rect x="${x + 6}" y="17" width="44" height="8" rx="2" fill="${C.coad}"/>` : T(x + 28, 25, '✗', { fill: C.bad, size: 12 })}
      ${b ? `<rect x="${x + 6}" y="39" width="44" height="8" rx="2" fill="${C.cal}"/>` : T(x + 28, 47, '✗', { fill: C.bad, size: 12 })}
      ${T(x + 28, 72, name, { fill: col, w: 700, size: 10 })}`;
    return svg(270, 80, ev(44, false, true, 'Được cứu', C.ok) + ev(176, true, false, 'Mới bỏ sót', C.bad), 'Được cứu và mới bỏ sót');
  }

  /* delta bars around zero */
  function delta() {
    const v = [-18, -6, 3, 9, 15, 26], y0 = 8;
    return svg(240, 92, `<line x1="120" x2="120" y1="2" y2="74" stroke="#0F172A"/>
      ${v.map((d, i) => `<rect x="${d < 0 ? 120 + d * 3 : 120}" y="${y0 + i * 11}" width="${Math.abs(d) * 3}" height="8" rx="2" fill="${d < 0 ? C.bad : C.ok}" class="motion">
        <animate attributeName="width" values="0;${Math.abs(d) * 3}" dur="1.2s" fill="freeze"/>${d < 0 ? `<animate attributeName="x" values="120;${120 + d * 3}" dur="1.2s" fill="freeze"/>` : ''}</rect>`).join('')}
      ${T(60, 88, '← Our Improvement kém hơn', { fill: C.bad, size: 9 })}${T(180, 88, 'Our Improvement tốt hơn →', { fill: C.ok, size: 9 })}`, 'Biểu đồ chênh lệch');
  }

  const FAMILY = {
    SVDB: ['', 'Điện tâm đồ (ECG)', 'nhịp tim đo liên tục; bất thường là các nhịp tim bất thường'],
    SED: ['', 'Dữ liệu y tế', 'chuỗi tín hiệu lĩnh vực y tế (theo phân loại của TSB-AD)'],
    IOPS: ['', 'Chỉ số dịch vụ web', 'số liệu vận hành (KPI) của dịch vụ Internet; bất thường là sự cố'],
    TODS: ['', 'Dữ liệu tổng hợp', 'chuỗi tạo nhân tạo, bất thường được chèn có chủ đích'],
    MGAB: ['', 'Chuỗi hỗn loạn tổng hợp', 'chuỗi Mackey–Glass nhân tạo, bất thường khó thấy bằng mắt'],
    WSD: ['', 'Chỉ số máy chủ web', 'số liệu vận hành máy chủ web (KPI)'],
    SMD: ['', 'Chỉ số máy chủ', 'số liệu giám sát máy chủ trong trung tâm dữ liệu'],
    NAB: ['', 'Dữ liệu thực tế (NAB)', 'chuỗi thực tế từ bộ Numenta Anomaly Benchmark'],
    NEK: ['', 'Chỉ số mạng', 'số liệu vận hành mạng'],
  };
  const family = f => FAMILY[f] || ['', f, 'chuỗi thời gian đơn biến'];

  window.Illus = { readChart, threshold, recon, smoothing, pipeline, patches, compare, delta, family };
})();
