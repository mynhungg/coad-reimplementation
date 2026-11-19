/* Event and point rules of the CoAD demo. The ONLY implementation: every screen calls these functions,
   so all pages share one definition (spec 6.3, 10). Exported as window.Rules. */
(function () {
  'use strict';

  const RULES = {
    gt_event: 'Đoạn label = 1 liên tiếp (theo nhãn của TSB-AD).',
    predicted_interval: 'Đoạn prediction = 1 liên tiếp; không gộp khoảng trống, không tolerance.',
    prediction: 'prediction[t] = score[t] ≥ ngưỡng, chỉ trên vùng hợp lệ (t ≥ valid_from = một chu kỳ đầu bị bỏ do warm-up).',
    hit: 'Event được phát hiện nếu có ít nhất một điểm prediction trong event (any-overlap).',
    coverage: 'Coverage = số điểm event được dự đoán / độ dài event. Phát hiện một phần: 0 < coverage < 1 (vẫn là hit).',
    false_alarm: 'Đoạn báo động sai: predicted interval không chồng lấp event nào.',
    delay: 'Delay = điểm dự đoán đầu tiên trong event − điểm bắt đầu event; bỏ sót thì N/A.',
    point: 'Point precision/recall tính trên vùng hợp lệ, so prediction với label từng timestamp.',
    peak: 'Peak của event lấy trong ranh giới event, không lấy phần ngữ cảnh hai bên.',
    comparison: 'Được cứu: gốc bỏ sót, cải tiến hit. Mới bỏ sót: gốc hit, cải tiến bỏ sót. Cùng quy tắc hit cho cả hai.',
  };

  function segments(arr, from) {
    const out = [];
    let start = -1;
    for (let i = from || 0; i < arr.length; i++) {
      if (arr[i] && start < 0) start = i;
      if (!arr[i] && start >= 0) { out.push([start, i]); start = -1; }
    }
    if (start >= 0) out.push([start, arr.length]);
    return out;
  }

  function predict(score, threshold, validFrom) {
    const p = new Uint8Array(score.length);
    for (let i = validFrom; i < score.length; i++) p[i] = score[i] >= threshold ? 1 : 0;
    return p;
  }

  function peakIn(signal, a, b) {
    let best = -Infinity, idx = a;
    for (let i = a; i < b; i++) if (signal[i] > best) { best = signal[i]; idx = i; }
    return { value: best, index: idx };
  }

  /* label: Float32Array 0/1; pred: Uint8Array; validFrom: first valid index */
  function evaluate(label, pred, validFrom, score) {
    const gt = segments(label, 0);
    const events = gt.map(([a, b], k) => {
      let covered = 0, first = -1;
      for (let i = a; i < b; i++) if (pred[i]) { covered++; if (first < 0) first = i; }
      const coverage = covered / (b - a);
      const status = covered === 0 ? 'missed' : coverage < 1 ? 'partial' : 'hit';
      const ev = { id: 'E' + String(k + 1).padStart(3, '0'), a, b, length: b - a, coverage, hit: covered > 0,
                   status, delay: first < 0 ? null : first - a, invalidPart: a < validFrom };
      if (score) ev.peak = peakIn(score, a, b);
      return ev;
    });
    const inEvent = new Uint8Array(label.length);
    gt.forEach(([a, b]) => { for (let i = a; i < b; i++) inEvent[i] = 1; });
    const intervals = segments(pred, validFrom);
    const falseAlarms = [];
    intervals.forEach(([a, b]) => {
      let overlap = false;
      for (let i = a; i < b; i++) if (inEvent[i]) { overlap = true; break; }
      if (!overlap) {
        const fa = { id: 'F' + String(falseAlarms.length + 1).padStart(3, '0'), a, b, length: b - a };
        if (score) fa.peak = peakIn(score, a, b);
        falseAlarms.push(fa);
      }
    });
    let tp = 0, fp = 0, fn = 0, flagged = 0;
    for (let i = validFrom; i < label.length; i++) {
      const y = label[i] > 0.5, p = pred[i] === 1;
      if (p) flagged++;
      if (p && y) tp++; else if (p && !y) fp++; else if (!p && y) fn++;
    }
    const nHit = events.filter(e => e.hit).length;
    return {
      events, falseAlarms, intervals,
      point: { precision: tp + fp ? tp / (tp + fp) : null, recall: tp + fn ? tp / (tp + fn) : null, tp, fp, fn },
      eventRecall: events.length ? nHit / events.length : null,
      nHit, nMissed: events.length - nHit, flagged, flaggedShare: flagged / Math.max(1, label.length - validFrom),
    };
  }

  /* original / improved: results of evaluate() on the same label */
  function compare(original, improved) {
    return original.events.map((e, k) => {
      const o = e.hit, n = improved.events[k].hit;
      const kind = o && n ? 'both_hit' : !o && n ? 'recovered' : o && !n ? 'lost' : 'both_missed';
      return { event: e, original: e, improved: improved.events[k], kind };
    });
  }

  window.Rules = { RULES, segments, predict, evaluate, compare, peakIn };
})();
