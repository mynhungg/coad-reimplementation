/* VI/EN switch for the demo. The UI is written in Vietnamese; in English mode every rendered text node, tooltip and
   chart label is looked up in i18n_en.json. Numbers are kept: the key replaces each number by '#', and the English
   template's '#' are filled with the original numbers in order. Exported as window.I18N. */
(function () {
  'use strict';
  const KEY = 'coad-lang';
  let lang = 'en';
  try { lang = localStorage.getItem(KEY) || 'en'; } catch (e) { /* storage blocked */ }
  const NUM = /\d+(?:[.,]\d+)*/g;
  const VI = /[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]/i;   // Vietnamese letters only
  let dict = null;
  const seen = new Set();                       // fragments with no translation yet (for maintenance)

  function tr(text) {
    if (!dict || !text) return text;
    const lead = text.match(/^\s*/)[0], trail = text.match(/\s*$/)[0], core = text.trim();
    if (!core) return text;
    if (dict[core] !== undefined) return lead + dict[core] + trail;
    const nums = core.match(NUM);
    if (nums) {
      const key = core.replace(NUM, '#');
      if (dict[key] !== undefined) { let k = 0; return lead + dict[key].replace(/#/g, () => (k < nums.length ? nums[k++] : '#')) + trail; }
    }
    if (VI.test(core) && dict.__subs__) {           // fallback: phrase substitutions inside longer, variable texts
      let out = core; for (const [v, e] of dict.__subs__) out = out.split(v).join(e);
      if (out !== core) { if (VI.test(out)) seen.add(out.replace(NUM, '#')); return lead + out + trail; }
    }
    if (VI.test(core)) seen.add(nums ? core.replace(NUM, '#') : core);
    return text;
  }
  const SKIP = new Set(['SCRIPT', 'STYLE', 'TEXTAREA', 'CODE', 'PRE']);
  function walk(node) {
    if (node.nodeType === 3) { const t = tr(node.nodeValue); if (t !== node.nodeValue) node.nodeValue = t; return; }
    if (node.nodeType !== 1 || SKIP.has(node.tagName)) return;
    for (const a of ['title', 'placeholder', 'aria-label', 'data-title']) {
      const v = node.getAttribute && node.getAttribute(a);
      if (v) { const t = tr(v); if (t !== v) node.setAttribute(a, t); }
    }
    if (node.tagName === 'OPTION' || node.tagName === 'SELECT' || node.childNodes.length) node.childNodes.forEach(walk);
  }
  function start() {
    document.documentElement.lang = 'en';
    walk(document.body);
    new MutationObserver(ms => ms.forEach(m => {
      if (m.type === 'characterData') walk(m.target);
      else if (m.type === 'attributes') walk(m.target);
      else m.addedNodes.forEach(walk);
    })).observe(document.body, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ['title', 'placeholder', 'aria-label'] });
  }
  function setLang(l) { try { localStorage.setItem(KEY, l); } catch (e) { /* storage blocked */ } location.reload(); }

  /* Plotly measures legend and label widths when it draws, so translating its SVG afterwards makes long English labels
     overlap. Translate the figure spec itself before Plotly sees it. */
  const PKEYS = new Set(['name', 'text', 'title', 'ticktext', 'hovertext', 'hovertemplate']);
  function trFig(o, k) {
    if (typeof o === 'string') return PKEYS.has(k) ? tr(o) : o;
    if (Array.isArray(o)) return (PKEYS.has(k) && o.length && typeof o[0] === 'string') ? o.map(x => tr(x)) : o.map(x => trFig(x, k));
    if (!o || typeof o !== 'object' || ArrayBuffer.isView(o)) return o;
    const out = {}; for (const key in o) out[key] = trFig(o[key], PKEYS.has(key) ? key : (k === 'title' ? 'title' : key)); return out;
  }
  function patchPlotly() {
    if (!window.Plotly || Plotly.__i18n) return;
    for (const fn of ['newPlot', 'react']) {
      const orig = Plotly[fn];
      Plotly[fn] = function (el, data, layout, cfg) {
        if (dict && data && !Array.isArray(data) && typeof data === 'object' && data.data) return orig.call(this, el, trFig(data));
        return orig.call(this, el, dict ? trFig(data) : data, dict ? trFig(layout) : layout, cfg);
      };
    }
    Plotly.__i18n = true;
  }

  window.I18N = { lang: () => lang, setLang, missing: () => [...seen].sort(), tr: s => tr(s) };
  if (lang === 'en') {
    fetch('i18n_en.json?v=13').then(r => r.json()).then(d => { dict = d; patchPlotly(); start(); }).catch(() => { /* no dictionary: stay Vietnamese */ });
  }
  document.addEventListener('DOMContentLoaded', () => {
    const b = document.getElementById('langBtn');
    if (b) { b.textContent = lang === 'en' ? 'Tiếng Việt' : 'English'; b.onclick = () => setLang(lang === 'en' ? 'vi' : 'en'); }
  });
})();
