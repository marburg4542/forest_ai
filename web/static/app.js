/* forest_ai web interface — no framework, no build step.
   Everything renders from the JSON served by web/server.py.  Plotly figures
   arrive already built by forest_ai/webviz.py, with their numeric arrays
   base64-encoded, so they are handed to Plotly.newPlot untouched. */

'use strict';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const VIEWS = ['Overview', 'Stem map', '3D view', 'Trees',
               'Species', 'Validation', 'QC figures'];

const state = {
  view: 'Overview',
  trees: null,
  summary: null,
  polling: null,
  cloudOpts: { n: 150000, color_by: 'tree', ground: false, size: 1.4 },
  stemColour: 'quality',
  filters: { quality: new Set(['good', 'fair']), dbhMin: null, dbhMax: null, arcMin: 0 },
  sort: { key: 'tree_id', dir: 1 },
  selectedTree: null,
  evalColumns: null,
  evalFile: null,
  serverConfig: null,
};

/* ------------------------------------------------------------------ utils */

/** Identifies this browser to the server so results, uploads and jobs stay
 *  separate between people using the same deployment.  A header rather than a
 *  cookie: the page can be embedded in an iframe on huggingface.co, where
 *  third-party cookies may be blocked, and a header we set ourselves is not. */
function sessionId() {
  let id = null;
  try { id = localStorage.getItem('fai_sid'); } catch (_) {}
  if (!id) {
    id = (crypto.randomUUID ? crypto.randomUUID() : String(Math.random())).replace(/-/g, '');
    try { localStorage.setItem('fai_sid', id); } catch (_) {}
  }
  return id;
}
const SID = sessionId();

async function api(path, opts = {}) {
  const headers = new Headers(opts.headers || {});
  headers.set('X-Session-Id', SID);
  const res = await fetch(path, { ...opts, headers });
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).error || msg; } catch (_) {}
    throw new Error(msg);
  }
  return res;
}
const getJSON = async (p) => (await api(p)).json();

function num(v, digits = 1) {
  if (v === null || v === undefined || Number.isNaN(v)) return '–';
  return Number(v).toFixed(digits);
}
function esc(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}
function html(target, markup) { target.innerHTML = markup; return target; }

/** Plotly figures are drawn into a div that must already be laid out, so this
 *  is always called after the view markup is in the DOM. */
function draw(id, fig) {
  const el = document.getElementById(id);
  if (!el) return;
  Plotly.newPlot(el, fig.data, fig.layout,
                 { responsive: true, displaylogo: false });
}

function loading(msg = 'loading…') { return `<div class="loading">${esc(msg)}</div>`; }

/* --------------------------------------------------------------- sidebar */
async function loadServerConfig() {
  try {
    state.serverConfig = await getJSON('/api/config');
  } catch (_) { state.serverConfig = {}; }
  const c = state.serverConfig;
  const limits = $('#limits');
  if (limits) {
    limits.textContent =
      `This deployment accepts uploads up to ${c.max_upload_mb} MB and keeps ` +
      `${c.max_sessions} result${c.max_sessions === 1 ? '' : 's'} in memory at a time — ` +
      `when a newer visitor takes the slot, your result and uploaded file are ` +
      `discarded, so download anything you want to keep. A run takes a couple of ` +
      `minutes on a small server.`;
  }
}

async function loadClouds() {
  const { clouds, current } = await getJSON('/api/clouds');
  const sel = $('#cloud');
  sel.innerHTML = clouds.map(c => `<option ${c === current ? 'selected' : ''}>${esc(c)}</option>`).join('')
    || '<option disabled>upload a .las or .laz to begin</option>';
  $('#run').disabled = clouds.length === 0;
  const hint = $('#start-hint');
  if (hint) hint.hidden = clouds.length > 0;
  if (!state.summary) showHeaderPreview();
}

async function showHeaderPreview() {
  const las = $('#cloud').value;
  const box = $('#header-preview');
  if (!box || !las) return;
  try {
    const h = await getJSON(`/api/header?las=${encodeURIComponent(las)}`);
    html(box, `<div class="surface" style="text-align:left;margin-top:20px">
      <h3 style="margin:0 0 8px;font-size:13px">${esc(las)}</h3>
      <dl class="kv">
        <dt>points</dt><dd>${h.point_count.toLocaleString()}</dd>
        <dt>LAS version</dt><dd>${esc(h.version)} · format ${h.point_format}</dd>
        <dt>extent x</dt><dd>${num(h.mins[0])} … ${num(h.maxs[0])} m</dd>
        <dt>extent y</dt><dd>${num(h.mins[1])} … ${num(h.maxs[1])} m</dd>
        <dt>extent z</dt><dd>${num(h.mins[2])} … ${num(h.maxs[2])} m</dd>
        <dt>extra dims</dt><dd>${h.extra_dims.length ? esc(h.extra_dims.join(', ')) : '–'}</dd>
        <dt>CRS</dt><dd>${h.crs && h.crs !== 'None' ? esc(h.crs) : 'none — coordinates are local'}</dd>
      </dl></div>`);
  } catch (e) { html(box, `<p class="error">${esc(e.message)}</p>`); }
}

async function loadParams() {
  const { groups } = await getJSON('/api/params');
  $('#params').innerHTML = groups.map((g, i) => `
    <details class="group" ${i === 0 ? 'open' : ''}>
      <summary>${esc(g.group)}</summary>
      <div>${g.controls.map(c => `
        <div class="control">
          <label for="p_${c.name}" title="${esc(c.help)}">${esc(c.label)}</label>
          <div class="slider-row">
            <input type="range" id="p_${c.name}" data-param="${c.name}"
                   min="${c.min}" max="${c.max}" step="${c.step}" value="${c.value}">
            <output for="p_${c.name}">${c.value}</output>
          </div>
        </div>`).join('')}</div>
    </details>`).join('');

  $$('#params input[type=range]').forEach(inp => {
    inp.addEventListener('input', () => {
      inp.parentElement.querySelector('output').textContent = inp.value;
    });
  });
}

function collectConfig() {
  const cfg = {};
  $$('#params input[data-param]').forEach(i => { cfg[i.dataset.param] = parseFloat(i.value); });
  return cfg;
}

/* ------------------------------------------------------------------- run */
async function startRun() {
  $('#run-error').hidden = true;
  $('#run').disabled = true;
  $('#progress').hidden = false;
  try {
    await api('/api/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        las: $('#cloud').value,
        config: collectConfig(),
        groups: parseInt($('#groups').value, 10),
        force: $('#force').checked,
      }),
    });
    poll();
  } catch (e) {
    $('#run').disabled = false;
    $('#progress').hidden = true;
    showRunError(e.message);
  }
}

function showRunError(msg) {
  const box = $('#run-error');
  box.textContent = msg;
  box.hidden = false;
}

function poll() {
  clearInterval(state.polling);
  state.polling = setInterval(async () => {
    let job;
    try { job = await getJSON('/api/job'); } catch (_) { return; }
    $('#progress-fill').style.width = `${Math.round(job.progress * 100)}%`;
    $('#progress-msg').textContent = job.message || '';
    if (job.state === 'running') return;

    clearInterval(state.polling);
    $('#run').disabled = false;
    if (job.state === 'error') {
      $('#progress').hidden = true;
      showRunError(job.error || 'pipeline failed');
      return;
    }
    if (job.state === 'done') {
      setTimeout(() => { $('#progress').hidden = true; }, 900);
      await onResultReady(job.las);
    }
  }, 500);
}

async function onResultReady(las) {
  state.trees = null;
  state.summary = null;
  state.selectedTree = null;
  $('#title').textContent = `🌲 ${las}`;
  buildNav();
  await loadClouds();
  show('Overview');
}

/* ------------------------------------------------------------------- nav */
function buildNav() {
  $('#nav').innerHTML = VIEWS.map(v =>
    `<button class="nav-btn" data-view="${v}" aria-current="${v === state.view}">${v}</button>`).join('');
  $$('#nav .nav-btn').forEach(b => b.addEventListener('click', () => show(b.dataset.view)));
}

async function show(view) {
  state.view = view;
  $$('#nav .nav-btn').forEach(b => b.setAttribute('aria-current', String(b.dataset.view === view)));
  const root = $('#view');
  html(root, loading());
  try {
    await RENDER[view](root);
  } catch (e) {
    html(root, `<p class="error">${esc(e.message)}</p>`);
  }
}

/* ------------------------------------------------------------ data cache */
async function trees() {
  if (!state.trees) state.trees = await getJSON('/api/trees');
  return state.trees;
}
async function summary() {
  if (!state.summary) state.summary = await getJSON('/api/summary');
  return state.summary;
}

/* ---------------------------------------------------------------- views */
const RENDER = {};

RENDER['Overview'] = async (root) => {
  const s = await summary();
  const m = s.summary, mr = s.median_range_by_quality || {};
  const corr = m['height-DBH corr good-only'];

  html(root, `
    <div class="cards">
      ${card('trees detected', m['trees detected'],
             `${m.good} good · ${m.fair} fair · ${m.poor} poor`)}
      ${card('stem density', `${num(m['stem density all (/ha)'], 0)} /ha`,
             `over ${num(m['stocked area (ha)'], 3)} ha stocked`)}
      ${card('DBH median', `${num(m['DBH median (cm)'])} cm`,
             `good-only mean ${num(m['DBH good-only mean (cm)'])} cm`)}
      ${card('basal area', `${num(m['basal area (m2/ha)'])} m²/ha`, 'good + fair only')}
      ${card('height median', `${num(m['height median (m)'])} m`,
             `mean ${num(m['height mean (m)'])} m`)}
      ${card('valid DTM area', `${num(m['valid DTM area (ha)'], 3)} ha`, 'of the whole file')}
      ${card('height–DBH corr (good)', (corr >= 0 ? '+' : '') + num(corr, 2),
             `all trees ${(m['height-DBH corr all'] >= 0 ? '+' : '') + num(m['height-DBH corr all'], 2)}`)}
      ${card('median range: good', `${num(mr.good, 0)} m`, `poor at ${num(mr.poor, 0)} m`)}
    </div>

    ${corr !== null && corr < 0.2 ? `<div class="warn">
      The height–diameter correlation among <strong>good</strong> trees is weak.
      In a real stand it should be clearly positive — check the DBH fits and
      consider raising the arc-coverage threshold.</div>` : ''}

    <div class="section">
      <h3>distributions &amp; occlusion</h3>
      <div class="surface"><div class="plot" id="plot-inventory"></div></div>
    </div>

    <div class="section">
      <h3>stand summary</h3>
      <pre class="code">${esc(s.text)}</pre>
    </div>

    <div class="section">
      <h3>outputs</h3>
      <div class="toolbar">
        <button class="btn btn-sm" id="dl-trees">trees.csv</button>
        <button class="btn btn-sm" id="dl-features">tree_features.csv</button>
        <button class="btn btn-sm" id="write-out">write to the server</button>
        <span class="hint" id="write-status"></span>
      </div>
      <p class="note">“Write to the server” saves the CSVs and the five QC figures
        server-side${state.serverConfig?.segmented_las
          ? ', plus the segmented point cloud (about 250 MB)' : ''}, so the QC view
        can show them. The two buttons on the left download straight to your
        machine and do not touch the server.</p>
    </div>`);

  draw('plot-inventory', await getJSON('/api/figure/inventory'));

  $('#dl-trees').addEventListener('click', () =>
    downloadFromApi('/api/download/trees.csv', 'trees.csv'));
  $('#dl-features').addEventListener('click', () =>
    downloadFromApi('/api/download/features.csv', 'tree_features.csv'));

  $('#write-out').addEventListener('click', async (e) => {
    e.target.disabled = true;
    $('#write-status').textContent = 'writing…';
    try {
      const r = await (await api('/api/write_outputs', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })).json();
      $('#write-status').textContent = `wrote ${r.written.join(', ')} to ${r.dir}`;
    } catch (err) {
      $('#write-status').textContent = err.message;
    } finally { e.target.disabled = false; }
  });
};

const card = (k, v, s = '') =>
  `<div class="card"><div class="k">${esc(k)}</div><div class="v">${esc(v)}</div>
   <div class="s">${esc(s)}</div></div>`;

RENDER['Stem map'] = async (root) => {
  const opts = [['quality', 'quality'], ['dbh_cm', 'DBH'], ['height_m', 'height'],
                ['dbh_arc', 'arc coverage'],
                ['dist_from_scan_centre_m', 'range from scanner'],
                ['structural_group', 'structural group']];
  html(root, `
    <div class="toolbar">
      <div class="control"><label for="stem-colour">colour by</label>
        <select id="stem-colour">${opts.map(([v, l]) =>
          `<option value="${v}" ${v === state.stemColour ? 'selected' : ''}>${l}</option>`).join('')}</select>
      </div>
    </div>
    <div class="surface"><div class="plot" id="plot-stemmap" style="min-height:640px"></div></div>
    <p class="note">Marker size is proportional to DBH. The dotted outline is the
      stocked area used for the per-hectare figures. Detection thins out with
      distance from the scanner — that is occlusion, not an algorithm failure.</p>`);

  const load = async () => draw('plot-stemmap',
    await getJSON(`/api/figure/stemmap?colour=${encodeURIComponent(state.stemColour)}`));
  $('#stem-colour').addEventListener('change', (e) => {
    state.stemColour = e.target.value; load();
  });
  await load();
};

RENDER['3D view'] = async (root) => {
  const o = state.cloudOpts;
  html(root, `
    <div class="toolbar">
      <div class="control"><label for="c-n">points shown</label>
        <select id="c-n">${[25000, 50000, 100000, 150000, 250000, 400000].map(v =>
          `<option value="${v}" ${v === o.n ? 'selected' : ''}>${v.toLocaleString()}</option>`).join('')}</select>
      </div>
      <div class="control"><label for="c-colour">colour by</label>
        <select id="c-colour">${['tree', 'height', 'quality'].map(v =>
          `<option ${v === o.color_by ? 'selected' : ''}>${v}</option>`).join('')}</select>
      </div>
      <div class="control"><label for="c-size">point size</label>
        <div class="slider-row"><input type="range" id="c-size" min="0.6" max="4" step="0.2" value="${o.size}">
        <output>${o.size}</output></div>
      </div>
      <label class="check"><input type="checkbox" id="c-ground" ${o.ground ? 'checked' : ''}>
        include ground / unassigned</label>
    </div>
    <div class="surface"><div class="plot" id="plot-cloud" style="min-height:660px"></div></div>
    <p class="note" id="cloud-note"></p>`);

  const load = async () => {
    const note = $('#cloud-note');
    note.textContent = 'building the 3-D view…';
    const t0 = performance.now();
    const res = await api(`/api/figure/cloud?n=${o.n}&color_by=${o.color_by}` +
                          `&ground=${o.ground}&size=${o.size}`);
    const shown = res.headers.get('X-Points-Shown');
    const total = res.headers.get('X-Points-Total');
    draw('plot-cloud', await res.json());
    note.innerHTML = `Showing a random ${Number(shown).toLocaleString()} of
      ${Number(total).toLocaleString()} assigned points. Drag to orbit, scroll to zoom
      (${((performance.now() - t0) / 1000).toFixed(1)} s).
      For the full-resolution cloud open <code>outputs/Forest_segmented.las</code>
      in CloudCompare and colour by the <code>tree_id</code> scalar field.`;
  };

  $('#c-n').addEventListener('change', e => { o.n = +e.target.value; load(); });
  $('#c-colour').addEventListener('change', e => { o.color_by = e.target.value; load(); });
  $('#c-ground').addEventListener('change', e => { o.ground = e.target.checked; load(); });
  $('#c-size').addEventListener('change', e => {
    o.size = +e.target.value;
    e.target.parentElement.querySelector('output').textContent = o.size;
    load();
  });
  $('#c-size').addEventListener('input', e => {
    e.target.parentElement.querySelector('output').textContent = e.target.value;
  });
  await load();
};

const TABLE_COLS = [
  ['tree_id', 'id', 0], ['quality', 'quality', null], ['dbh_cm', 'DBH cm', 1],
  ['height_m', 'height m', 1], ['crown_diameter_m', 'crown m', 1],
  ['crown_base_m', 'crown base m', 1], ['dbh_arc', 'arc', 2],
  ['dbh_rmse_cm', 'resid cm', 1], ['n_stem_layers', 'layers', 0],
  ['stem_lean_deg', 'lean °', 1], ['dist_from_scan_centre_m', 'range m', 1],
  ['structural_group', 'group', 0], ['x', 'x', 2], ['y', 'y', 2],
];

RENDER['Trees'] = async (root) => {
  const { rows } = await trees();
  const maxDbh = Math.ceil(Math.max(...rows.map(r => r.dbh_cm || 0)));
  const f = state.filters;
  if (f.dbhMax === null) { f.dbhMin = 0; f.dbhMax = maxDbh; }

  html(root, `
    <div class="toolbar">
      <div class="control"><label>quality</label>
        <div class="chips">${['good', 'fair', 'poor'].map(q =>
          `<label><input type="checkbox" data-q="${q}" ${f.quality.has(q) ? 'checked' : ''}>
           <span class="pill ${q}">${q}</span></label>`).join('')}</div>
      </div>
      <div class="control"><label for="f-dbh-min">DBH from (cm)</label>
        <div class="slider-row"><input type="range" id="f-dbh-min" min="0" max="${maxDbh}" step="1" value="${f.dbhMin}">
          <output>${f.dbhMin}</output></div></div>
      <div class="control"><label for="f-dbh-max">DBH to (cm)</label>
        <div class="slider-row"><input type="range" id="f-dbh-max" min="0" max="${maxDbh}" step="1" value="${f.dbhMax}">
          <output>${f.dbhMax}</output></div></div>
      <div class="control"><label for="f-arc">min arc coverage</label>
        <div class="slider-row"><input type="range" id="f-arc" min="0" max="1" step="0.05" value="${f.arcMin}">
          <output>${f.arcMin}</output></div></div>
      <button class="btn btn-sm" id="dl-filtered">download selection</button>
    </div>
    <p class="hint" id="tree-count"></p>
    <div class="table-wrap"><table id="tree-table"></table></div>
    <div class="section" style="margin-top:22px">
      <h3>inspect one tree</h3>
      <div class="split">
        <div class="surface"><div class="plot" id="plot-slice" style="min-height:420px"></div></div>
        <div class="surface"><div class="plot" id="plot-tree3d" style="min-height:420px"></div></div>
      </div>
      <p class="note">Grey points in the slice lie outside the search radius around
        the stem axis and were excluded from the fit. Click any row to inspect it.</p>
    </div>`);

  const bind = (id, key, digits) => {
    const inp = $(id);
    inp.addEventListener('input', () => {
      inp.parentElement.querySelector('output').textContent = inp.value;
      state.filters[key] = parseFloat(inp.value);
      renderTable();
    });
  };
  bind('#f-dbh-min', 'dbhMin'); bind('#f-dbh-max', 'dbhMax'); bind('#f-arc', 'arcMin');
  $$('input[data-q]').forEach(cb => cb.addEventListener('change', () => {
    cb.checked ? f.quality.add(cb.dataset.q) : f.quality.delete(cb.dataset.q);
    renderTable();
  }));
  $('#dl-filtered').addEventListener('click', downloadFiltered);

  renderTable();
};

function filtered() {
  const f = state.filters;
  return state.trees.rows.filter(r =>
    f.quality.has(r.quality) &&
    (r.dbh_cm ?? 0) >= f.dbhMin && (r.dbh_cm ?? 0) <= f.dbhMax &&
    (r.dbh_arc ?? 0) >= f.arcMin);
}

function renderTable() {
  const rows = filtered();
  const { key, dir } = state.sort;
  rows.sort((a, b) => {
    const x = a[key], y = b[key];
    if (x === null || x === undefined) return 1;
    if (y === null || y === undefined) return -1;
    return (x > y ? 1 : x < y ? -1 : 0) * dir;
  });

  $('#tree-count').textContent = `${rows.length} of ${state.trees.rows.length} trees`;
  $('#tree-table').innerHTML = `
    <thead><tr>${TABLE_COLS.map(([k, label, d]) =>
      `<th data-key="${k}" class="${d === null ? '' : 'num'}">${esc(label)}${
        key === k ? (dir > 0 ? ' ▲' : ' ▼') : ''}</th>`).join('')}</tr></thead>
    <tbody>${rows.slice(0, 500).map(r => `
      <tr data-id="${r.tree_id}" class="${r.tree_id === state.selectedTree ? 'sel' : ''}">
        ${TABLE_COLS.map(([k, , d]) => d === null
          ? `<td><span class="pill ${esc(r[k])}">${esc(r[k])}</span></td>`
          : `<td class="num">${num(r[k], d)}</td>`).join('')}
      </tr>`).join('')}</tbody>`;

  $$('#tree-table th').forEach(th => th.addEventListener('click', () => {
    const k = th.dataset.key;
    state.sort = { key: k, dir: state.sort.key === k ? -state.sort.dir : 1 };
    renderTable();
  }));
  $$('#tree-table tbody tr').forEach(tr => tr.addEventListener('click', () => {
    selectTree(parseInt(tr.dataset.id, 10));
  }));

  if (rows.length && !rows.some(r => r.tree_id === state.selectedTree)) {
    selectTree(rows[0].tree_id);
  }
}

async function selectTree(id) {
  state.selectedTree = id;
  $$('#tree-table tbody tr').forEach(tr =>
    tr.classList.toggle('sel', +tr.dataset.id === id));
  const [slice, tree3d] = await Promise.all([
    getJSON(`/api/figure/dbh_slice/${id}`),
    getJSON(`/api/figure/tree3d/${id}`),
  ]);
  draw('plot-slice', slice);
  draw('plot-tree3d', tree3d);
}

function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/** Anchor links and <img src> cannot carry the X-Session-Id header, so
 *  anything session-scoped has to be fetched and handed over as a blob. */
async function downloadFromApi(path, filename) {
  saveBlob(await (await api(path)).blob(), filename);
}

function downloadFiltered() {
  const rows = filtered();
  const cols = state.trees.columns;
  const csv = [cols.join(',')].concat(rows.map(r =>
    cols.map(c => {
      const v = r[c];
      return v === null || v === undefined ? '' : String(v);
    }).join(','))).join('\n');
  saveBlob(new Blob([csv], { type: 'text/csv' }), 'trees_filtered.csv');
}

RENDER['Species'] = async (root) => {
  const sp = await getJSON('/api/species');
  html(root, `
    <div class="warn">There are <strong>no species labels yet</strong>, so nothing
      here is a species prediction. <code>structural_group</code> is a K-means
      grouping of the 51 geometric features — two species with the same
      architecture land in one group, and one species splits across groups when
      some individuals are suppressed. Its use is to cut field work: identify
      10–15 trees per group instead of all of them.</div>
    ${sp.available ? `
      <div class="section"><h3>group profiles (medians)</h3>
        <div class="table-wrap"><table>
          <thead><tr><th>group</th><th class="num">n</th><th class="num">DBH cm</th>
            <th class="num">height m</th><th class="num">crown m</th>
            <th class="num">crown base m</th><th class="num">h/d</th></tr></thead>
          <tbody>${sp.profile.map(p => `<tr>
            <td>g${p.structural_group}</td><td class="num">${p.n}</td>
            <td class="num">${num(p.dbh_cm)}</td><td class="num">${num(p.height_m)}</td>
            <td class="num">${num(p.crown_diameter_m)}</td>
            <td class="num">${num(p.crown_base_m)}</td>
            <td class="num">${num(p.h_d_ratio)}</td></tr>`).join('')}</tbody>
        </table></div>
      </div>
      <div class="section"><h3>groups in plan view</h3>
        <div class="surface"><div class="plot" id="plot-groups" style="min-height:600px"></div></div>
      </div>` : '<p class="note">clustering was skipped for this run</p>'}

    <div class="section"><h3>training a real species model</h3>
      <p class="note">Make a <code>labels.csv</code> with <code>tree_id,species</code>, then:</p>
      <pre class="code">import pandas as pd
from forest_ai import features

feat = pd.read_csv('outputs/tree_features.csv', index_col='tree_id')
lab  = pd.read_csv('labels.csv', index_col='tree_id')['species']
res  = features.train_species_model(feat, lab)   # 5-fold cross-validated
print(res['cv_report'])
print(res['importance'].head(15))</pre>
    </div>`);

  if (sp.available) draw('plot-groups', await getJSON('/api/figure/stemmap?colour=structural_group'));
};

RENDER['Validation'] = async (root) => {
  html(root, `
    <p class="note" style="margin-bottom:16px">Upload a reference tree list — a field
      survey, or another tool's output (FSCT, TreeLS, TreeLearn). It needs
      <strong>x</strong> and <strong>y</strong> columns in the same coordinates as the
      cloud, plus whatever you want compared (<code>dbh_cm</code>, <code>height_m</code>).</p>
    <div class="toolbar">
      <input type="file" id="ref-file" accept=".csv">
    </div>
    <div id="eval-body"><p class="note">Until this is done the accuracy of the
      numbers in Overview is unknown. 30–50 field-measured trees within ~20 m of
      the scanner is enough to calibrate the quality thresholds.</p></div>`);

  $('#ref-file').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    state.evalFile = file;
    const body = $('#eval-body');
    html(body, loading('reading the CSV…'));
    try {
      const fd = new FormData(); fd.append('file', file);
      const info = await (await api('/api/evaluate/columns', { method: 'POST', body: fd })).json();
      state.evalColumns = info.columns;
      renderEvalForm(body, info);
    } catch (err) { html(body, `<p class="error">${esc(err.message)}</p>`); }
  });
};

function renderEvalForm(body, info) {
  const cols = info.columns;
  const pick = (id, label, want) => `
    <div class="control"><label for="${id}">${label}</label>
      <select id="${id}">${(id === 'm-dbh' || id === 'm-height' ? ['<none>'] : []).concat(cols)
        .map(c => `<option ${c === want ? 'selected' : ''}>${esc(c)}</option>`).join('')}</select></div>`;

  html(body, `
    <div class="toolbar">
      ${pick('m-x', 'x column', 'x')}
      ${pick('m-y', 'y column', 'y')}
      ${pick('m-dbh', 'DBH column (cm)', 'dbh_cm')}
      ${pick('m-height', 'height column (m)', 'height_m')}
      <div class="control"><label for="m-dist">max match distance (m)</label>
        <div class="slider-row"><input type="range" id="m-dist" min="0.5" max="5" step="0.1" value="1.5">
          <output>1.5</output></div></div>
      <div class="control"><label>compare which predictions</label>
        <div class="chips">${['good', 'fair', 'poor'].map(q =>
          `<label><input type="checkbox" data-eq="${q}" ${q === 'good' ? 'checked' : ''}>
           <span class="pill ${q}">${q}</span></label>`).join('')}</div></div>
      <button class="btn btn-sm btn-primary" style="width:auto" id="run-eval">compare</button>
    </div>
    <p class="note">In a dense stand keep the match distance below half the mean
      spacing between stems, or predictions get paired to the wrong reference tree.</p>
    <div id="eval-result"></div>`);

  $('#m-dist').addEventListener('input', e =>
    e.target.parentElement.querySelector('output').textContent = e.target.value);
  $('#run-eval').addEventListener('click', runEval);
}

async function runEval() {
  const out = $('#eval-result');
  html(out, loading('matching…'));
  const mapping = {};
  const put = (id, to) => {
    const v = $(id).value;
    if (v && v !== '<none>' && v !== to) mapping[v] = to;
  };
  put('#m-x', 'x'); put('#m-y', 'y'); put('#m-dbh', 'dbh_cm'); put('#m-height', 'height_m');

  const fd = new FormData();
  fd.append('file', state.evalFile);
  fd.append('mapping', JSON.stringify(mapping));
  fd.append('max_dist', $('#m-dist').value);
  fd.append('quality', $$('input[data-eq]').filter(c => c.checked).map(c => c.dataset.eq).join(','));

  try {
    const r = await (await api('/api/evaluate', { method: 'POST', body: fd })).json();
    const st = r.stats;
    html(out, `
      <div class="cards">
        ${card('F1', num(st.f1, 3))}
        ${card('precision', num(st.precision, 3), `${st.FP} false positives`)}
        ${card('recall', num(st.recall, 3), `${st.FN} missed`)}
        ${card('mean offset', `${num(st.mean_match_dist_m, 2)} m`, `${st.TP} matched`)}
      </div>
      <pre class="code">${esc(r.text)}</pre>
      <div class="split" style="margin-top:16px">
        ${Object.keys(r.errors).map(f =>
          `<div class="surface"><div class="plot" id="plot-agree-${f}" style="min-height:400px"></div></div>`).join('')}
      </div>`);

    for (const [f, e] of Object.entries(r.errors)) {
      const p = r.pairs[f];
      const lim = [Math.min(...p.ref, ...p.pred), Math.max(...p.ref, ...p.pred)];
      draw(`plot-agree-${f}`, {
        data: [
          { x: p.ref, y: p.pred, mode: 'markers', type: 'scatter',
            marker: { size: 8, color: '#3d5a80' }, name: 'trees' },
          { x: lim, y: lim, mode: 'lines', type: 'scatter',
            line: { dash: 'dash', color: '#888' }, name: '1:1' },
        ],
        layout: {
          height: 400, margin: { l: 55, r: 10, t: 45, b: 45 },
          xaxis: { title: { text: `reference ${f}` } },
          yaxis: { title: { text: `predicted ${f}` }, scaleanchor: 'x', scaleratio: 1 },
          title: { text: `${f} — bias ${num(e.bias, 2)}, rmse ${num(e.rmse, 2)}, R² ${num(e.r2, 3)} (n=${e.n})` },
          paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
        },
      });
    }
  } catch (err) { html(out, `<p class="error">${esc(err.message)}</p>`); }
}

RENDER['QC figures'] = async (root) => {
  const captions = {
    'qc_01_ground.png': 'DTM, cross-sections and the height histogram',
    'qc_02_stem_map.png': 'stem map over the DTM',
    'qc_03_inventory.png': 'DBH / height distributions and occlusion vs range',
    'qc_04_segmentation.png': 'tree instances in cross-section — the key check',
    'qc_05_dbh_fits.png': 'real stem slices with the fitted circles',
  };
  const { files } = await getJSON('/api/qc');
  html(root, `
    <div class="toolbar">
      <button class="btn btn-sm" id="gen-qc">regenerate QC figures</button>
      <span class="hint" id="qc-status"></span>
    </div>
    <p class="note" style="margin-bottom:18px"><strong>Look at these before trusting
      any number.</strong> Every bug found while building this pipeline showed up in a
      figure, not in the statistics.</p>
    ${files.length ? files.map(f => `<figure>
        <img class="qc" data-qc="${esc(f)}" alt="${esc(f)}">
        <figcaption>${esc(f)} — ${esc(captions[f] || '')}</figcaption>
      </figure>`).join('')
      : '<p class="note">No figures on the server yet — press the button above.</p>'}`);

  // <img src> cannot send the session header, so fetch each one and swap in a blob
  for (const img of $$('#view img[data-qc]')) {
    try {
      img.src = URL.createObjectURL(await (await api(`/api/qc/${img.dataset.qc}`)).blob());
    } catch (_) { img.replaceWith(Object.assign(document.createElement('p'),
      { className: 'note', textContent: `could not load ${img.dataset.qc}` })); }
  }

  $('#gen-qc').addEventListener('click', async (e) => {
    e.target.disabled = true;
    $('#qc-status').textContent = 'rendering…';
    try {
      await api('/api/write_outputs', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ figures: true, segmented_las: false }),
      });
      show('QC figures');
    } catch (err) {
      $('#qc-status').textContent = err.message;
    } finally { e.target.disabled = false; }
  });
};

/* ------------------------------------------------------------------ init */
async function init() {
  $('#groups').addEventListener('input', e => { $('#groups-out').textContent = e.target.value; });
  $('#run').addEventListener('click', startRun);
  $('#cloud').addEventListener('change', () => { if (!state.summary) showHeaderPreview(); });
  $('#upload-btn').addEventListener('click', () => $('#upload').click());
  $('#upload').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const limit = state.serverConfig?.max_upload_mb;
    if (limit && file.size > limit * 1e6) {
      $('#upload-status').textContent =
        `${(file.size / 1e6).toFixed(0)} MB is over this deployment's ${limit} MB limit`;
      e.target.value = '';
      return;
    }
    $('#upload-status').textContent = `uploading ${(file.size / 1e6).toFixed(0)} MB…`;
    try {
      const fd = new FormData(); fd.append('file', file);
      const r = await (await api('/api/upload', { method: 'POST', body: fd })).json();
      $('#upload-status').textContent = `saved ${r.path}`;
      await loadClouds();
      $('#cloud').value = r.path;
      showHeaderPreview();
    } catch (err) { $('#upload-status').textContent = err.message; }
  });

  await Promise.all([loadServerConfig(), loadClouds(), loadParams()]);

  // a result may already exist from an earlier browser session
  const job = await getJSON('/api/job');
  if (job.state === 'running') { $('#run').disabled = true; $('#progress').hidden = false; poll(); }
  else if (job.has_result) { await onResultReady(job.las); }
}

init();
