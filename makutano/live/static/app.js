// Makutano live console: draws frames streamed by makutano.live.server and sends
// operator commands back. Positions arrive in metres (x east, y south, junction
// centre at 0,0); the canvas maps them to pixels.
(() => {
'use strict';
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;');

const APPS = ['north', 'south', 'east', 'west'];
const SHORT = { north: 'N', south: 'S', east: 'E', west: 'W' };
const NAME = { north: 'North', south: 'South', east: 'East', west: 'West' };
const SCOL = { north: 'var(--s1)', south: 'var(--s2)', east: 'var(--s3)', west: 'var(--s4)' };
const VCOL = { north: '#4d9cf0', south: '#f07a4a', east: '#2fc48d', west: '#f2b225' };  // overlays on the video
const BOX = { car: '#5ec8ff', matatu: '#ffbe2e', boda_boda: '#ff6fa3', bus: '#a5d86a', truck: '#c2a8ff', tuk_tuk: '#ffe36e', person: '#ffffff' };
const PAINT = {
  car: ['#d8dcdf', '#9ba4a9', '#353c42', '#a33434', '#2c5884', '#e6dfcc', '#5d6f5a', '#c9c9c4'],
  matatu: [['#f2f2f0', '#d6262e'], ['#1d1d1f', '#f6c90e'], ['#f4f0e6', '#1f8f4e'], ['#7a2db0', '#f2f2f2'], ['#e8e8e8', '#2b6fd6'], ['#202020', '#ff5ea8'], ['#f2f2f0', '#ff8c1a'], ['#123c69', '#f2f2f2']],
  boda: ['#c62828', '#202325', '#c62828', '#1e4f9c', '#202325', '#c62828', '#3b3b3b', '#7a1f1f'],
  bus: ['#c79a1b', '#2b6b4e', '#c79a1b', '#8a1c1c', '#2b6b4e', '#c79a1b', '#25507a', '#c79a1b'],
  truck: ['#86694a', '#5a6b78', '#86694a', '#6b6f4a', '#5a6b78', '#86694a', '#9a4b2c', '#5a6b78'],
  tuktuk: ['#e2c11b', '#2f8a4a', '#e2c11b', '#1f63a8', '#e2c11b', '#e2c11b', '#c43b3b', '#e2c11b'],
};
const CAMERA_RANGE = 50; // metres of each approach the queue zone covers

let L = null;            // layout + session info from /api/layout
let cur = null, prev = null, curAt = 0, frameGap = 50, lastFull = null;
let ws = null, connected = false;

/* ---------------- tooltip ---------------- */
const tip = $('#tip');
document.addEventListener('pointermove', e => {
  const t = e.target.closest && e.target.closest('[data-tip]');
  if (!t || !t.dataset.tip) { tip.hidden = true; return; }
  tip.innerHTML = t.dataset.tip; tip.hidden = false;
  const r = tip.getBoundingClientRect();
  let x = e.clientX + 14, y = e.clientY + 14;
  if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 14;
  if (y + r.height > innerHeight - 8) y = e.clientY - r.height - 14;
  tip.style.left = Math.max(8, x) + 'px'; tip.style.top = Math.max(8, y) + 'px';
});
const row = (k, v) => `<div class="r"><span>${k}</span><span>${v}</span></div>`;

/* ---------------- connection ---------------- */
function send(msg) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(msg)); }
function setConn(state) {
  connected = state === 'live';
  const c = $('#conn');
  c.innerHTML = state === 'live' ? '<i class="dot"></i><span>Live</span>' : '<i class="dot ' + (state === 'down' ? 'bad' : 'warn') + '"></i><span>' + (state === 'down' ? 'Disconnected, retrying…' : 'Connecting…') + '</span>';
  $('#offline').hidden = state !== 'down';
}
function connect() {
  setConn('connecting');
  ws = new WebSocket((location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws');
  ws.onopen = () => setConn('live');
  ws.onmessage = ev => onFrame(JSON.parse(ev.data));
  ws.onclose = () => { setConn('down'); setTimeout(connect, 1500); };
}
function onFrame(f) {
  const now = performance.now();
  if (cur) frameGap = Math.min(150, Math.max(20, 0.8 * frameGap + 0.2 * (now - curAt)));
  prev = cur; cur = f; curAt = now;
  cur.byId = new Map(f.vehicles.map(v => [v[0], v]));
  if (f.full) { lastFull = f; uiSlow(f); }
}

/* ---------------- controls ---------------- */
function buildControls() {
  $('#ctrlSeg').innerHTML = L.controllers.map(c => `<button data-c="${c.key}" aria-pressed="false">${esc(c.label)}</button>`).join('');
  $('#scenChips').innerHTML = L.scenarios.map(s => `<button class="chip" data-s="${s.key}" aria-pressed="false">${esc(s.label)}</button>`).join('');
  $$('#ctrlSeg button').forEach(b => b.addEventListener('click', () => send({ cmd: 'controller', value: b.dataset.c })));
  $$('#scenChips button').forEach(b => b.addEventListener('click', () => send({ cmd: 'scenario', value: b.dataset.s })));
  $$('#speedSeg button').forEach(b => b.addEventListener('click', () => send({ cmd: 'speed', value: +b.dataset.x })));
  $('#playBtn').addEventListener('click', () => cur && send({ cmd: 'pause', value: !cur.paused }));
  $('#resetBtn').addEventListener('click', () => send({ cmd: 'reset' }));
  const sh = L.shield;
  $('#shield').innerHTML = [
    ['min', `Min green ${sh.min_green} s`], ['max', `Max green ${sh.max_green} s`],
    ['clr', `Amber ${sh.yellow} s + all-red ${sh.all_red} s`], ['wait', `Max wait ${sh.max_wait} s`],
  ].map(([k, t]) => `<span class="sh" data-k="${k}">${t}</span>`).join('');
  $('#queues').innerHTML = APPS.map(a => `<div class="qrow" data-a="${a}"><span class="nm"><i class="sw" style="background:${SCOL[a]}"></i>${SHORT[a]}</span><span class="qbar"><i style="background:${SCOL[a]};width:0"></i></span><span class="v">0</span><span class="w">0 s</span></div>`).join('');
  $('#forecast').innerHTML = APPS.map(a => `<div class="frow" data-a="${a}"><span class="nm"><i class="sw" style="background:${SCOL[a]}"></i>${SHORT[a]}</span><svg viewBox="0 0 120 30" preserveAspectRatio="none" aria-hidden="true"></svg><span class="v">–</span><span class="v">–</span><span class="v">–</span></div>`).join('');
}

/* ---------------- canvas ---------------- */
const cv = $('#cam'), ctx = cv.getContext('2d');
let W = 800, H = 519, K = 1, DPR = 1, bg = null;
const px = x => W / 2 + x * K, py = y => H / 2 + y * K;

function sizeCanvas() {
  if (!L) return;
  H = Math.round(W * L.view[1] / L.view[0]); K = W / L.view[0];
  const r = cv.getBoundingClientRect();
  DPR = Math.min(3, (window.devicePixelRatio || 1) * Math.max(1, r.width / W));
  cv.width = Math.round(W * DPR); cv.height = Math.round(H * DPR);
  bg = drawStatic();
}
function rng(seed) { let s = seed >>> 0; return () => { s = (s * 1664525 + 1013904223) >>> 0; return s / 4294967296; }; }
function laneOffset(a) { // unit vector from the centre line to the inbound lane of arm a
  const e = L.inbound[a][1], o = L.out[a];
  const lx = e[0] - o[0] * L.half, ly = e[1] - o[1] * L.half, n = Math.hypot(lx, ly);
  return [lx / n, ly / n];
}
function drawStatic() {
  const c = document.createElement('canvas'); c.width = cv.width; c.height = cv.height;
  const g = c.getContext('2d'); g.setTransform(DPR, 0, 0, DPR, 0, 0);
  const r = rng(42), hw = L.half * K, sw = 3.3 * K, cx = W / 2, cy = H / 2;
  g.fillStyle = '#2a2f2c'; g.fillRect(0, 0, W, H);
  const blocks = [[0, 0, cx - hw - sw, cy - hw - sw], [cx + hw + sw, 0, W, cy - hw - sw], [0, cy + hw + sw, cx - hw - sw, H], [cx + hw + sw, cy + hw + sw, W, H]];
  for (const [x0, y0, x1, y1] of blocks) {
    g.fillStyle = '#343a37'; g.fillRect(x0, y0, x1 - x0, y1 - y0);
    for (let i = 0; i < 9; i++) {
      const bw = 40 + r() * 90, bh = 34 + r() * 70;
      const bx = x0 + 8 + r() * Math.max(1, x1 - x0 - bw - 16), by = y0 + 8 + r() * Math.max(1, y1 - y0 - bh - 16);
      g.fillStyle = 'rgba(0,0,0,0.35)'; g.fillRect(bx + 5, by + 5, bw, bh);
      g.fillStyle = ['#5b6460', '#6c6a62', '#4e5a63', '#7a6f60', '#5f5752', '#6f7a76'][(r() * 6) | 0]; g.fillRect(bx, by, bw, bh);
      g.strokeStyle = 'rgba(255,255,255,0.07)'; g.lineWidth = 1; g.strokeRect(bx + 3.5, by + 3.5, bw - 7, bh - 7);
    }
    for (let i = 0; i < 7; i++) {
      const tx = x0 + 10 + r() * (x1 - x0 - 20), ty = y0 + 10 + r() * (y1 - y0 - 20), tr = 6 + r() * 9;
      g.fillStyle = 'rgba(0,0,0,0.3)'; g.beginPath(); g.arc(tx + 3, ty + 3, tr, 0, 7); g.fill();
      g.fillStyle = r() < 0.5 ? '#3f6b3a' : '#4d7a41'; g.beginPath(); g.arc(tx, ty, tr, 0, 7); g.fill();
    }
  }
  g.fillStyle = '#6d716c'; g.fillRect(cx - hw - sw, 0, 2 * (hw + sw), H); g.fillRect(0, cy - hw - sw, W, 2 * (hw + sw));
  g.fillStyle = '#5c605b'; g.fillRect(cx - hw - 2, 0, 2 * hw + 4, H); g.fillRect(0, cy - hw - 2, W, 2 * hw + 4);
  g.fillStyle = '#3b3f42'; g.fillRect(cx - hw, 0, 2 * hw, H); g.fillRect(0, cy - hw, W, 2 * hw);
  for (let i = 0; i < 900; i++) { g.fillStyle = r() < 0.5 ? 'rgba(255,255,255,0.025)' : 'rgba(0,0,0,0.05)'; g.fillRect(r() * W, r() * H, 2, 2); }
  // centre lines: dashed away from the junction, solid on the approach
  const stopPx = L.stop * K, solid = (L.stop + 18) * K;
  g.strokeStyle = 'rgba(240,240,235,0.75)'; g.lineWidth = 1.5;
  g.setLineDash([10, 9]); g.beginPath();
  g.moveTo(cx, 0); g.lineTo(cx, cy - solid); g.moveTo(cx, cy + solid); g.lineTo(cx, H);
  g.moveTo(0, cy); g.lineTo(cx - solid, cy); g.moveTo(cx + solid, cy); g.lineTo(W, cy); g.stroke();
  g.setLineDash([]); g.beginPath();
  g.moveTo(cx, cy - stopPx); g.lineTo(cx, cy - solid); g.moveTo(cx, cy + stopPx); g.lineTo(cx, cy + solid);
  g.moveTo(cx - stopPx, cy); g.lineTo(cx - solid, cy); g.moveTo(cx + stopPx, cy); g.lineTo(cx + solid, cy); g.stroke();
  // zebra crossings and stop lines
  g.fillStyle = 'rgba(240,240,235,0.8)';
  const [c0, c1] = L.cross;
  for (const a of APPS) {
    const o = L.out[a], side = [o[1], -o[0]];
    for (let k = -L.half + 0.5; k < L.half - 0.4; k += 1.6) {
      const p0 = [o[0] * c0 + side[0] * k, o[1] * c0 + side[1] * k], p1 = [o[0] * c1 + side[0] * (k + 0.8), o[1] * c1 + side[1] * (k + 0.8)];
      g.fillRect(Math.min(px(p0[0]), px(p1[0])), Math.min(py(p0[1]), py(p1[1])), Math.abs(px(p1[0]) - px(p0[0])) || 1, Math.abs(py(p1[1]) - py(p0[1])) || 1);
    }
    const off = laneOffset(a), s0 = [o[0] * L.stop, o[1] * L.stop], s1 = [s0[0] + off[0] * L.half, s0[1] + off[1] * L.half];
    g.strokeStyle = '#f2f2ee'; g.lineWidth = 2.5; g.beginPath(); g.moveTo(px(s0[0]), py(s0[1])); g.lineTo(px(s1[0]), py(s1[1])); g.stroke();
    // turn arrows painted on the approach
    const ax = o[0] * (L.stop + 7) + off[0] * L.lane, ay = o[1] * (L.stop + 7) + off[1] * L.lane;
    g.save(); g.translate(px(ax), py(ay)); g.rotate(Math.atan2(-o[1], -o[0]));
    g.strokeStyle = 'rgba(240,240,235,0.55)'; g.lineWidth = 1.4; g.beginPath();
    g.moveTo(-10, 0); g.lineTo(8, 0); g.moveTo(4, -3); g.lineTo(8, 0); g.lineTo(4, 3); g.stroke(); g.restore();
  }
  const vg = g.createRadialGradient(cx, cy, 120, cx, cy, 540); vg.addColorStop(0, 'rgba(0,0,0,0)'); vg.addColorStop(1, 'rgba(0,0,0,0.35)');
  g.fillStyle = vg; g.fillRect(0, 0, W, H);
  return c;
}

function lerpAngle(a, b, t) { let d = b - a; while (d > Math.PI) d -= 2 * Math.PI; while (d < -Math.PI) d += 2 * Math.PI; return a + d * t; }
function drawVehicle(x, y, h, ti, paint, stopped) {
  const T = L.types[ti], len = T.length, wid = T.width;
  ctx.save(); ctx.translate(px(x), py(y)); ctx.rotate(h); ctx.scale(K, K);
  ctx.fillStyle = 'rgba(0,0,0,0.35)'; ctx.fillRect(-len + 0.4, -wid / 2 + 0.4, len, wid);
  const p = PAINT[T.name][paint % 8];
  if (T.name === 'boda') {
    ctx.fillStyle = p; ctx.fillRect(-len, -wid / 2, len, wid);
    ctx.fillStyle = '#f3d23a'; ctx.beginPath(); ctx.arc(-len * 0.55, 0, 0.42, 0, 7); ctx.fill();
  } else {
    ctx.fillStyle = T.name === 'matatu' ? p[0] : p; ctx.fillRect(-len, -wid / 2, len, wid);
    if (T.name === 'matatu') { ctx.fillStyle = p[1]; ctx.fillRect(-len, -wid * 0.13, len, wid * 0.26); }
    if (T.name === 'truck') { ctx.fillStyle = 'rgba(255,255,255,0.13)'; ctx.fillRect(-len, -wid / 2 + 0.2, len - 2.2, wid - 0.4); }
    if (T.name === 'bus') { ctx.fillStyle = 'rgba(255,255,255,0.12)'; for (let k = -len + 1; k < -2; k += 1.6) ctx.fillRect(k, -wid / 2 + 0.25, 1, wid - 0.5); }
    ctx.fillStyle = 'rgba(18,28,38,0.75)'; ctx.fillRect(-1.5, -wid / 2 + 0.25, 0.75, wid - 0.5);
    if (stopped) { ctx.fillStyle = '#ff3b3b'; ctx.fillRect(-len - 0.15, -wid / 2, 0.3, 0.5); ctx.fillRect(-len - 0.15, wid / 2 - 0.5, 0.3, 0.5); }
  }
  ctx.restore();
}
function bbox(x, y, h, ti) { // axis-aligned box around the vehicle, in pixels
  const T = L.types[ti], c = Math.cos(h), s = Math.sin(h);
  const pts = [[0, -T.width / 2], [0, T.width / 2], [-T.length, -T.width / 2], [-T.length, T.width / 2]].map(([u, v]) => [px(x + u * c - v * s), py(y + u * s + v * c)]);
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}
function drawLamp(a, st) {
  const o = L.out[a], off = laneOffset(a);
  const x = px(o[0] * (L.stop + 1.5) + off[0] * (L.half + 1.6)), y = py(o[1] * (L.stop + 1.5) + off[1] * (L.half + 1.6));
  const vertical = o[0] === 0;
  ctx.fillStyle = '#121416'; if (vertical) ctx.fillRect(x - 3.5, y - 10, 7, 20); else ctx.fillRect(x - 10, y - 3.5, 20, 7);
  [['R', '#ff4b4b'], ['A', '#ffb21e'], ['G', '#36d17a']].forEach(([k, col], i) => {
    ctx.fillStyle = st === k ? col : '#2c3134'; const d = (i - 1) * 6.2;
    ctx.beginPath(); vertical ? ctx.arc(x, y + d, 2.4, 0, 7) : ctx.arc(x + d, y, 2.4, 0, 7); ctx.fill();
  });
}
function alongInbound(a, s, lateral = 0) {
  const [p0, p1] = L.inbound[a], n = Math.hypot(p1[0] - p0[0], p1[1] - p0[1]);
  const d = [(p1[0] - p0[0]) / n, (p1[1] - p0[1]) / n], off = laneOffset(a);
  return [p0[0] + d[0] * s + off[0] * lateral, p0[1] + d[1] * s + off[1] * lateral, Math.atan2(d[1], d[0])];
}

function render(now) {
  if (!bg) return;
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx.drawImage(bg, 0, 0, W, H);
  if (!cur) return;
  const fail = !cur.camera_ok, boxes = $('#optBoxes').checked, zones = $('#optZones').checked;
  if (zones && !fail) {
    for (const a of APPS) {
      const o = L.out[a], off = laneOffset(a), lat = L.lane;
      const near = [o[0] * L.stop + off[0] * lat, o[1] * L.stop + off[1] * lat], far = [o[0] * (L.stop + CAMERA_RANGE) + off[0] * lat, o[1] * (L.stop + CAMERA_RANGE) + off[1] * lat];
      const hw = 1.7;
      const x0 = px(Math.min(near[0], far[0]) - (o[0] === 0 ? hw : 0)), x1 = px(Math.max(near[0], far[0]) + (o[0] === 0 ? hw : 0));
      const y0 = py(Math.min(near[1], far[1]) - (o[1] === 0 ? hw : 0)), y1 = py(Math.max(near[1], far[1]) + (o[1] === 0 ? hw : 0));
      ctx.fillStyle = VCOL[a] + '20'; ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
      ctx.strokeStyle = VCOL[a] + 'aa'; ctx.lineWidth = 1; ctx.setLineDash([5, 4]); ctx.strokeRect(x0 + 0.5, y0 + 0.5, x1 - x0 - 1, y1 - y0 - 1); ctx.setLineDash([]);
      ctx.font = '600 10px "JetBrains Mono", monospace'; ctx.fillStyle = VCOL[a];
      const lx = o[0] === 0 ? x1 + 4 : (o[0] > 0 ? x1 - 52 : x0 + 2), ly = o[1] === 0 ? y0 - 4 : (o[1] > 0 ? y1 - 4 : y0 + 10);
      ctx.fillText(SHORT[a] + ' queue', lx, ly);
    }
  }
  for (const inc of cur.incidents) {
    const [x, y, h] = alongInbound(inc.approach, (inc.s_from + inc.s_to) / 2, 1.6);
    ctx.save(); ctx.translate(px(x), py(y)); ctx.rotate(h); ctx.scale(K, K);
    ctx.fillStyle = 'rgba(0,0,0,0.35)'; ctx.fillRect(-7.6, -0.9, 8, 2.5);
    ctx.fillStyle = '#7b6a55'; ctx.fillRect(-8, -1.3, 8, 2.5); ctx.fillStyle = '#ffb21e';
    for (const u of [-11, -13.5, 2.5]) { ctx.beginPath(); ctx.moveTo(u, -1.6); ctx.lineTo(u + 0.6, -0.2); ctx.lineTo(u - 0.6, -0.2); ctx.fill(); }
    ctx.restore();
  }
  const t = Math.min(1, (now - curAt) / frameGap);
  const drawn = [];
  for (const v of cur.vehicles) {
    let [id, x, y, h, ti, paint, stopped] = v;
    const p = prev && prev.byId && prev.byId.get(id);
    if (p && Math.hypot(p[1] - x, p[2] - y) < 30) { x = p[1] + (x - p[1]) * t; y = p[2] + (y - p[2]) * t; h = lerpAngle(p[3], h, t); }
    if (Math.abs(x) > L.view[0] / 2 + 12 || Math.abs(y) > L.view[1] / 2 + 12) continue;
    drawVehicle(x, y, h, ti, paint, stopped);
    drawn.push([id, x, y, h, ti, stopped]);
  }
  for (const [x, y, crossing] of cur.peds) { ctx.fillStyle = crossing ? '#f1e2c4' : '#d9c7a5'; ctx.beginPath(); ctx.arc(px(x), py(y), 2.4, 0, 7); ctx.fill(); }
  for (const a of APPS) drawLamp(a, cur.signal.heads[a]);
  if (fail) {
    ctx.fillStyle = 'rgba(30,30,30,0.72)'; ctx.fillRect(0, 0, W, H);
    for (let i = 0; i < 700; i++) { const g = (Math.random() * 120) | 0; ctx.fillStyle = `rgba(${g},${g},${g},0.5)`; ctx.fillRect(Math.random() * W, Math.random() * H, 3, 2); }
    ctx.fillStyle = 'rgba(255,255,255,0.05)'; ctx.fillRect(0, (now / 8) % H, W, 14);
    return;
  }
  if (boxes) {
    ctx.font = '500 8.5px "JetBrains Mono", monospace'; ctx.lineWidth = 1.2;
    for (const [id, x, y, h, ti, stopped] of drawn) {
      const [x0, y0, x1, y1] = bbox(x, y, h, ti), cls = L.types[ti].cls, col = BOX[cls];
      ctx.strokeStyle = col; ctx.strokeRect(x0 - 2, y0 - 2, x1 - x0 + 4, y1 - y0 + 4);
      if (!stopped || id % 3 === 0) {
        const label = `${cls} ${id}`, tw = ctx.measureText(label).width + 5;
        ctx.fillStyle = col; ctx.fillRect(x0 - 2, y0 - 13, tw, 10.5); ctx.fillStyle = '#111'; ctx.fillText(label, x0 + 0.5, y0 - 5);
      }
    }
    ctx.strokeStyle = BOX.person; ctx.lineWidth = 1;
    for (const [x, y] of cur.peds) ctx.strokeRect(px(x) - 4, py(y) - 4, 8, 8);
  }
}

/* ---------------- panels ---------------- */
const STATE_WORD = { G: 'green', A: 'amber', R: 'red' };
function lampSet(el, st) { $$('i', el).forEach(i => i.classList.toggle('on', i.classList.contains(st.toLowerCase()))); }
function uiFast() {
  if (!cur || !L) return;
  const f = cur, sig = f.signal;
  $('#clock').textContent = f.clock; $('#hudTime').textContent = f.clock;
  const sNS = sig.heads.north, sEW = sig.heads.east;
  lampSet($('#lampNS'), sNS); lampSet($('#lampEW'), sEW);
  const txt = st => st === 'R' ? (sig.state === 'all_red' ? 'all-red' : 'red') : st === 'G' ? `green ${Math.floor(sig.green_time)} s` : 'amber';
  $('#stNS').textContent = txt(sNS); $('#stEW').textContent = txt(sEW);
  const ctl = L.controllers.find(c => c.key === f.controller);
  $('#ctrlName').textContent = ctl.adaptive ? `${ctl.label} · decides every ${L.decision_interval} s` : `${ctl.label} · fixed timings`;
  const green = sig.state === 'green', d = f.decision;
  $('#decLabel').textContent = f.controller === 'rl' ? 'Agent output · p(switch)' : ctl.adaptive ? 'Controller · switch score' : 'Fixed plan · progress';
  $('#decBig').textContent = !green ? 'CLEARING' : d.reason === 'min_green' && d.wanted !== sig.phase ? 'HELD' : 'KEEP';
  $('#decBar').style.width = (green ? d.p * 100 : 100).toFixed(0) + '%';
  $('#decWhy').textContent = green ? d.why + (ctl.adaptive ? ` · next decision in ${Math.ceil(sig.next_decision_in)} s` : '') : 'Amber and all-red clearance';
  $$('#shield .sh').forEach(s => {
    const k = s.dataset.k;
    const on = (k === 'clr' && !green) || (k === 'min' && green && sig.green_time < L.shield.min_green) ||
      (k === 'max' && d.reason === 'max_green' && !green) || (k === 'wait' && d.reason === 'max_wait' && !green);
    s.classList.toggle('active', on);
  });
  $$('#ctrlSeg button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.c === f.controller)));
  $$('#scenChips button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.s === f.scenario)));
  $$('#speedSeg button').forEach(b => b.setAttribute('aria-pressed', String(+b.dataset.x === f.speed)));
  $('#playBtn').textContent = f.paused ? 'Play' : 'Pause';
  $$('#queues .qrow').forEach(r => {
    const a = r.dataset.a, [n, m, w, beyond] = f.queues[a];
    r.querySelector('.qbar i').style.width = Math.min(100, n / 40 * 100) + '%';
    r.querySelector('.v').innerHTML = `${n} veh` + (n > 28 ? '<span class="flag">spillback</span>' : '');
    r.querySelector('.w').textContent = Math.round(w) + ' s';
    r.dataset.tip = `<div class="tt">${NAME[a]} approach</div>` + row('Queued', n + ' veh') + row('Queue length', m.toFixed(0) + ' m') + row('Beyond camera view', beyond + ' veh') + row('Longest wait', Math.round(w) + ' s') + row('Signal', STATE_WORD[sig.heads[a]]);
  });
  const counts = {}; let tot = 0;
  for (const v of f.vehicles) if (Math.abs(v[1]) < L.view[0] / 2 && Math.abs(v[2]) < L.view[1] / 2) { const c = L.types[v[4]].cls; counts[c] = (counts[c] || 0) + 1; tot++; }
  const peds = f.peds.length;
  $('#hudCounts').textContent = f.camera_ok ? `Tracked ${tot + peds} · car ${counts.car || 0} · matatu ${counts.matatu || 0} · boda ${counts.boda_boda || 0} · bus ${counts.bus || 0} · truck ${counts.truck || 0} · tuk-tuk ${counts.tuk_tuk || 0} · person ${peds}` : 'Tracked: — · detector offline';
  $('#pipeSee').textContent = f.camera_ok ? `YOLOv8s fine-tuned · ByteTrack · ${tot + peds} tracked` : 'Camera offline · using forecast and profile';
  $('#dotSee').className = 'dot' + (f.camera_ok ? '' : ' bad');
  $('#nosignal').hidden = f.camera_ok;
  $('#pipeControl').textContent = `${ctl.label} behind the safety shield`;
}
function spark(vals, x0, dx, ymax) { return vals.map((v, i) => `${i ? 'L' : 'M'}${(x0 + i * dx).toFixed(1)},${(28 - v / ymax * 25).toFixed(1)}`).join(''); }
function uiSlow(f) {
  $('#fcSub').textContent = f.camera_ok ? 'vehicles in the next…' : 'profile only · camera down';
  $$('#forecast .frow').forEach(r => {
    const a = r.dataset.a, F = f.forecast[a], ymax = Math.max(6, ...F.past, ...F.next) * 1.15, dx = 59 / 14;
    const hi = F.next.map((v, k) => v * (1.08 + k * 0.012)), lo = F.next.map((v, k) => v * (0.92 - k * 0.012));
    const band = hi.map((v, k) => `${k ? 'L' : 'M'}${(61 + k * dx).toFixed(1)},${(28 - v / ymax * 25).toFixed(1)}`).join('') + lo.map((v, k) => [k, v]).reverse().map(([k, v]) => `L${(61 + k * dx).toFixed(1)},${(28 - v / ymax * 25).toFixed(1)}`).join('') + 'Z';
    r.querySelector('svg').innerHTML = `<line x1="60" x2="60" y1="1" y2="29" stroke="var(--line)" vector-effect="non-scaling-stroke"/>`
      + `<path d="${band}" fill="${SCOL[a]}" opacity="0.16"/>`
      + `<path d="${spark(F.past, 0, 59 / Math.max(1, F.past.length - 1), ymax)}" fill="none" stroke="${SCOL[a]}" stroke-width="1.6" vector-effect="non-scaling-stroke"/>`
      + `<path d="${spark(F.next, 61, dx, ymax)}" fill="none" stroke="${SCOL[a]}" stroke-width="1.6" stroke-dasharray="3 2" vector-effect="non-scaling-stroke"/>`;
    const vs = r.querySelectorAll('.v'); vs[0].textContent = F.n1.toFixed(0); vs[1].textContent = F.n5.toFixed(0); vs[2].textContent = F.n15.toFixed(0);
    r.dataset.tip = `<div class="tt">${NAME[a]} arrivals</div>` + row('Last minute, counted', (F.past[F.past.length - 1] ?? 0) + ' veh') + row('Next 1 min', F.n1.toFixed(1)) + row('Next 5 min', F.n5.toFixed(1)) + row('Next 15 min', F.n15.toFixed(1));
  });
  const k = f.kpi;
  $('#kDelay').innerHTML = (k.delay ?? '—') + '<small>s/veh</small>';
  $('#kThru').innerHTML = (k.throughput != null ? k.throughput.toLocaleString() : '—') + '<small>veh/h</small>';
  $('#kPed').innerHTML = (k.ped_wait != null ? Math.round(k.ped_wait) : '—') + '<small>s</small>';
  $('#kServed').textContent = k.served.toLocaleString();
  const label = key => (L.controllers.find(c => c.key === key) || {}).label || key;
  const scen = key => (L.scenarios.find(s => s.key === key) || {}).label || key;
  $('#runSub').textContent = `${label(f.controller)} · ${scen(f.scenario).toLowerCase()} · since ${k.since.slice(0, 5)}`;
  $('#scoreBody').innerHTML = f.runs.map(r => `<tr class="${r.current ? 'cur' : ''}"><td>${r.current ? '▸ ' : ''}${esc(label(r.controller))} · ${esc(scen(r.scenario).toLowerCase())}</td><td class="num">${r.minutes.toFixed(0)}</td><td class="num">${r.delay != null ? r.delay.toFixed(1) + ' s' : '—'}</td></tr>`).join('');
  $('#scoreNote').hidden = f.runs.length > 1;
  $('#log').innerHTML = f.log.map(([t, kind, m]) => `<li class="${kind}"><span class="t">${t}</span><span class="m">${esc(m)}</span></li>`).join('');
}

/* ---------------- start ---------------- */
let fastAt = 0;
function loop(now) {
  render(now);
  if (now - fastAt > 200) { uiFast(); fastAt = now; }
  requestAnimationFrame(loop);
}
async function start() {
  for (;;) {
    try { L = await (await fetch('/api/layout')).json(); break; }
    catch (e) { setConn('down'); await new Promise(r => setTimeout(r, 1500)); }
  }
  buildControls(); sizeCanvas(); connect();
  requestAnimationFrame(loop);
  let rz; addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(sizeCanvas, 120); });
}
start();
})();
