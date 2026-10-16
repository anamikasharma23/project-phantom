/* PROJECT PHANTOM mission console.
 *
 * Two ways to run:
 *   1. Open ui/index.html directly        -> REPLAY of the saved trace in ui/data/mission.js
 *   2. python -m phantom serve            -> LIVE: Start / Pause / Step / Reset and the
 *                                            configuration screen drive the real simulation
 *
 * In both cases every number on screen is read from trace events produced by the
 * Python simulation. For step i:
 *   E[i].position / battery / time_left -> agent marker and resource panels
 *   E[i].belief                         -> heatmap, belief summary, Signal DNA (belief AFTER step i)
 *   E[i].decision                       -> the scored options that produced step i
 *   observations of steps 1..i          -> route, explored area, scans, clues, bearings, log
 * Ground truth (trace.evaluation) only exists after a mission ends and is only drawn
 * in the clearly labelled evaluation view switched on from the Report screen.
 */
(function () {
  "use strict";

  // Older browsers (Safari < 16, Firefox < 112) lack roundRect; without it the map would not draw.
  if (window.CanvasRenderingContext2D && !CanvasRenderingContext2D.prototype.roundRect) {
    CanvasRenderingContext2D.prototype.roundRect = function (x, y, w, h, r) {
      r = Math.min(typeof r === "number" ? r : 0, w / 2, h / 2);
      this.moveTo(x + r, y); this.arcTo(x + w, y, x + w, y + h, r); this.arcTo(x + w, y + h, x, y + h, r);
      this.arcTo(x, y + h, x, y, r); this.arcTo(x, y, x + w, y, r); this.closePath();
    };
  }

  // ================================================================ state
  let M = null, E = [], W = 0, H = 0, cur = 0;
  let D = null;                                  // derived data
  let view = "mission";
  let playing = false, playTimer = null;         // replay playback
  const live = { on: false, info: null, session: null, req: null, running: false, timer: null, busy: false, done: false, savedTo: null };
  const layers = { terrain: true, heat: true, route: true, signals: true, clues: true, explored: true, grid: true };
  const cam = { zoom: 1, panX: 0, panY: 0 };
  let selected = null, selectedSource = null, showTruth = false;
  let eventFilter = new Set(["clue", "detect", "safety", "label", "fail", "local", "end"]);
  let traceFilter = "all", openRows = new Set();
  let pulseStart = 0, dirty = true;
  const speed = { replay: 5, live: 8 };   // remembered across transport re-renders
  const cine = { follow: false, anim: null, band: null, scrub: false, rec: null };   // cinematic replay state
  const isCine = () => view === "replay";
  let terrainImg = null, heatImg = null, heatFor = -1;
  const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const $ = (id) => document.getElementById(id);
  const canvas = $("map"), ctx = canvas.getContext("2d");
  const C = { cyan: "#38e1f0", cyanDeep: "#1aa9c4", amber: "#ffb547", red: "#ff5a5f", green: "#56e39f", text: "#e3eef7", dim: "#93a8bd", faint: "#5f768c" };
  const DASH = [[], [10, 6], [2, 5], [12, 4, 2, 4]];
  const DIRS = { N: [0, -1], E: [1, 0], S: [0, 1], W: [-1, 0] };
  const WORD = { N: "north", E: "east", S: "south", W: "west" };

  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const pct = (p, d = 0) => (100 * p).toFixed(d) + "%";
  const cell = (c) => (c ? `(${c[0]}, ${c[1]})` : "–");
  const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

  // ================================================================ heat colour (log scale)
  // 0.03% -> 0, 0.3% -> 1/3, 3% -> 2/3, 30%+ -> 1
  const RAMP = [[0, [10, 30, 90], 0], [0.15, [24, 64, 176], 0.34], [0.4, [22, 172, 212], 0.55], [0.6, [110, 214, 140], 0.62],
                [0.76, [250, 210, 72], 0.76], [0.9, [255, 142, 42], 0.86], [1, [255, 74, 42], 0.94]];
  const heatT = (p) => clamp((Math.log10(Math.max(p, 1e-9)) + 3.5) / 3, 0, 1);
  function heatRGBA(t) {
    for (let k = 1; k < RAMP.length; k++) if (t <= RAMP[k][0]) {
      const [t0, c0, a0] = RAMP[k - 1], [t1, c1, a1] = RAMP[k], f = (t - t0) / (t1 - t0);
      return [c0.map((v, j) => Math.round(v + (c1[j] - v) * f)), a0 + (a1 - a0) * f];
    }
    const L = RAMP[RAMP.length - 1]; return [L[1], L[2]];
  }
  function rampCSS(dir) {
    const s = [];
    for (let i = 0; i <= 10; i++) { const [c] = heatRGBA(i / 10); s.push(`rgb(${c})` + ` ${i * 10}%`); }
    return `linear-gradient(${dir}, ${s.join(",")})`;
  }

  // ================================================================ derived data
  function dijkstraFrom(start) {
    const cost = new Float64Array(W * H).fill(Infinity), T = M.public.terrain, k = M.public.costs;
    const open = [[0, start[0], start[1]]]; cost[start[1] * W + start[0]] = 0;
    while (open.length) {
      let bi = 0; for (let i = 1; i < open.length; i++) if (open[i][0] < open[bi][0]) bi = i;
      const [d, x, y] = open.splice(bi, 1)[0];
      if (d > cost[y * W + x]) continue;
      for (const [dx, dy] of Object.values(DIRS)) {
        const nx = x + dx, ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= W || ny >= H || T[ny][nx] === 2) continue;
        const nd = d + (T[ny][nx] === 1 ? k.move_rough : k.move_plain);
        if (nd < cost[ny * W + nx] - 1e-9) { cost[ny * W + nx] = nd; open.push([nd, nx, ny]); }
      }
    }
    return cost;
  }
  function massWithin(loc, c, r) {
    let s = 0;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (Math.hypot(x - c[0], y - c[1]) <= r + 1e-9) s += loc[y * W + x];
    return s;
  }

  function derive() {
    const exploredAt = new Int32Array(W * H).fill(-1);
    const scans = [], readings = [], clues = [], counts = [], keyEvents = [], conf = [], heading = [];
    let c = { moves: 0, local: 0, long: 0, clues: 0, recharges: 0 }, h = "N";
    const mark = (x, y, s) => { const k = y * W + x; if (exploredAt[k] < 0) exploredAt[k] = s; };
    const r = M.public.local_scan.radius;
    mark(E[0].position[0], E[0].position[1], 0);
    let prevLabels = {}, returning = false;
    for (let i = 0; i < E.length; i++) {
      const ev = E[i];
      const top = ev.belief.confirmed || ev.belief.top_cells[0].cell;
      conf.push(massWithin(ev.belief.location, top, r));
      if (i > 0) {
        const o = ev.observation, a = o.action.type, d = ev.decision;
        mark(o.position[0], o.position[1], i);
        if (o.ok && a === "move") { c.moves++; h = o.action.direction; }
        if (o.ok && a === "long_scan") { c.long++; keyEvents.push({ step: i, type: "long", text: "Long-range scan" }); }
        if (o.ok && a === "local_scan") { c.local++; keyEvents.push({ step: i, type: "local", text: `Local scan at ${cell(o.position)}` }); }
        if (o.ok && a === "recharge") { c.recharges++; keyEvents.push({ step: i, type: "safety", text: `Recharged to ${o.battery.toFixed(0)}` }); }
        if (o.ok && a === "move") keyEvents.push({ step: i, type: "move", text: `Moved ${WORD[o.action.direction]} to ${cell(o.position)}` });
        for (const rd of o.readings) readings.push({ ...rd, step: i });
        if (o.scan) {
          scans.push({ step: i, center: o.scan.center, radius: o.scan.radius });
          for (const [x, y] of o.scan.cells) mark(x, y, i);
          if (o.scan.eve_detected) keyEvents.push({ step: i, type: "detect", text: `EVE detected at ${cell(o.scan.eve_position)}` });
        }
        for (const cl of o.clues) if (cl.first_time) {
          clues.push({ ...cl, step: i }); c.clues++;
          keyEvents.push({ step: i, type: "clue", text: cl.type === "track" ? `Track clue at ${cell(cl.position)}` : `Residue clue at ${cell(cl.position)}` });
        }
        if (!o.ok) keyEvents.push({ step: i, type: "fail", text: `Action failed: ${o.error}` });
        const isReturn = d && d.chosen_id === "return_base";
        if (isReturn && !returning) keyEvents.push({ step: i, type: "safety", text: `Turned back: ${d.forced_by || "returning to base"}` });
        returning = isReturn;
        if (o.mission_over) keyEvents.push({ step: i, type: "end", text: `Mission ended (${o.end_reason.replace("_", " ")})` });
      }
      for (const [s, v] of Object.entries(ev.belief.dna || {})) {
        if (prevLabels[s] && prevLabels[s] !== v.label && v.label !== "uncertain")
          keyEvents.push({ step: i, type: "label", text: `Signal ${s} now rated ${v.label}` });
        prevLabels[s] = v.label;
      }
      counts.push({ ...c });
      heading.push(h);
    }
    D = { exploredAt, scans, readings, clues, counts, keyEvents, conf, heading,
          sources: Object.keys(M.public.sources), homeCost: dijkstraFrom(M.public.base) };
  }

  // ================================================================ loading
  function setTrace(trace, keepStep) {
    if (!trace || !trace.events || !trace.public) { toast("That file is not a PROJECT PHANTOM mission trace (mission.json)."); return; }
    const sameMap = M && M.public.mission_name === trace.public.mission_name && W === trace.public.width;
    M = trace; E = M.events; W = M.public.width; H = M.public.height;
    derive();
    if (!sameMap || !terrainImg) { terrainImg = buildTerrain(); cam.zoom = 1; cam.panX = 0; cam.panY = 0; }
    heatFor = -1; maskFor = -1;
    if (!selectedSource || !D.sources.includes(selectedSource)) selectedSource = D.sources[0];
    if (!keepStep) { cur = 0; selected = null; showTruth = false; }
    $("scrubber").max = E.length - 1;
    $("stepMax").textContent = live.on ? M.public.time_budget : E.length - 1;
    $("missionId").textContent = `PH-${String(M.meta.seed).padStart(4, "0")}`;
    $("missionTime").textContent = M.public.start_time.replace("T", " ").replace("Z", " UTC");
    $("footRun").textContent = `${M.public.mission_name} // agent ${M.meta.agent} // ${live.on ? "live simulation" : "saved run replay"}`;
    buildJump(); buildLegend(); renderTransport();
    setStep(keepStep ? cur : 0);
  }

  // ================================================================ terrain texture (built once per map)
  function hash(x, y) { let h = (x * 374761393 + y * 668265263) ^ 0x5bf03635; h = (h ^ (h >>> 13)) * 1274126177; return ((h ^ (h >>> 16)) >>> 0) / 4294967295; }
  function vnoise(x, y) {
    const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi;
    const s = (t) => t * t * (3 - 2 * t), a = hash(xi, yi), b = hash(xi + 1, yi), c = hash(xi, yi + 1), d = hash(xi + 1, yi + 1);
    return a + (b - a) * s(xf) + (c - a) * s(yf) + (a - b - c + d) * s(xf) * s(yf);
  }
  function buildTerrain() {
    const P = 20, cv = document.createElement("canvas"); cv.width = W * P; cv.height = H * P;
    const g = cv.getContext("2d"), img = g.createImageData(W * P, H * P), T = M.public.terrain;
    const terr = (x, y) => (x < 0 || y < 0 || x >= W || y >= H ? 0 : T[y][x]);
    for (let py = 0; py < H * P; py++) for (let px = 0; px < W * P; px++) {
      const x = Math.floor(px / P), y = Math.floor(py / P), fx = px / P, fy = py / P;
      const n = vnoise(fx * 1.7, fy * 1.7) * 0.6 + vnoise(fx * 5, fy * 5) * 0.4;
      // organic shapes: smooth the 0/1 rock and rough masks bilinearly, perturb with noise, threshold
      const sm = (code) => {
        const ax = fx - 0.5, ay = fy - 0.5, x0 = Math.floor(ax), y0 = Math.floor(ay), tx = ax - x0, ty = ay - y0;
        const f = (xx, yy) => (terr(xx, yy) === code ? 1 : 0);
        return f(x0, y0) * (1 - tx) * (1 - ty) + f(x0 + 1, y0) * tx * (1 - ty) + f(x0, y0 + 1) * (1 - tx) * ty + f(x0 + 1, y0 + 1) * tx * ty;
      };
      const jitter = (vnoise(fx * 3.1 + 17, fy * 3.1 + 5) - 0.5) * 0.36;
      let t = sm(2) + jitter > 0.42 ? 2 : sm(1) + jitter > 0.45 ? 1 : 0;
      if (T[y][x] === 2 && Math.abs(fx - x - 0.5) < 0.25 && Math.abs(fy - y - 0.5) < 0.25) t = 2;   // blocked cells always look blocked
      const ex = fx - x, ey = fy - y;
      let r, gg, b;
      if (t === 2) {
        // height field + light from the top-left gives craggy, lit rock
        const hh = (a, b) => vnoise(a * 2.2, b * 2.2) * 0.65 + vnoise(a * 6, b * 6) * 0.35;
        const shade = (hh(fx - 0.08, fy - 0.08) - hh(fx + 0.08, fy + 0.08)) * 260;
        const ridge = vnoise(fx * 8 + 5, fy * 8 + 9) > 0.78 ? 26 : 0;
        const v = 58 + 46 * n + shade + ridge;
        r = v * 0.86; gg = v * 0.93; b = v * 1.06 + 6;
        const up = sm(2) - (function () { const a = fx, b2 = fy - 0.12; const ax = a - 0.5, ay = b2 - 0.5, x0 = Math.floor(ax), y0 = Math.floor(ay), tx = ax - x0, ty = ay - y0; const f = (xx, yy) => (terr(xx, yy) === 2 ? 1 : 0); return f(x0, y0) * (1 - tx) * (1 - ty) + f(x0 + 1, y0) * tx * (1 - ty) + f(x0, y0 + 1) * (1 - tx) * ty + f(x0 + 1, y0 + 1) * tx * ty; })();
        if (up > 0.08) { r += 34 * up * 4; gg += 38 * up * 4; b += 44 * up * 4; }   // lit upper rim
      } else if (t === 1) {
        const sp = hash(px, py) > 0.93 ? 26 : 0;
        r = 22 + 16 * n + sp; gg = 36 + 18 * n + sp; b = 50 + 18 * n + sp;
      } else {
        const crater = vnoise(fx * 9 + 3, fy * 9 + 11) > 0.86 ? -8 : 0;
        r = 14 + 14 * n + crater; gg = 34 + 18 * n + crater; b = 54 + 20 * n + crater;
      }
      const k = (py * W * P + px) * 4;
      img.data[k] = clamp(r, 0, 255); img.data[k + 1] = clamp(gg, 0, 255); img.data[k + 2] = clamp(b, 0, 255); img.data[k + 3] = 255;
    }
    g.putImageData(img, 0, 0);
    return cv;
  }
  let maskImg = null, maskFor = -1;
  function exploredMask() {
    if (maskImg && maskFor === cur) return maskImg;
    const cv = maskImg || document.createElement("canvas"); cv.width = W; cv.height = H;
    const g = cv.getContext("2d"), img = g.createImageData(W, H);
    for (let k = 0; k < W * H; k++) {
      const s = D.exploredAt[k], seen = s >= 0 && s <= cur;
      const [r, gg, b, a] = seen ? [40, 140, 180, 0.2] : [2, 6, 12, 0.38];
      img.data[4 * k] = r; img.data[4 * k + 1] = gg; img.data[4 * k + 2] = b; img.data[4 * k + 3] = Math.round(a * 255);
    }
    g.putImageData(img, 0, 0); maskImg = cv; maskFor = cur; return cv;
  }
  function buildHeat(i) {
    if (heatFor === i && heatImg) return heatImg;
    const cv = heatImg || document.createElement("canvas"); cv.width = W; cv.height = H;
    const g = cv.getContext("2d"), img = g.createImageData(W, H), loc = E[i].belief.location, T = M.public.terrain;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const k = (y * W + x) * 4;
      if (T[y][x] === 2) { img.data[k + 3] = 0; continue; }
      const [c, a] = heatRGBA(heatT(loc[y * W + x]));
      img.data[k] = c[0]; img.data[k + 1] = c[1]; img.data[k + 2] = c[2]; img.data[k + 3] = Math.round(a * 255);
    }
    g.putImageData(img, 0, 0); heatImg = cv; heatFor = i; return cv;
  }

  // ================================================================ camera
  function geom() {
    const r = canvas.getBoundingClientRect();
    const gl = layers.grid ? 30 : 10, gb = layers.grid ? 24 : 10, pad = 10, top = isCine() ? 62 : 48, bot = isCine() ? 78 : 0;
    const base = Math.max(4, Math.min((r.width - gl - pad) / W, (r.height - gb - top - 4 - bot) / H));
    const cs = base * cam.zoom;
    const ox = gl + (r.width - gl - pad - W * base) / 2 + cam.panX;
    const oy = top + (r.height - gb - top - 4 - bot - H * base) / 2 + cam.panY;
    return { cs, ox, oy, w: r.width, h: r.height, gl, gb };
  }
  function zoomAt(f, px, py) {
    const g = geom(); if (px === undefined) { px = g.w / 2; py = g.h / 2; }
    const gx = (px - g.ox) / g.cs, gy = (py - g.oy) / g.cs;
    cam.zoom = clamp(cam.zoom * f, 0.6, 6);
    const g2 = geom(); cam.panX += px - (g2.ox + gx * g2.cs); cam.panY += py - (g2.oy + gy * g2.cs);
    $("zoomRead").textContent = `Zoom ${Math.round(cam.zoom * 100)}%`; redraw();
  }
  function fit() { cam.zoom = 1; cam.panX = 0; cam.panY = 0; $("zoomRead").textContent = "Zoom 100%"; redraw(); }
  function centerOn(c) { const g = geom(); cam.panX += g.w / 2 - (g.ox + (c[0] + 0.5) * g.cs); cam.panY += g.h / 2 - (g.oy + (c[1] + 0.5) * g.cs); redraw(); }

  // ================================================================ map drawing
  const redraw = () => { dirty = true; };
  function labelStyle(label) {
    if (label === "trusted") return C.cyan;
    if (label === "conflicting") return C.red;
    if (label === "doubtful") return C.amber;
    return "#9fd6e6";
  }

  const ease = (t) => t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
  function agentPos(now) {
    const a = cine.anim;
    if (!a) return E[cur].position;
    const t = clamp((now - a.start) / a.dur, 0, 1);
    if (t < 1) dirty = true;
    const k = ease(t);
    return [a.from[0] + (a.to[0] - a.from[0]) * k, a.from[1] + (a.to[1] - a.from[1]) * k];
  }

  function draw(now) {
    const dpr = window.devicePixelRatio || 1, r = canvas.getBoundingClientRect();
    if (!r.width) return;
    if (canvas.width !== Math.round(r.width * dpr) || canvas.height !== Math.round(r.height * dpr)) { canvas.width = Math.round(r.width * dpr); canvas.height = Math.round(r.height * dpr); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, r.width, r.height);
    if (isCine()) { ctx.fillStyle = "#02060b"; ctx.fillRect(0, 0, r.width, r.height); }
    if (!M) return;
    if (isCine() && cine.follow) {                    // camera glides to keep WALL-E centred
      const p0 = agentPos(now), g0 = geom();
      const tx = g0.w / 2 - (g0.ox - cam.panX + (p0[0] + 0.5) * g0.cs), ty = (g0.h - 78) / 2 + 20 - (g0.oy - cam.panY + (p0[1] + 0.5) * g0.cs);
      // never slide past the map edge: clamp so the map keeps filling the viewport
      const bx = g0.ox - cam.panX, by = g0.oy - cam.panY, mw = W * g0.cs, mh = H * g0.cs;
      const L = g0.gl, Rr = g0.w - 10, T0 = 64, B0 = g0.h - g0.gb - 82;
      const cx = mw > Rr - L ? clamp(tx, Rr - mw - bx, L - bx) : (L + Rr - mw) / 2 - bx;
      const cy = mh > B0 - T0 ? clamp(ty, B0 - mh - by, T0 - by) : (T0 + B0 - mh) / 2 - by;
      cam.panX += (cx - cam.panX) * 0.18; cam.panY += (cy - cam.panY) * 0.18;
      if (Math.abs(cx - cam.panX) + Math.abs(cy - cam.panY) > 0.5) dirty = true;
    }
    const g = geom(), cs = g.cs, ox = g.ox, oy = g.oy, ev = E[cur], T = M.public.terrain;
    const AP = agentPos(now);
    const X = (x) => ox + x * cs, Y = (y) => oy + y * cs, CX = (x) => ox + (x + 0.5) * cs, CY = (y) => oy + (y + 0.5) * cs;

    // terrain
    ctx.imageSmoothingEnabled = true;
    if (layers.terrain) ctx.drawImage(terrainImg, X(0), Y(0), W * cs, H * cs);
    else { ctx.fillStyle = "#0b1a28"; ctx.fillRect(X(0), Y(0), W * cs, H * cs); ctx.fillStyle = "#33414f";
      for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) if (T[y][x] === 2) ctx.fillRect(X(x), Y(y), cs + 0.5, cs + 0.5); }

    // explored vs unexplored: one W x H mask image scaled up smoothly (no seams)
    if (layers.explored) {
      ctx.save(); ctx.imageSmoothingEnabled = true; ctx.drawImage(exploredMask(), X(0), Y(0), W * cs, H * cs); ctx.restore();
    }

    // grid
    if (layers.grid) {
      ctx.strokeStyle = "rgba(120,180,220,0.09)"; ctx.lineWidth = 1; ctx.beginPath();
      for (let x = 0; x <= W; x++) { ctx.moveTo(Math.round(X(x)) + 0.5, Y(0)); ctx.lineTo(Math.round(X(x)) + 0.5, Y(H)); }
      for (let y = 0; y <= H; y++) { ctx.moveTo(X(0), Math.round(Y(y)) + 0.5); ctx.lineTo(X(W), Math.round(Y(y)) + 0.5); }
      ctx.stroke();
    }

    // belief heatmap: smooth glow drawn from the real per-cell probabilities
    if (layers.heat) {
      const hm = buildHeat(cur);
      ctx.save();
      ctx.beginPath(); ctx.rect(X(0), Y(0), W * cs, H * cs); ctx.clip();
      ctx.globalAlpha = 0.95; ctx.imageSmoothingEnabled = true;
      ctx.drawImage(hm, X(0), Y(0), W * cs, H * cs);
      if (!reduceMotion || true) { ctx.filter = `blur(${Math.max(2, cs * 0.55)}px)`; ctx.globalCompositeOperation = "screen"; ctx.globalAlpha = 0.55; ctx.drawImage(hm, X(0), Y(0), W * cs, H * cs); }
      ctx.restore();
    }

    // previous local scans
    if (layers.clues) {
      ctx.save(); ctx.setLineDash([3, 4]); ctx.lineWidth = 1; ctx.strokeStyle = "rgba(56,225,240,0.35)";
      for (const s of D.scans) if (s.step <= cur) { ctx.beginPath(); ctx.arc(CX(s.center[0]), CY(s.center[1]), (s.radius + 0.5) * cs, 0, Math.PI * 2); ctx.stroke(); }
      ctx.restore();
    }

    // signal bearings
    if (layers.signals) {
      const len = (W + H) * cs;
      ctx.save(); ctx.beginPath(); ctx.rect(X(0), Y(0), W * cs, H * cs); ctx.clip();
      const sig = M.public.initial_signal;
      if (sig.bearing_deg != null) {
        const b = sig.bearing_deg * Math.PI / 180;
        ctx.setLineDash(DASH[3]); ctx.lineWidth = 1.4; ctx.strokeStyle = "rgba(227,238,247,0.4)";
        ctx.beginPath(); ctx.moveTo(CX(sig.origin[0]), CY(sig.origin[1])); ctx.lineTo(CX(sig.origin[0]) + Math.sin(b) * len, CY(sig.origin[1]) - Math.cos(b) * len); ctx.stroke();
      }
      for (const rd of D.readings) {
        if (rd.step > cur || rd.bearing_deg == null) continue;
        const lab = ((ev.belief.dna || {})[rd.source] || {}).label;
        const col = labelStyle(lab), fresh = rd.step === cur, focus = rd.source === selectedSource;
        const b = rd.bearing_deg * Math.PI / 180;
        ctx.setLineDash(DASH[D.sources.indexOf(rd.source) % 3]);
        ctx.lineWidth = fresh ? 2.2 : 1;
        ctx.globalAlpha = fresh ? 0.95 : (focus ? 0.32 : 0.12);
        ctx.strokeStyle = col;
        if (fresh) { ctx.shadowColor = col; ctx.shadowBlur = 8; }
        ctx.beginPath(); ctx.moveTo(CX(rd.origin[0]), CY(rd.origin[1])); ctx.lineTo(CX(rd.origin[0]) + Math.sin(b) * len, CY(rd.origin[1]) - Math.cos(b) * len); ctx.stroke();
        ctx.shadowBlur = 0;
      }
      ctx.restore(); ctx.globalAlpha = 1; ctx.setLineDash([]);
    }

    // route history
    if (layers.route && cur > 0) {
      ctx.save(); ctx.strokeStyle = C.cyan; ctx.lineWidth = Math.max(1.6, cs * 0.09);
      ctx.setLineDash([cs * 0.26, cs * 0.2]); ctx.lineDashOffset = playing || live.running ? -(now / 45) % 200 : 0;
      ctx.shadowColor = C.cyan; ctx.shadowBlur = 10; ctx.lineJoin = "round";
      ctx.beginPath(); ctx.moveTo(CX(E[0].position[0]), CY(E[0].position[1]));
      for (let i = 1; i < cur; i++) ctx.lineTo(CX(E[i].position[0]), CY(E[i].position[1]));
      ctx.lineTo(CX(AP[0]), CY(AP[1]));
      ctx.stroke(); ctx.restore();
    }

    // clues found so far
    if (layers.clues) for (const c of D.clues) {
      if (c.step > cur) continue;
      const x = CX(c.position[0]), y = CY(c.position[1]), s = Math.max(6, cs * 0.32);
      ctx.save(); ctx.shadowColor = C.cyan; ctx.shadowBlur = 10; ctx.lineWidth = 1.8; ctx.strokeStyle = C.cyan; ctx.fillStyle = "rgba(4,16,26,0.9)";
      if (c.type === "track") {
        ctx.beginPath(); ctx.moveTo(x, y - s); ctx.lineTo(x + s, y); ctx.lineTo(x, y + s); ctx.lineTo(x - s, y); ctx.closePath(); ctx.fill(); ctx.stroke();
        const b = c.bearing_deg * Math.PI / 180; ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + Math.sin(b) * s * 2.3, y - Math.cos(b) * s * 2.3); ctx.stroke();
      } else {
        ctx.beginPath(); ctx.arc(x, y, s, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        ctx.fillStyle = c.freshness === "fresh" ? C.cyan : C.dim; ctx.beginPath(); ctx.arc(x, y, s * 0.4, 0, Math.PI * 2); ctx.fill();
      }
      ctx.restore();
    }

    // base / return point
    {
      const [bx, by] = M.public.base, x = CX(bx), y = CY(by), R = Math.max(10, cs * 0.55);
      ctx.save(); ctx.strokeStyle = C.cyan; ctx.lineWidth = 1.6; ctx.fillStyle = "rgba(4,16,26,0.85)"; ctx.shadowColor = C.cyan; ctx.shadowBlur = 12;
      ctx.beginPath(); ctx.arc(x, y, R, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); ctx.shadowBlur = 0;
      ctx.globalAlpha = 0.35; ctx.beginPath(); ctx.arc(x, y, R * 1.45, 0, Math.PI * 2); ctx.stroke(); ctx.globalAlpha = 1;
      const s = R * 0.55; ctx.fillStyle = C.cyan;                      // house icon
      ctx.beginPath(); ctx.moveTo(x, y - s); ctx.lineTo(x + s, y - s * 0.05); ctx.lineTo(x + s * 0.68, y - s * 0.05); ctx.lineTo(x + s * 0.68, y + s * 0.8);
      ctx.lineTo(x - s * 0.68, y + s * 0.8); ctx.lineTo(x - s * 0.68, y - s * 0.05); ctx.lineTo(x - s, y - s * 0.05); ctx.closePath(); ctx.fill();
      ctx.fillStyle = "#04101a"; ctx.fillRect(x - s * 0.2, y + s * 0.25, s * 0.4, s * 0.55);
      ctx.restore();
    }

    // planned target of the decision that produces the next step
    const nx = E[cur + 1];
    if (nx && nx.decision && nx.decision.target && nx.decision.chosen_id.startsWith("goto")) {
      const t = nx.decision.target, tx = CX(t[0]), ty = CY(t[1]);
      ctx.save(); ctx.strokeStyle = "rgba(56,225,240,0.75)"; ctx.lineWidth = 1.2; ctx.setLineDash([3, 4]);
      ctx.beginPath(); ctx.moveTo(CX(ev.position[0]), CY(ev.position[1])); ctx.lineTo(tx, ty); ctx.stroke(); ctx.setLineDash([]);
      ctx.beginPath(); ctx.arc(tx, ty, cs * 0.42, 0, Math.PI * 2); ctx.stroke();
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { ctx.beginPath(); ctx.moveTo(tx + dx * cs * 0.3, ty + dy * cs * 0.3); ctx.lineTo(tx + dx * cs * 0.6, ty + dy * cs * 0.6); ctx.stroke(); }
      ctx.restore();
    }

    // most likely EVE location (belief)
    const top = ev.belief.top_cells[0];
    if (layers.heat && !ev.belief.confirmed) diamond(CX(top.cell[0]), CY(top.cell[1]), Math.max(7, cs * 0.3), C.amber);

    // EVE confirmed by WALL-E's own scan
    if (ev.belief.confirmed) {
      const [ex, ey] = ev.belief.confirmed; const x = CX(ex), y = CY(ey);
      ctx.save(); ctx.shadowColor = C.amber; ctx.shadowBlur = 18; ctx.strokeStyle = C.amber; ctx.lineWidth = 2.2; ctx.fillStyle = "rgba(255,181,71,0.22)";
      ctx.beginPath(); ctx.arc(x, y, cs * 0.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); ctx.restore();
      diamond(x, y, Math.max(7, cs * 0.3), C.amber);
    }

    // ground truth, evaluation view only
    if (showTruth && M.evaluation) {
      const [tx, ty] = M.evaluation.eve_position; star(CX(tx), CY(ty), Math.max(8, cs * 0.38), C.red);
      for (const d of Object.values(M.evaluation.decoys || {})) { ctx.strokeStyle = C.red; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(CX(d[0]), CY(d[1]), Math.max(5, cs * 0.25), 0, Math.PI * 2); ctx.stroke(); }
    }

    // scan ripple: only when this step really was a scan
    const act = ev.observation && ev.observation.ok ? ev.observation.action.type : null;
    if (act === "local_scan" || act === "long_scan") {
      const p = reduceMotion ? 1 : Math.min(1, (now - pulseStart) / 900);
      const R = act === "local_scan" ? ev.observation.scan.radius + 0.5 : 3.4;
      const x = CX(ev.position[0]), y = CY(ev.position[1]);
      ctx.save(); ctx.strokeStyle = C.cyan; ctx.lineWidth = 2;
      for (const k of [0, 0.35]) {
        const q = clamp(p - k, 0, 1); if (q <= 0 && !reduceMotion) continue;
        ctx.globalAlpha = reduceMotion ? 0.5 : 0.85 * (1 - q) + 0.1;
        ctx.beginPath(); ctx.arc(x, y, R * cs * (reduceMotion ? 1 : 0.25 + 0.75 * q), 0, Math.PI * 2); ctx.stroke();
      }
      ctx.restore(); if (p < 1) dirty = true;
    }

    // WALL-E (original rover glyph, rings, heading)
    {
      const x = CX(AP[0]), y = CY(AP[1]), s = Math.max(8, cs * 0.36);
      if (isCine()) {                                   // direction of travel, from the last real move
        const ha = { N: -Math.PI / 2, E: 0, S: Math.PI / 2, W: Math.PI }[D.heading[cur]], L = cs * 3.2;
        const gr = ctx.createRadialGradient(x, y, s, x, y, L);
        gr.addColorStop(0, "rgba(56,225,240,0.38)"); gr.addColorStop(1, "rgba(56,225,240,0)");
        ctx.save(); ctx.fillStyle = gr; ctx.beginPath(); ctx.moveTo(x, y); ctx.arc(x, y, L, ha - 0.45, ha + 0.45); ctx.closePath(); ctx.fill(); ctx.restore();
      }
      ctx.save(); ctx.strokeStyle = C.cyan;
      ctx.globalAlpha = 0.25; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(x, y, s * 2.4, 0, Math.PI * 2); ctx.stroke();
      ctx.globalAlpha = 0.5; ctx.beginPath(); ctx.arc(x, y, s * 1.65, 0, Math.PI * 2); ctx.stroke();
      ctx.globalAlpha = 1; ctx.shadowColor = C.cyan; ctx.shadowBlur = 18; ctx.fillStyle = "rgba(4,18,30,0.95)"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(x, y, s * 1.05, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); ctx.shadowBlur = 0;
      const a = { N: 0, E: Math.PI / 2, S: Math.PI, W: -Math.PI / 2 }[D.heading[cur]];
      ctx.translate(x, y); ctx.rotate(a); ctx.fillStyle = C.cyan;
      ctx.fillRect(-s * 0.55, -s * 0.45, s * 0.22, s * 0.95); ctx.fillRect(s * 0.33, -s * 0.45, s * 0.22, s * 0.95);   // treads
      ctx.fillRect(-s * 0.28, -s * 0.3, s * 0.56, s * 0.62);                                                         // body
      ctx.beginPath(); ctx.moveTo(0, -s * 0.85); ctx.lineTo(s * 0.22, -s * 0.5); ctx.lineTo(-s * 0.22, -s * 0.5); ctx.closePath(); ctx.fill(); // heading
      ctx.restore();
    }

    if (selected) { ctx.strokeStyle = C.text; ctx.lineWidth = 2; ctx.strokeRect(X(selected[0]) + 1, Y(selected[1]) + 1, cs - 2, cs - 2); }

    // labels on dark pills, drawn last
    if (ev.belief.confirmed) pill(`EVE confirmed by scan ${cell(ev.belief.confirmed)}`, CX(ev.belief.confirmed[0]), CY(ev.belief.confirmed[1]) - cs * 1.1, C.amber, g);
    else if (layers.heat) pill(`Most likely EVE location ${cell(top.cell)} · ${pct(top.p, 1)}`, CX(top.cell[0]), CY(top.cell[1]) - cs * 1.05, C.amber, g);
    if (showTruth && M.evaluation) pill("Ground truth (evaluation only)", CX(M.evaluation.eve_position[0]), CY(M.evaluation.eve_position[1]) + cs * 1.1, C.red, g);

    // coordinates: y on the left, x along the bottom
    if (layers.grid) {
      ctx.fillStyle = C.dim; ctx.font = `500 11px "JetBrains Mono", monospace`; ctx.textBaseline = "middle";
      const every = cs < 16 ? 2 : 1;
      ctx.textAlign = "right";
      for (let y = 0; y < H; y += every) { const sy = CY(y); if (sy > 6 && sy < g.h - g.gb) ctx.fillText(y, Math.min(g.gl - 6, X(0) - 6), sy); }
      ctx.textAlign = "center";
      for (let x = 0; x < W; x += every) { const sx = CX(x); if (sx > g.gl && sx < g.w) ctx.fillText(x, sx, Math.max(Y(H) + 12, Math.min(g.h - 10, Y(H) + 12))); }
    }

    if (isCine()) drawCineOverlay(g, now);

    // compass
    {
      const x = 26, y = 26; ctx.save(); ctx.strokeStyle = "rgba(147,168,189,0.7)"; ctx.lineWidth = 1.2; ctx.fillStyle = "rgba(4,10,18,0.8)";
      ctx.beginPath(); ctx.arc(x, y, 16, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.fillStyle = C.text; ctx.beginPath(); ctx.moveTo(x, y - 11); ctx.lineTo(x + 6, y + 7); ctx.lineTo(x, y + 3); ctx.lineTo(x - 6, y + 7); ctx.closePath(); ctx.fill();
      ctx.font = "600 10px Rajdhani, sans-serif"; ctx.textAlign = "center"; ctx.fillStyle = C.dim; ctx.fillText("N", x, y - 21 < 4 ? y + 24 : y - 21); ctx.restore();
    }
  }
  function rrect(x, y, w, h, r, fill, stroke) {
    ctx.beginPath(); ctx.roundRect(x, y, w, h, r);
    if (fill) { ctx.fillStyle = fill; ctx.fill(); }
    if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.stroke(); }
  }
  function fitText(t, maxW) {
    if (ctx.measureText(t).width <= maxW) return t;
    while (t.length > 3 && ctx.measureText(t + "…").width > maxW) t = t.slice(0, -1);
    return t + "…";
  }
  function drawCineOverlay(g, now) {
    const ev = E[cur], last = E.length - 1;
    ctx.save();
    // caption: what WALL-E is doing at this step
    const capX = 52, capW = Math.min(g.w * 0.56, 620);
    rrect(capX, 10, capW, 46, 10, "rgba(4,10,18,0.78)", "rgba(86,190,230,0.3)");
    ctx.textBaseline = "middle"; ctx.textAlign = "left";
    ctx.font = "700 12px Rajdhani, sans-serif"; ctx.fillStyle = C.cyan;
    ctx.fillText(`STEP ${cur} / ${last}`, capX + 12, 24);
    ctx.font = "600 14px Barlow, sans-serif"; ctx.fillStyle = C.text;
    ctx.fillText(fitText(actionLabel(ev), capW - 110), capX + 92, 24);
    ctx.font = "400 12.5px Barlow, sans-serif"; ctx.fillStyle = C.dim;
    const why = ev.decision ? ev.decision.plain_english : "Mission start: WALL-E is at base with only the initial signal.";
    ctx.fillText(fitText(why, capW - 24), capX + 12, 43);
    // stats: battery, best guess, status
    const sw = 300, sx = g.w - sw - 14;
    if (sx > capX + capW + 10) {
      rrect(sx, 10, sw, 46, 10, "rgba(4,10,18,0.78)", "rgba(86,190,230,0.3)");
      const cap = M.public.battery_capacity, bf = clamp(ev.battery / cap, 0, 1);
      ctx.font = "700 11px Rajdhani, sans-serif"; ctx.fillStyle = C.dim; ctx.textAlign = "left";
      ctx.fillText("BATTERY", sx + 12, 24);
      rrect(sx + 96, 21, 120, 6, 3, "rgba(120,160,200,0.18)");
      rrect(sx + 96, 21, Math.max(3, 120 * bf), 6, 3, bf < 0.25 ? C.amber : C.cyan);
      ctx.font = "500 12px JetBrains Mono, monospace"; ctx.fillStyle = C.text; ctx.textAlign = "right";
      ctx.fillText(`${ev.battery.toFixed(0)}/${cap}`, sx + sw - 12, 24);
      const top = ev.belief.confirmed || ev.belief.top_cells[0].cell;
      ctx.textAlign = "left"; ctx.font = "700 11px Rajdhani, sans-serif"; ctx.fillStyle = C.dim;
      ctx.fillText(ev.belief.confirmed ? "EVE FOUND" : "BEST GUESS", sx + 12, 43);
      ctx.font = "500 12px JetBrains Mono, monospace"; ctx.fillStyle = C.amber;
      ctx.fillText(cell(top), sx + 96, 43);
      ctx.fillStyle = C.dim; ctx.textAlign = "right";
      ctx.fillText(ev.belief.confirmed ? "confirmed" : `${pct(D.conf[cur])} in range`, sx + sw - 12, 43);
      ctx.textAlign = "left";
    }
    // timeline band (video-player style)
    const by = g.h - 66, bh = 50, bx = 14, bw = g.w - 28;
    rrect(bx, by, bw, bh, 14, "rgba(4,10,18,0.82)", "rgba(86,190,230,0.22)");
    ctx.font = "600 13px Barlow, sans-serif";
    const label = `Mission replay · PH-${String(M.meta.seed).padStart(4, "0")} · ${M.meta.agent}`;
    const lw = ctx.measureText(label).width + 22;
    rrect(bx + 10, by + 12, lw, 26, 13, "rgba(227,238,247,0.1)", "rgba(227,238,247,0.25)");
    ctx.fillStyle = C.text; ctx.fillText(label, bx + 21, by + 25.5);
    const ctlW = cine.rec ? 10 : Math.min(360, $("cineCtl").offsetWidth + 24 || 360);   // the video has no buttons, so use the full width
    const x0 = bx + 10 + lw + 22, x1 = Math.max(x0 + 60, bx + bw - ctlW - 10), ty = by + 25;
    const X = (i) => x0 + (i / Math.max(1, last)) * (x1 - x0), px = X(cur);
    ctx.lineCap = "round";
    ctx.strokeStyle = "rgba(255,181,71,0.45)"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(px, ty); ctx.lineTo(x1, ty); ctx.stroke();
    ctx.strokeStyle = C.cyan; ctx.shadowColor = C.cyan; ctx.shadowBlur = 6; ctx.beginPath(); ctx.moveTo(x0, ty); ctx.lineTo(px, ty); ctx.stroke(); ctx.shadowBlur = 0;
    // event dots: past = cyan, still to come = amber
    for (const k of D.keyEvents) {
      if (!["local", "clue", "detect", "safety", "label", "fail", "end"].includes(k.type)) continue;
      const big = k.type === "detect" || k.type === "end";
      ctx.fillStyle = k.type === "fail" ? C.red : k.step <= cur ? C.cyan : C.amber;
      ctx.beginPath(); ctx.arc(X(k.step), ty, big ? 5 : 3.5, 0, Math.PI * 2); ctx.fill();
    }
    // playhead
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(px, by + 7); ctx.lineTo(px, by + bh - 7); ctx.stroke();
    ctx.fillStyle = "#ffffff"; ctx.beginPath(); ctx.arc(px, by + 7, 4, 0, Math.PI * 2); ctx.fill();
    ctx.font = "500 10.5px JetBrains Mono, monospace"; ctx.fillStyle = C.faint; ctx.textAlign = "center";
    ctx.fillText("0", x0, by + bh - 6); ctx.fillText(String(last), x1, by + bh - 6);
    cine.band = { x0, x1, y0: by, y1: by + bh, last };
    ctx.restore();
  }
  function diamond(x, y, s, col) {
    ctx.save(); ctx.shadowColor = col; ctx.shadowBlur = 14; ctx.fillStyle = "rgba(20,8,0,0.85)"; ctx.strokeStyle = col; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x, y - s); ctx.lineTo(x + s, y); ctx.lineTo(x, y + s); ctx.lineTo(x - s, y); ctx.closePath(); ctx.fill(); ctx.stroke();
    ctx.fillStyle = col; ctx.beginPath(); ctx.moveTo(x, y - s * 0.4); ctx.lineTo(x + s * 0.4, y); ctx.lineTo(x, y + s * 0.4); ctx.lineTo(x - s * 0.4, y); ctx.closePath(); ctx.fill(); ctx.restore();
  }
  function star(x, y, s, col) {
    ctx.save(); ctx.fillStyle = col; ctx.shadowColor = col; ctx.shadowBlur = 12; ctx.beginPath();
    for (let k = 0; k < 10; k++) { const a = -Math.PI / 2 + k * Math.PI / 5, r = k % 2 ? s * 0.45 : s; ctx.lineTo(x + r * Math.cos(a), y + r * Math.sin(a)); }
    ctx.closePath(); ctx.fill(); ctx.restore();
  }
  function pill(text, x, y, col, g) {
    ctx.save(); ctx.font = "600 12.5px Barlow, sans-serif";
    const w = ctx.measureText(text).width + 16, h = 22;
    const lx = clamp(x - w / 2, 4, g.w - w - 4), ly = clamp(y - h / 2, isCine() ? 64 : 4, g.h - h - (isCine() ? 72 : 4));
    ctx.fillStyle = "rgba(4,10,18,0.92)"; ctx.strokeStyle = col; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.roundRect(lx, ly, w, h, 6); ctx.fill(); ctx.stroke();
    ctx.fillStyle = col; ctx.textAlign = "left"; ctx.textBaseline = "middle"; ctx.fillText(text, lx + 8, ly + h / 2 + 0.5); ctx.restore();
  }
  function loop(now) { if (cine.rec) dirty = true; if (dirty) { dirty = false; draw(now); } requestAnimationFrame(loop); }

  // ================================================================ event strip
  const STRIP = { long: ["rgba(56,225,240,0.45)", 6, "Long scan"], local: [C.cyan, 10, "Local scan"], clue: ["#9ff3ff", 14, "Clue"],
                  safety: [C.text, 12, "Return / recharge"], label: [C.amber, 12, "Signal trust change"], detect: [C.amber, 20, "EVE detected"],
                  fail: [C.red, 16, "Failure"], end: [C.green, 16, "End"] };
  function drawStrip() {
    const c = $("eventStrip"), r = c.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    if (!r.width) return;
    c.width = r.width * dpr; c.height = r.height * dpr; const g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, r.width, r.height);
    if (!M) return;
    const n = Math.max(1, live.on ? Math.max(E.length - 1, M.public.time_budget / 3) : E.length - 1), pad = 8, w = r.width - 2 * pad;
    for (const k of D.keyEvents) { if (!STRIP[k.type]) continue; const [col, h] = STRIP[k.type]; g.fillStyle = col; g.fillRect(pad + (k.step / n) * w - 1, 20 - h, 2, h); }
    $("stripKey").innerHTML = Object.values(STRIP).map(([col, , l]) => `<span><i style="background:${col}"></i>${l}</span>`).join("");
  }

  // ================================================================ step change + status
  function setStep(i) {
    if (!M) return;
    const prev = cur;
    cur = clamp(i, 0, E.length - 1);
    cine.anim = (playing || cine.rec) && cur === prev + 1 && !reduceMotion
      ? { from: E[prev].position, to: E[cur].position, start: performance.now(), dur: 850 / speed.replay } : null;
    $("scrubber").max = E.length - 1; $("scrubber").value = cur; $("stepNow").textContent = cur;
    pulseStart = performance.now();
    renderView(); redraw(); drawStrip(); updateStatus();
    if (selected) renderInspector();
  }

  function outcomeStatus() {
    const last = E[E.length - 1], o = last.observation;
    if (!o || !o.mission_over) return null;
    if (o.end_reason === "stranded") return ["Failed: stranded", "red"];
    if (last.belief.confirmed) return ["Complete: EVE located", "green"];
    return ["Ended: EVE not located", "amber"];
  }
  function updateStatus() {
    let s;
    const out = outcomeStatus();
    if (live.on) {
      if (out && cur === E.length - 1) s = out;
      else if (live.running) s = ["Running", "cyan"];
      else if (E.length === 1) s = ["Ready", "cyan"];
      else if (out) s = ["Replay", "cyan"];
      else s = ["Paused", "amber"];
    } else s = out && cur === E.length - 1 ? out : ["Replay", "cyan"];
    const el = $("sysStatus"); el.className = "v status " + s[1]; el.lastElementChild.textContent = s[0];
    return s;
  }

  // ================================================================ text helpers from real trace data
  function actionLabel(ev) {
    if (!ev.observation) return "Mission start";
    const a = ev.observation.action, o = ev.observation;
    if (a.type === "move") return `Move ${WORD[a.direction]} to ${cell(o.position)}`;
    if (a.type === "long_scan") return `Long scan at ${cell(o.position)}`;
    if (a.type === "local_scan") return `Local scan at ${cell(o.position)}`;
    if (a.type === "recharge") return "Recharge at base";
    return "End mission";
  }
  function observationText(ev) {
    const o = ev.observation; if (!o) return "No action yet.";
    if (!o.ok) return `Refused by the environment: ${o.error}.`;
    if (o.action.type === "move") return `Arrived at ${cell(o.position)}. Battery ${o.battery_delta.toFixed(1)}.`;
    if (o.action.type === "long_scan") {
      const parts = o.readings.map((r) => r.bearing_deg == null ? `${r.source} silent` : `${r.source} ${r.bearing_deg}°${r.strength != null ? ` (strength ${r.strength})` : ""}`);
      return `Heard: ${parts.join(", ")}.`;
    }
    if (o.action.type === "local_scan") {
      if (o.scan.eve_detected) return `EVE detected at ${cell(o.scan.eve_position)}. Status reading: ${o.scan.status_reading}.`;
      const cl = o.clues.filter((c) => c.first_time);
      return `Nothing within ${o.scan.radius} cells${cl.length ? `, but found ${cl.map((c) => c.type + " clue").join(" and ")}` : ""}.`;
    }
    if (o.action.type === "recharge") return `Battery now ${o.battery.toFixed(0)} of ${M.public.battery_capacity}.`;
    return `Mission over: ${String(o.end_reason).replace("_", " ")}.`;
  }
  function logLine(i) {
    const ev = E[i], o = ev.observation, d = ev.decision;
    let dot = "dim", text = actionLabel(ev);
    if (!o.ok) { dot = "red"; text = `Failed: ${o.error}`; }
    else if (o.action.type === "long_scan") { dot = "cyan"; text = "Long scan: " + o.readings.map((r) => r.bearing_deg == null ? `${r.source} silent` : `${r.source} ${r.bearing_deg}°`).join(", "); }
    else if (o.action.type === "local_scan") { dot = o.scan.eve_detected ? "amber" : "cyan"; text = o.scan.eve_detected ? `Scan ${cell(o.position)}: EVE detected` : `Scanned ${cell(o.position)}: ${o.clues.some((c) => c.first_time) ? "clue found" : "no signal"}`; }
    else if (o.action.type === "recharge") { dot = "white"; text = `Recharged to ${o.battery.toFixed(0)}`; }
    else if (o.action.type === "end_mission") { dot = "white"; text = `Mission ended`; }
    if (d && d.forced_by && o.action.type === "move") { dot = "amber"; text += " (safety rule)"; }
    return { dot, text };
  }
  const riskWord = (r) => r == null ? ["–", ""] : r < 0.25 ? ["Low", "risk-low"] : r < 0.6 ? ["Med", "risk-med"] : ["High", "risk-high"];
  function confidenceWord(p) { return p >= 0.9 ? ["Confident", "green"] : p >= 0.4 ? ["Narrowing", "cyan"] : ["Uncertain", "amber"]; }
  function uncertaintyWord(bits) { return bits > 6 ? ["High", "amber"] : bits > 2.5 ? ["Medium", "cyan"] : ["Low", "green"]; }

  // ================================================================ icons
  const ICON = {
    rover: `<svg viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linejoin="round"><rect x="14" y="26" width="36" height="16" rx="4"/><path d="M32 26V14M24 10a10 10 0 0 1 16 0"/><circle cx="32" cy="14" r="2.5" fill="currentColor"/><circle cx="18" cy="48" r="5"/><circle cx="32" cy="48" r="5"/><circle cx="46" cy="48" r="5"/><path d="M20 32h24"/></svg>`,
    beacon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="2.5" fill="currentColor"/><path d="M7 7a7 7 0 0 0 0 10M17 7a7 7 0 0 1 0 10M4 4a11 11 0 0 0 0 16M20 4a11 11 0 0 1 0 16"/></svg>`,
    relay: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3v18M5 9l7-6 7 6M7 14h10"/></svg>`,
    initial: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="8"/><path d="M12 7v5l3 3"/></svg>`,
  };
  const srcIcon = (name) => (M.public.sources[name] || {}).kind === "beacon" ? ICON.beacon : ICON.relay;
  const labelClass = (l) => l === "trusted" ? "cyan" : l === "conflicting" ? "red" : l === "doubtful" ? "amber" : "dim";

  // ================================================================ panels
  function panel(title, body, extra = "") { return `<section class="panel"><div class="panel-title"><span>${title}</span>${extra}</div>${body}</section>`; }

  function pAgent() {
    const ev = E[cur], cap = M.public.battery_capacity, bat = ev.battery, c = D.counts[cur];
    const home = D.homeCost[ev.position[1] * W + ev.position[0]], spare = bat - home - M.public.reserve;
    const bcls = spare < 0 ? "red" : spare < 0.15 * cap ? "amber" : "";
    return panel("Agent status", `
      <div class="agent"><div class="agent-emblem">${ICON.rover}</div>
        <div><div class="agent-name">WALL-E</div><div class="agent-role">Autonomous search agent · ${esc(M.meta.agent)}</div></div></div>
      <div class="kv"><span class="k">Position</span><span class="v num">${cell(ev.position)}</span>
        <span class="k">Steps taken</span><span class="v num">${cur} / ${M.public.time_budget}</span>
        <span class="k">Local / long scans</span><span class="v num">${c.local} / ${c.long}</span>
        <span class="k">Clues found</span><span class="v num">${c.clues}</span></div>
      <div style="margin-top:12px" class="battery-row"><span class="dim">Battery</span>
        <div class="bar" title="Amber tick = battery needed to drive home plus reserve"><div class="fill ${bcls}" style="width:${pct(bat / cap, 2)}"></div><div class="tick" style="left:${pct(clamp((home + M.public.reserve) / cap, 0, 1), 2)}"></div></div>
        <span class="num">${bat.toFixed(0)}<span class="dim"> / ${cap}</span></span></div>
      <div class="small dim" style="margin-top:6px">Drive home needs ${home.toFixed(1)} + ${M.public.reserve} reserve · <span class="${spare < 0 ? "red" : ""}">${spare >= 0 ? spare.toFixed(1) + " spare" : "below safe level"}</span></div>`);
  }

  function pMissionStatus() {
    const s = updateStatus(), ev = E[cur], top = ev.belief.confirmed || ev.belief.top_cells[0].cell;
    const cf = D.conf[cur], [cw, cc] = ev.belief.confirmed ? ["Confirmed", "green"] : confidenceWord(cf);
    const pa = ev.belief.p_active;
    return panel("Mission status", `
      <div style="display:flex;justify-content:space-between;align-items:center;gap:10px;margin-bottom:10px"><span class="pill ${s[1]}">${esc(s[0])}</span><span class="small dim">${live.on ? "Live simulation" : "Saved run replay"}</span></div>
      <div class="small dim">Objective</div><div style="margin-bottom:10px">Locate EVE, infer her status, and decide WALL-E's next move.</div>
      <div class="kv"><span class="k">Most likely EVE cell</span><span class="v num amber">${cell(top)}</span>
        <span class="k">Belief</span><span class="v ${cc}">${cw}</span>
        <span class="k">Within scan range</span><span class="v num">${pct(cf)}</span>
        <span class="k">EVE status guess</span><span class="v">${pa >= 0.5 ? "active" : "dormant"} <span class="num dim">${pct(Math.max(pa, 1 - pa))}</span></span></div>`);
  }

  function pSources() {
    const ev = E[cur], dna = ev.belief.dna || {};
    const rows = D.sources.map((s) => {
      const d = dna[s], rel = d ? d.reliability : (ev.belief.reliability_used || {})[s];
      const lab = d ? d.label : "fixed", col = labelClass(lab);
      const n = D.readings.filter((r) => r.source === s && r.step <= cur && r.bearing_deg != null).length;
      return `<button class="src ${s === selectedSource ? "active" : ""}" data-src="${esc(s)}"><span class="src-icon ${col}">${srcIcon(s)}</span>
        <span><span class="src-name">${esc(s)}</span><span class="src-sub">${esc(M.public.sources[s].kind)} · ${n} reading${n === 1 ? "" : "s"} · <span class="${col}">${esc(lab)}</span></span></span>
        <span class="src-val ${col}">${rel != null ? rel.toFixed(2) : "–"}</span>
        <span class="bar"><span class="fill ${col === "red" ? "red" : col === "amber" ? "amber" : ""}" style="width:${pct(rel || 0, 1)}"></span></span></button>`;
    }).join("");
    const t = ev.belief.initial_signal_trust;
    const init = `<button class="src ${selectedSource === "__initial" ? "active" : ""}" data-src="__initial"><span class="src-icon ${t < 0.5 ? "amber" : "dim"}">${ICON.initial}</span>
      <span><span class="src-name">Initial signal</span><span class="src-sub">heard before launch · ${t < 0.5 ? "stale" : "aging"}</span></span><span class="src-val">${t.toFixed(2)}</span>
      <span class="bar"><span class="fill ${t < 0.5 ? "amber" : ""}" style="width:${pct(t, 1)}"></span></span></button>`;
    return panel("Signal sources", rows + init + `<div class="small faint" style="margin-top:8px">Number = learned reliability (0 never right, 1 always right). Click a source for details.</div>`);
  }

  function pLog() {
    const items = [];
    for (let i = cur; i >= 1 && items.length < 9; i--) { const l = logLine(i); items.push(`<li class="${i === cur ? "now" : ""}"><span class="dot ${l.dot}"></span><span class="st">#${i}</span><button data-step="${i}">${esc(l.text)}</button></li>`); }
    return panel("Recent log", items.length ? `<ul class="log">${items.join("")}</ul>` : `<p class="empty">No actions yet. ${live.on ? "Press Start or Step." : "Press play."}</p>`);
  }

  function pLatest() {
    const ev = E[cur];
    if (!ev.decision) {
      const sig = M.public.initial_signal;
      return panel("Latest action", `<div class="action-big">Waiting at base</div>
        <p class="action-obs">Initial signal: bearing <span class="num">${sig.bearing_deg ?? "–"}°</span>${sig.strength != null ? `, strength <span class="num">${sig.strength}</span>` : ""}, trusted at <span class="num">${pct(ev.belief.initial_signal_trust)}</span> because it is ${Math.round((Date.parse(M.public.start_time) - Date.parse(sig.timestamp)) / 60000)} minutes old.</p>
        <p class="action-why">${live.on ? "Press Start to run the agent, or Step to advance one action." : "Press play or drag the timeline to replay the mission."}</p>`);
    }
    const d = ev.decision;
    return panel("Latest action", `<div class="action-big">${esc(actionLabel(ev))}</div>
      <p class="action-obs"><span class="dim">Observation:</span> ${esc(observationText(ev))}</p>
      <p class="action-why"><span class="dim">Why:</span> ${esc(d.plain_english)}</p>
      ${d.forced_by ? `<div class="callout amber"><span class="t">Safety rule applied:</span> ${esc(d.forced_by)}</div>` : ""}`, `<span class="num small dim" style="letter-spacing:0">step ${cur}</span>`);
  }

  function shortLabel(c) {
    const t = c.target;
    if (c.type === "goto_scan") return `Scan ${cell(t)}`;
    if (c.id === "scan_here") return "Scan here";
    if (c.id === "status_scan") return "Status scan";
    if (c.type === "long_scan") return "Long scan";
    if (c.type === "return_base") return "Return to base";
    if (c.type === "recharge") return "Recharge";
    if (c.type === "end_mission") return "End mission";
    return c.label;
  }
  function candidateRows(cands, compact) {
    const scored = cands.filter((c) => c.score != null), mx = Math.max(1e-9, ...scored.map((c) => c.score));
    let list = [...cands].sort((a, b) => (b.chosen - a.chosen) || ((b.feasible && b.score != null) - (a.feasible && a.score != null)) || ((b.score ?? -1) - (a.score ?? -1)));
    if (compact) { const ch = list.find((c) => c.chosen); list = [ch, ...list.filter((c) => !c.chosen && c.score != null && c.feasible)].slice(0, 5); }
    return list.map((c) => {
      const [rw, rc] = riskWord(c.score == null ? null : c.risk);
      const cls = c.chosen ? "chosen" : !c.feasible ? "vetoed" : "";
      return `<tr class="${cls}"><td title="${esc(c.label)}">${esc(shortLabel(c))}${!compact && c.veto ? `<span class="veto">Vetoed: ${esc(c.veto)}</span>` : ""}${!compact && c.score != null ? `<div class="scorebar" style="width:${Math.max(2, 100 * c.score / mx)}%"></div>` : ""}</td>
        <td class="n">${c.eig_bits != null ? c.eig_bits.toFixed(2) : "–"}</td><td class="n">${c.battery_cost != null ? c.battery_cost.toFixed(1) : "–"}</td>
        ${compact ? "" : `<td class="n">${c.time_cost ?? "–"}</td>`}<td class="n ${rc}">${rw}</td><td class="n">${c.score != null ? c.score.toFixed(3) : "–"}</td></tr>`;
    }).join("");
  }
  function selectedWhy(d, ch) {
    if (d.forced_by) return `${d.forced_by}: ${d.reason}`;
    const others = d.candidates.filter((c) => !c.chosen && c.feasible && c.score != null).sort((a, b) => b.score - a.score);
    const rw = ch.risk < 0.25 ? "low" : ch.risk < 0.6 ? "medium" : "high";
    return `Best information per unit of effort: ${ch.eig_bits.toFixed(2)} bits for ${ch.battery_cost.toFixed(1)} battery, ${rw} risk` +
      (others.length ? `. Runner-up "${shortLabel(others[0])}" scored ${others[0].score.toFixed(3)} vs ${ch.score.toFixed(3)}.` : ".");
  }
  function pDecision(full) {
    const ev = E[cur];
    if (!ev.decision) return panel(full ? "Ghost decisions" : "Decision inspector", `<p class="empty">Decisions appear from step 1.</p>`);
    const d = ev.decision, cands = d.candidates;
    if (cands.length === 1 && cands[0].type === "rule")
      return panel(full ? "Ghost decisions" : "Decision inspector", `<p>The <b>${esc(M.meta.agent)}</b> baseline follows fixed rules, so there are no scored alternatives.</p><div class="callout"><span class="t">Action:</span> ${esc(d.reason)}</div>`);
    const chosen = cands.find((c) => c.chosen);
    const head = full ? `<th>Option</th><th class="n" title="Expected information gain in bits">Info</th><th class="n" title="Battery cost">Batt</th><th class="n">Ticks</th><th class="n">Risk</th><th class="n">Score</th>`
                      : `<th>Action</th><th class="n" title="Expected information gain in bits">Info</th><th class="n" title="Battery cost">Cost</th><th class="n">Risk</th><th class="n">Score</th>`;
    const body = `${full ? `<p class="small dim" style="margin:0 0 10px">Every option WALL-E scored before step ${cur}. The rejected ones are the ghosts. "Scan (x, y)" means drive there, then local-scan.</p>` : `<p class="small faint" style="margin:-4px 0 8px">"Scan (x, y)" = drive there, then local-scan. Info gain in bits, cost in battery.</p>`}
      <table class="dt"><thead><tr>${head}</tr></thead><tbody>${candidateRows(cands, !full)}</tbody></table>
      <div class="callout ${d.forced_by ? "amber" : "green"}"><span class="t">Selected action: ${esc(shortLabel(chosen))}</span><div class="small" style="margin-top:3px">${full ? esc(d.reason) : esc(selectedWhy(d, chosen))}</div></div>
      ${full ? `<div class="formula">score = info ÷ (battery + 0.5 × ticks) × (1 − 0.5 × risk)<br>+10% when continuing toward the current target (stops dithering)<br>vetoed if battery left &lt; cost to drive home + reserve</div>
        <p class="small faint" style="margin-top:8px">Info = expected drop in uncertainty about EVE, in bits. Local scans are exact; long-range scans are estimated from 24 simulated readings. A hand-designed rule, not machine learning.</p>` : ""}`;
    return panel(full ? "Ghost decisions" : "Decision inspector", body, full ? "" : `<button class="more" data-goview="investigation">All options ›</button>`);
  }

  function pSignal() {
    const ev = E[cur], s = selectedSource;
    if (s === "__initial") {
      const sig = M.public.initial_signal, t = ev.belief.initial_signal_trust, age = Math.round((Date.parse(M.public.start_time) - Date.parse(sig.timestamp)) / 60000);
      return panel("Signal analysis", `<div class="dna-head"><span class="nm"><span class="src-icon ${t < 0.5 ? "amber" : "dim"}" style="width:28px;height:28px">${ICON.initial}</span>Initial signal</span><span class="pill ${t < 0.5 ? "amber" : "dim"}">${t < 0.5 ? "stale" : "aging"}</span></div>
        <div class="kv"><span class="k">Direction</span><span class="v num">${sig.bearing_deg != null ? sig.bearing_deg + "°" : "–"}</span>
        ${sig.coordinates ? `<span class="k">Coordinates</span><span class="v num">${cell(sig.coordinates)}</span>` : ""}
        <span class="k">Strength</span><span class="v num">${sig.strength ?? "–"}</span><span class="k">Heard from</span><span class="v num">${cell(sig.origin)}</span>
        <span class="k">Timestamp</span><span class="v num">${esc(sig.timestamp)}</span><span class="k">Age at launch</span><span class="v num">${age} min</span>
        <span class="k">Trust used</span><span class="v num">${pct(t)}</span></div>
        <p class="small dim" style="margin-top:8px">Trust = ${pct(M.public.agent.initial_signal_trust)} × ½ for every ${M.public.agent.staleness_half_life_min} minutes of age.</p>`);
    }
    const dna = (ev.belief.dna || {})[s], spec = M.public.sources[s];
    const last = [...D.readings].reverse().find((r) => r.source === s && r.step <= cur);
    const heard = D.readings.filter((r) => r.source === s && r.step <= cur);
    const view = (ev.belief.source_view || {})[s];
    const est = ev.belief.confirmed || ev.belief.top_cells[0].cell;
    let agree = "";
    if (view && view.solo_top) {
      const dd = Math.hypot(view.solo_top.cell[0] - est[0], view.solo_top.cell[1] - est[1]);
      agree = dd <= 3 ? `<span class="cyan">agrees with the overall estimate</span>` : `<span class="amber">points ${dd.toFixed(0)} cells away from the overall estimate</span>`;
    }
    const lab = dna ? dna.label : "fixed", col = labelClass(lab);
    const warn = lab === "conflicting" ? `<div class="callout red"><span class="t">Conflicts with independent evidence.</span> Its readings rarely match empty scans, clues and the initial signal, so it now barely moves the heatmap.</div>`
      : lab === "doubtful" ? `<div class="callout amber"><span class="t">Doubtful.</span> Its readings disagree with independent evidence more often than not.</div>` : "";
    return panel("Signal analysis", `<div class="dna-head"><span class="nm"><span class="src-icon ${col}" style="width:28px;height:28px">${srcIcon(s)}</span>${esc(s)}</span><span class="pill ${col}">${esc(lab)}</span></div>
      <div class="kv"><span class="k">Type</span><span class="v">${spec.kind === "beacon" ? "Beacon (EVE's transmitter)" : "Relay report"}</span>
      <span class="k">Last reading</span><span class="v num">${last ? (last.bearing_deg != null ? `${last.bearing_deg}°${last.strength != null ? " · " + last.strength : ""} @ step ${last.step}` : `silent @ step ${last.step}`) : "none yet"}</span>
      <span class="k">Readings so far</span><span class="v num">${heard.filter((r) => r.bearing_deg != null).length}${spec.kind === "beacon" ? ` heard · ${heard.filter((r) => r.bearing_deg == null).length} silent` : ""}</span>
      <span class="k">Sensor noise</span><span class="v num">±${spec.noise_deg}°</span>
      <span class="k">Reliability</span><span class="v num ${col}">${dna ? dna.reliability.toFixed(2) : "0.80 fixed"}</span></div>
      ${dna ? `<div class="bar" style="margin:10px 0 2px"><div class="band" style="left:${pct(dna.interval_90[0], 1)};width:${pct(dna.interval_90[1] - dna.interval_90[0], 1)}"></div><div class="pt ${col === "red" ? "red" : col === "amber" ? "amber" : ""}" style="left:${pct(dna.reliability, 1)}"></div></div><div class="scale"><span>0</span><span>90% range ${dna.interval_90[0].toFixed(2)}–${dna.interval_90[1].toFixed(2)}</span><span>1</span></div>` : ""}
      ${view && view.solo_top ? `<p class="small" style="margin:10px 0 0">On its own it points to <b class="num">${cell(view.solo_top.cell)}</b>, which ${agree}.</p>` : ""}
      ${warn}`, `<button class="more" data-goview="investigation">Signal DNA ›</button>`);
  }

  function pPhase() {
    const phases = [["listen", "Listen"], ["travel", "Travel"], ["scan", "Scan"], ["return", "Return"], ["located", "Located"]];
    const ph = (i) => {
      const ev = E[i]; if (!ev.observation) return "listen";
      if (ev.belief.confirmed) return "located";
      const a = ev.observation.action.type, id = ev.decision.chosen_id;
      if (a === "recharge" || id === "return_base") return "return";
      if (a === "long_scan") return "listen"; if (a === "local_scan") return "scan"; return "travel";
    };
    const now = ph(cur), seen = new Set(); for (let i = 1; i <= cur; i++) seen.add(ph(i));
    return panel("Mission phase", `<div class="phase">${phases.map(([k, l]) => `<div class="${k === now ? "on" : seen.has(k) ? "seen" : ""}"><b></b>${l}</div>`).join("")}</div>
      <h3>Confidence over time</h3><canvas class="chart" id="confChart"></canvas><div class="small faint">Share of WALL-E's belief inside scan range of his best guess.</div>`);
  }

  function pDNA() {
    const ev = E[cur], dna = ev.belief.dna || {};
    if (!Object.keys(dna).length) return panel("Signal DNA", `<p class="empty">The ${esc(M.meta.agent)} agent does not learn signal trust; it uses a fixed reliability for every source.</p>`);
    const anchored = Object.values(dna).some((d) => d.anchored);
    const cards = D.sources.map((s) => {
      const d = dna[s], col = labelClass(d.label), view = (ev.belief.source_view || {})[s];
      const last = [...D.readings].reverse().find((r) => r.source === s && r.step <= cur);
      return `<div class="dna-card"><div class="dna-head"><span class="nm"><span class="src-icon ${col}" style="width:28px;height:28px">${srcIcon(s)}</span>${esc(s)}</span><span class="pill ${col}">${esc(d.label)}</span></div>
        <div class="bar" style="margin:8px 0 2px"><div class="band" style="left:${pct(d.interval_90[0], 1)};width:${pct(d.interval_90[1] - d.interval_90[0], 1)}"></div><div class="pt ${col === "red" ? "red" : col === "amber" ? "amber" : ""}" style="left:${pct(d.reliability, 1)}"></div></div>
        <div class="scale"><span>0 never right</span><span>1 always right</span></div>
        <div style="margin-top:8px"><b class="num ${col}" style="font-size:18px">${d.reliability.toFixed(2)}</b> <span class="small dim">90% range ${d.interval_90[0].toFixed(2)}–${d.interval_90[1].toFixed(2)}</span></div>
        <div class="small dim">agreed a ${d.a.toFixed(1)} · disagreed b ${d.b.toFixed(1)} · judged ${d.judged}${d.pending ? ` · waiting ${d.pending}` : ""}</div>
        <div class="small dim">last ${last ? (last.bearing_deg != null ? last.bearing_deg + "° at step " + last.step : "silent at step " + last.step) : "none yet"}${view && view.solo_top ? ` · alone points to <span class="num">${cell(view.solo_top.cell)}</span>` : ""}</div>
        <canvas class="spark" data-src="${esc(s)}"></canvas></div>`;
    }).join("");
    return panel("Signal DNA", `<p class="small dim" style="margin-top:0">${anchored ? "EVE has been seen, so every past reading has been re-scored against her confirmed cell (the strongest evidence available)." : "Each reading is judged only against evidence no long-range signal produced: empty scans, physical clues and the initial signal. Sources never judge each other, so a confident decoy cannot make an honest source look bad."}</p>${cards}
      <div class="formula">reliability = a ÷ (a + b)<br>agreement = q_fit ÷ (q_fit + q_random)<br>conflicting only if 90% range &lt; 0.45 after ≥ 4 judged</div>`);
  }

  function pEvidence() {
    if (!cur) return panel("Evidence at this step", `<p class="empty">Only the initial signal so far; everything else starts equally likely.</p>`);
    const u = E[cur].belief_update, items = (u.new_evidence || []).map(describeEvidence);
    const j = (u.dna_judgments || []).map((x) => `<li><b>${esc(x.source)}</b> reading #${x.step}: agreement <span class="num">${x.agreement}</span> <span class="faint">${x.q_true != null ? `(fit ${x.q_true.toFixed(3)} vs random ${x.q_noise.toFixed(3)})` : `(error ${x.bearing_error_deg}° vs tolerance ${x.tolerance_deg}°)`}</span></li>`).join("");
    return panel("Evidence at this step", `${items.length ? `<ul class="ev-list">${items.join("")}</ul>` : `<p class="empty">No new evidence: moving does not change the belief.</p>`}
      ${j ? `<h3>Signal DNA judgments</h3><ul class="ev-list">${j}</ul>` : ""}`, `<span class="num small dim" style="letter-spacing:0">step ${cur}</span>`);
  }
  function describeEvidence(e) {
    switch (e.kind) {
      case "bearing": return `<li><b>${esc(e.source)}</b> bearing <span class="num">${e.bearing_deg}°</span> from ${cell(e.origin)}${e.strength != null ? `, strength <span class="num">${e.strength}</span>` : ""}. Cells along that line become more likely, weighted by trust.</li>`;
      case "beacon_silence": return `<li><b>${esc(e.source)}</b> silent: nearby cells and an active EVE become slightly less likely.</li>`;
      case "scan_empty": return `<li>Scan at ${cell(e.center)} found nothing: cells within ${e.radius} drop sharply (not to zero; the scanner misses 10%).</li>`;
      case "scan_detect": return `<li><b class="amber">EVE detected</b> at ${cell(e.eve_position)}.</li>`;
      case "status": return `<li>Status reading <b>${esc(e.status_reading)}</b> (right ${pct(e.accuracy)} of the time).</li>`;
      case "clue_track": return `<li>Track clue at ${cell(e.position)} pointing <span class="num">${e.bearing_deg}°</span>.</li>`;
      case "clue_residue": return `<li>Residue clue at ${cell(e.position)}: ${esc(e.freshness)}, says EVE is ${esc(e.proximity)}.</li>`;
      default: return `<li>${esc(JSON.stringify(e))}</li>`;
    }
  }

  function pKeyEvents() {
    const F = [["move", "Moves"], ["long", "Long scans"], ["local", "Local scans"], ["clue", "Clues"], ["detect", "Detections"], ["safety", "Return / recharge"], ["label", "Trust changes"], ["fail", "Failures"], ["end", "End"]];
    const list = D.keyEvents.filter((k) => eventFilter.has(k.type));
    return panel("Key events", `<div class="chips">${F.map(([k, l]) => `<button class="chip" data-filter="${k}" aria-pressed="${eventFilter.has(k)}">${l}</button>`).join("")}</div>
      <ul class="log">${list.map((k) => { const col = (STRIP[k.type] || ["#93a8bd"])[0]; return `<li class="${k.step === cur ? "now" : ""}"><span class="dot" style="background:${col}"></span><span class="st">#${k.step}</span><button data-step="${k.step}">${esc(k.text)}</button></li>`; }).join("") || `<li class="empty">No events match these filters.</li>`}</ul>`);
  }

  function pStateAtStep() {
    const ev = E[cur], c = D.counts[cur];
    return panel("State at this step", `<div class="kv"><span class="k">Step</span><span class="v num">${cur} of ${E.length - 1}</span>
      <span class="k">Position</span><span class="v num">${cell(ev.position)}</span><span class="k">Battery</span><span class="v num">${ev.battery.toFixed(1)} / ${M.public.battery_capacity}</span>
      <span class="k">Time left</span><span class="v num">${ev.time_left} ticks</span><span class="k">Moves</span><span class="v num">${c.moves}</span><span class="k">Local / long scans</span><span class="v num">${c.local} / ${c.long}</span>
      <span class="k">Clues found</span><span class="v num">${c.clues}</span><span class="k">Uncertainty</span><span class="v num">${ev.belief.entropy_bits.toFixed(2)} bits</span></div>
      <h3>Belief at step ${cur}</h3><div class="mini"><canvas id="miniNow" width="240" height="180" style="width:100%;height:auto"></canvas></div>`);
  }

  function evoPanel() {
    if (!M) return "";
    const ev = E[cur], prev = E[Math.max(0, cur - 1)], u = ev.belief_update;
    let sum = 0, ok = true; for (const v of ev.belief.location) { if (!Number.isFinite(v) || v < 0) ok = false; sum += v; }
    const valid = ok && Math.abs(sum - 1) < 2e-3;
    return `<div class="panel-title"><span>Belief evolution</span><span class="small ${valid ? "green" : "red"}" style="letter-spacing:0;text-transform:none;font-family:var(--body)">${valid ? "✓" : "✗"} probabilities finite, ≥ 0, sum ${sum.toFixed(3)}</span></div>
      <div class="evo"><div class="mini"><canvas id="miniPrev" width="120" height="90"></canvas>Before · step ${Math.max(0, cur - 1)}</div><div class="arrow">→</div>
      <div class="mini"><canvas id="miniCur" width="120" height="90"></canvas>After · step ${cur}</div>
      <div class="kv"><span class="k">Uncertainty</span><span class="v num">${u ? `${u.entropy_before.toFixed(2)} → ${u.entropy_after.toFixed(2)} bits` : `${ev.belief.entropy_bits.toFixed(2)} bits`}</span>
        <span class="k">Information gained</span><span class="v num">${u ? u.info_gained_bits.toFixed(3) + " bits" : "–"}</span>
        <span class="k">Most likely cell</span><span class="v num">${cell(prev.belief.top_cells[0].cell)} → ${cell(ev.belief.top_cells[0].cell)}</span>
        <span class="k">Its probability</span><span class="v num">${pct(prev.belief.top_cells[0].p, 1)} → ${pct(ev.belief.top_cells[0].p, 1)}</span></div></div>`;
  }

  // ================================================================ mini maps + charts
  function drawMini(id, i) {
    const c = $(id); if (!c) return; const g = c.getContext("2d"), w = c.width, h = c.height;
    g.fillStyle = "#040a12"; g.fillRect(0, 0, w, h);
    const cs = Math.min(w / W, h / H), ox = (w - W * cs) / 2, oy = (h - H * cs) / 2;
    g.imageSmoothingEnabled = true; g.globalAlpha = 0.9; g.drawImage(terrainImg, ox, oy, W * cs, H * cs);
    const keep = heatFor; const hm = buildHeat(i); g.globalAlpha = 1; g.drawImage(hm, ox, oy, W * cs, H * cs); g.filter = "blur(2px)"; g.globalCompositeOperation = "screen"; g.drawImage(hm, ox, oy, W * cs, H * cs); g.filter = "none"; g.globalCompositeOperation = "source-over";
    if (keep !== i) heatFor = -1;
    const p = E[i].position; g.fillStyle = C.cyan; g.beginPath(); g.arc(ox + (p[0] + 0.5) * cs, oy + (p[1] + 0.5) * cs, Math.max(2.5, cs * 0.4), 0, Math.PI * 2); g.fill();
  }
  function drawConfChart() {
    const c = $("confChart"); if (!c) return;
    const r = c.getBoundingClientRect(), dpr = window.devicePixelRatio || 1; c.width = r.width * dpr; c.height = r.height * dpr;
    const g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const L = 26, B = 16, w = r.width - L - 6, h = r.height - B - 6, N = Math.max(1, E.length - 1);
    const X = (i) => L + (i / N) * w, Y = (v) => 4 + (1 - v) * h;
    g.font = "10.5px JetBrains Mono, monospace"; g.fillStyle = C.faint; g.strokeStyle = "rgba(120,180,220,0.12)"; g.textAlign = "right"; g.textBaseline = "middle";
    for (const v of [0, 0.5, 1]) { g.beginPath(); g.moveTo(L, Y(v)); g.lineTo(L + w, Y(v)); g.stroke(); g.fillText(v.toFixed(1), L - 4, Y(v)); }
    g.textAlign = "center"; g.textBaseline = "top"; g.fillText("0", X(0), 6 + h); g.fillText(String(E.length - 1), X(N), 6 + h);
    const line = (to, col, a, wd) => { g.strokeStyle = col; g.globalAlpha = a; g.lineWidth = wd; g.beginPath(); for (let i = 0; i <= to; i++) i ? g.lineTo(X(i), Y(D.conf[i])) : g.moveTo(X(i), Y(D.conf[i])); g.stroke(); g.globalAlpha = 1; };
    line(E.length - 1, C.dim, 0.25, 1.2); line(cur, C.cyan, 1, 1.8);
    g.fillStyle = C.cyan; g.beginPath(); g.arc(X(cur), Y(D.conf[cur]), 3, 0, Math.PI * 2); g.fill();
  }
  function drawSparks() {
    document.querySelectorAll("canvas.spark").forEach((c) => {
      const s = c.dataset.src, r = c.getBoundingClientRect(), dpr = window.devicePixelRatio || 1; c.width = r.width * dpr; c.height = r.height * dpr;
      const g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0);
      const N = Math.max(1, E.length - 1), X = (i) => 2 + (i / N) * (r.width - 4), Y = (v) => r.height - 3 - v * (r.height - 6);
      g.strokeStyle = "rgba(147,168,189,0.25)"; g.setLineDash([2, 3]); g.beginPath(); g.moveTo(0, Y(0.5)); g.lineTo(r.width, Y(0.5)); g.stroke(); g.setLineDash([]);
      const ln = (to, col, a) => { g.strokeStyle = col; g.globalAlpha = a; g.lineWidth = 1.6; g.beginPath(); for (let i = 0; i <= to; i++) { const v = E[i].belief.dna[s].reliability; i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v)); } g.stroke(); g.globalAlpha = 1; };
      ln(E.length - 1, C.dim, 0.3); ln(cur, labelStyle(E[cur].belief.dna[s].label), 1);
    });
  }

  // ================================================================ inspector
  function renderInspector() {
    const el = $("inspector"); if (!selected || !M) { el.hidden = true; return; }
    const [x, y] = selected, k = y * W + x, ev = E[cur], T = M.public.terrain[y][x], p = ev.belief.location[k];
    const rank = ev.belief.top_cells.findIndex((t) => t.cell[0] === x && t.cell[1] === y);
    const exp = D.exploredAt[k], scans = D.scans.filter((s) => s.step <= cur && Math.hypot(s.center[0] - x, s.center[1] - y) <= s.radius + 1e-9).map((s) => s.step);
    const clue = D.clues.find((c) => c.position[0] === x && c.position[1] === y && c.step <= cur);
    const from = D.readings.filter((r) => r.origin[0] === x && r.origin[1] === y && r.step <= cur);
    const isBase = x === M.public.base[0] && y === M.public.base[1];
    el.innerHTML = `<h4><span>Cell ${cell(selected)}</span><button id="inspClose" aria-label="Close">✕</button></h4>
      <div class="kv"><span class="k">Terrain</span><span class="v">${isBase ? "base / charger" : ["plain ground", "rough (2.5× battery)", "rock (blocked)"][T]}</span>
      <span class="k">Chance EVE is here</span><span class="v num">${T === 2 ? "0" : (p * 100).toFixed(p < 0.001 ? 3 : 2) + "%"}${rank >= 0 ? ` · #${rank + 1}` : ""}</span>
      <span class="k">Explored</span><span class="v">${exp >= 0 && exp <= cur ? "since step " + exp : "not yet"}</span>
      <span class="k">Scanned</span><span class="v num">${scans.length ? "steps " + scans.join(", ") : "never"}</span></div>
      ${clue ? `<ul><li>${clue.type === "track" ? `Track clue pointing ${clue.bearing_deg}° (step ${clue.step})` : `Residue clue: ${clue.freshness}, EVE ${clue.proximity} (step ${clue.step})`}</li></ul>` : ""}
      ${from.length ? `<ul>${from.slice(-4).map((r) => `<li>#${r.step} ${esc(r.source)} ${r.bearing_deg != null ? r.bearing_deg + "°" : "silent"}</li>`).join("")}</ul>` : ""}`;
    el.hidden = false;
    const g = geom(), sx = g.ox + (x + 1.2) * g.cs, sy = g.oy + y * g.cs;
    el.style.left = clamp(sx, 8, g.w - 280) + "px"; el.style.top = clamp(sy, 8, g.h - el.offsetHeight - 8) + "px";
    $("inspClose").onclick = () => { selected = null; el.hidden = true; redraw(); };
  }

  // ================================================================ views
  function renderView() {
    if (!M) return;
    const consoleViews = ["mission", "investigation", "replay"];
    $("console").hidden = !consoleViews.includes(view);
    const wasCine = $("shell").classList.contains("cinema");
    $("shell").classList.toggle("cinema", view === "replay");
    $("cineCtl").hidden = view !== "replay";
    if (wasCine !== (view === "replay")) { if (!cine.follow) { cam.zoom = 1; cam.panX = 0; cam.panY = 0; } }
    syncCineCtl();
    $("page").hidden = consoleViews.includes(view);
    document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", b.dataset.view === view));
    if (view === "mission") {
      $("colLeft").innerHTML = pAgent() + pMissionStatus() + pSources() + pLog();
      $("colRight").innerHTML = pLatest() + pDecision(false) + pSignal() + pPhase();
      $("evoPanel").hidden = true; $("mapTitle").textContent = "Mission map";
    } else if (view === "investigation") {
      $("colLeft").innerHTML = pDNA() + pSources();
      $("colRight").innerHTML = pDecision(true) + pEvidence() + pSignal();
      $("evoPanel").hidden = false; $("evoPanel").innerHTML = evoPanel(); $("mapTitle").textContent = "Search map";
    } else if (view === "replay") {
      $("colLeft").innerHTML = ""; $("colRight").innerHTML = "";
      $("evoPanel").hidden = true; $("mapTitle").textContent = "Mission replay";
    } else {
      $("page").innerHTML = { report: pageReport, trace: pageTrace, evaluation: pageEvaluation }[view]();
      wirePage(); return;
    }
    drawConfChart(); drawSparks(); drawMini("miniPrev", Math.max(0, cur - 1)); drawMini("miniCur", cur); drawMini("miniNow", cur);
    redraw();
  }

  // ---------------- report page
  function pageReport() {
    const last = E[E.length - 1], over = last.observation && last.observation.mission_over;
    if (!over) return `<div class="page-inner"><h1>Mission report</h1><section class="panel"><p class="empty">The report is generated when the mission ends. ${live.on ? "Press Start or Run to end." : ""}</p></section></div>`;
    const r = M.report, lc = r.location_confidence, ev = M.evaluation, res = r.resources;
    const est = r.estimated_location, [ow, oc] = r.outcome === "EVE located" ? ["EVE located", "green"] : r.end_reason === "stranded" ? ["Failed: stranded", "red"] : ["EVE not located", "amber"];
    // strongest observations and contradictory evidence, each traceable to a step
    const strong = [], contra = [];
    for (const d of r.evidence.eve_detections) strong.push([d.step, `Local scan detected EVE at ${cell(d.cell)}; status reading ${d.status_reading}.`]);
    for (const c of r.evidence.clues_found) {
      if (c.type === "track") {
        const want = bearingTo(c.position, est), err = Math.abs(((c.bearing_deg - want + 540) % 360) - 180);
        (err <= 30 ? strong : contra).push([c.found_at_step, `Track clue at ${cell(c.position)} points ${c.bearing_deg}°; the estimate lies at ${want.toFixed(0)}° (${err.toFixed(0)}° off).`]);
      } else {
        const near = Math.hypot(c.position[0] - est[0], c.position[1] - est[1]) <= M.public.clue_specs.residue_near_radius;
        const fits = (c.proximity === "near") === near && (c.freshness === "fresh") === (r.inferred_status === "active");
        (fits ? strong : contra).push([c.found_at_step, `Residue clue at ${cell(c.position)}: ${c.freshness}, says EVE is ${c.proximity}${fits ? "" : " (does not match the final estimate; residue is 85% accurate)"}.`]);
      }
    }
    for (const [s, t] of Object.entries(r.signal_trust || {})) if (t.label === "conflicting" || t.label === "doubtful")
      contra.push([null, `Signal ${s} rated ${t.label} (reliability ${t.reliability.toFixed(2)}); its bearings were down-weighted.`]);
    const li = ([st, t]) => `<li>${st != null ? `<button class="steplink" data-jump="${st}">step ${st}</button> · ` : ""}${esc(t)}</li>`;
    return `<div class="page-inner">
      <div style="display:flex;justify-content:space-between;align-items:flex-end;gap:16px;flex-wrap:wrap"><div><h1>Mission report</h1><p class="lede">${esc(r.mission)} · agent ${esc(r.agent)} · seed ${r.seed} · ended: ${esc(String(r.end_reason).replace("_", " "))}. Built only from the mission trace.</p></div><span class="pill ${oc}" style="font-size:15px;padding:6px 16px">${ow}</span></div>
      <div class="grid-4">
        <section class="panel stat-box"><div class="k">Inferred location</div><div class="v num amber">${cell(est)}</div><div class="s">${lc.confirmed_by_local_scan ? "confirmed by WALL-E's own local scan" : "candidate region from the belief"}</div></section>
        <section class="panel stat-box"><div class="k">Location confidence</div><div class="v num">${pct(lc.p_within_scan_radius)}</div><div class="s">within scan range · ${pct(lc.p_estimated_cell, 1)} the exact cell</div></section>
        <section class="panel stat-box"><div class="k">Inferred status</div><div class="v">${esc(r.inferred_status)}</div><div class="s">${pct(r.status_confidence)} confidence from scans, beacon and clues</div></section>
        <section class="panel stat-box"><div class="k">Resources</div><div class="v num">${res.steps} <span style="font-size:16px" class="dim">actions</span></div><div class="s">battery spent ${res.battery_spent}, recharged ${res.battery_recharged}, left ${res.battery_left} · ${res.ticks_used}/${res.time_budget} ticks</div></section>
      </div>
      <section class="panel"><div class="panel-title"><span>Recommended next step</span><span class="small dim" style="letter-spacing:0;text-transform:none;font-family:var(--body)">A recommendation, not a verified fact</span></div><p style="margin:0;font-size:17px">${esc(r.recommended_next_step)}</p></section>
      <div class="grid-2">
        ${panel("Strongest observations", strong.length ? `<ul class="ev-list">${strong.map(li).join("")}</ul>` : `<p class="empty">No direct observations of EVE.</p>`)}
        ${panel("Contradictory evidence", contra.length ? `<ul class="ev-list">${contra.map(li).join("")}</ul>` : `<p class="empty">None found.</p>`)}
      </div>
      <div class="grid-2">
        ${panel("Signal trust at the end", Object.keys(r.signal_trust || {}).length ? `<table class="dt"><thead><tr><th>Source</th><th class="n">Reliability</th><th class="n">90% range</th><th class="n">Judged</th><th>Label</th></tr></thead><tbody>${Object.entries(r.signal_trust).map(([s, t]) => `<tr><td>${esc(s)}</td><td class="n">${t.reliability.toFixed(2)}</td><td class="n">${t.interval_90[0].toFixed(2)}–${t.interval_90[1].toFixed(2)}</td><td class="n">${t.judged}</td><td class="${labelClass(t.label)}">${esc(t.label)}</td></tr>`).join("")}</tbody></table>` : `<p class="empty">This agent does not learn signal trust.</p>`)}
        ${panel("Remaining uncertainty", `<div class="kv"><span class="k">Belief entropy</span><span class="v num">${r.remaining_uncertainty.entropy_bits.toFixed(2)} bits</span>${r.remaining_uncertainty.top_alternatives.map((a, k) => `<span class="k">Alternative ${k + 1}</span><span class="v num">${cell(a.cell)} · ${pct(a.p, 1)}</span>`).join("")}<span class="k">Evidence gathered</span><span class="v num">${r.evidence.long_scans} long, ${r.evidence.local_scans} local scans</span></div>`)}
      </div>
      ${panel("Export", `<div class="row-actions"><button class="btn primary" id="dlReport">Download report.json</button><button class="btn" id="dlTrace">Download trace.jsonl</button><button class="btn" data-goview="replay">Open replay</button></div><p class="small dim" style="margin:10px 0 0">The command line also saves <span class="num">report.md</span>, <span class="num">report.json</span> and <span class="num">trace.jsonl</span> in <span class="num">${live.savedTo ? esc(live.savedTo) : "outputs/&lt;config&gt;_&lt;agent&gt;/"}</span>.</p>`)}
      ${ev ? `<details class="truth" ${showTruth ? "open" : ""}><summary>Evaluation only: compare with ground truth (never available to the agent)</summary>
        <div class="kv" style="margin-top:10px;max-width:640px"><span class="k">True EVE position</span><span class="v num">${cell(ev.eve_position)}</span><span class="k">True status</span><span class="v">${esc(ev.eve_status)}</span>
        <span class="k">Estimate error</span><span class="v num">${ev.estimate_error_cells} cells</span><span class="k">Belief on the true cell</span><span class="v num">${pct(ev.metrics.p_true_cell, 1)}</span>
        <span class="k">True source reliability</span><span class="v num">${Object.entries(ev.source_true_reliability).map(([k, v]) => `${esc(k)} ${v}`).join(" · ")}</span></div>
        <label style="display:flex;gap:8px;align-items:center;margin-top:12px"><input type="checkbox" id="truthToggle" ${showTruth ? "checked" : ""}> Show ground truth on the map (red star, labelled evaluation only)</label></details>` : ""}
    </div>`;
  }
  function bearingTo(a, b) { return (Math.atan2(b[0] - a[0], -(b[1] - a[1])) * 180 / Math.PI + 360) % 360; }

  // ---------------- trace explorer
  const TF = [["all", "All"], ["move", "Movement"], ["long_scan", "Long scans"], ["local_scan", "Local scans"], ["recharge", "Recharge / return"], ["rule", "Safety decisions"], ["fail", "Warnings"]];
  function traceMatch(ev) {
    const o = ev.observation; if (!o) return traceFilter === "all";
    if (traceFilter === "all") return true;
    if (traceFilter === "fail") return !o.ok || o.end_reason === "stranded";
    if (traceFilter === "rule") return !!ev.decision.forced_by;
    if (traceFilter === "recharge") return o.action.type === "recharge" || ev.decision.chosen_id === "return_base";
    return o.action.type === traceFilter;
  }
  function pageTrace() {
    const rows = E.map((ev, i) => [ev, i]).filter(([ev]) => traceMatch(ev)).map(([ev, i]) => {
      const o = ev.observation, u = ev.belief_update, open = openRows.has(i);
      const main = `<tr class="row ${open ? "open" : ""}" data-row="${i}"><td class="num">${i}</td><td>${esc(actionLabel(ev))}</td><td>${esc(o ? observationText(ev) : "Start state")}</td>
        <td class="num">${u ? `${u.entropy_before.toFixed(2)}→${u.entropy_after.toFixed(2)}` : ev.belief.entropy_bits.toFixed(2)}</td>
        <td class="num">${ev.battery.toFixed(1)}${o ? ` <span class="${o.battery_delta < 0 ? "dim" : "green"}">(${o.battery_delta >= 0 ? "+" : ""}${o.battery_delta.toFixed(1)})</span>` : ""}</td>
        <td>${ev.decision ? esc(ev.decision.chosen_label) + (ev.decision.forced_by ? ` <span class="amber">· ${esc(ev.decision.forced_by)}</span>` : "") : "–"}</td></tr>`;
      if (!open) return main;
      const raw = JSON.parse(JSON.stringify(ev)); raw.belief.location = `[${raw.belief.location.length} probabilities, one per cell]`;
      return main + `<tr class="detail"><td></td><td colspan="5"><p style="margin:0 0 6px">${ev.decision ? esc(ev.decision.plain_english) + " " + esc(ev.decision.reason) : "Initial belief from the initial signal."}</p>
        <button class="btn" data-jump="${i}">Show on map</button><pre class="raw">${esc(JSON.stringify(raw, null, 2))}</pre></td></tr>`;
    }).join("");
    return `<div class="page-inner"><div><h1>Trace explorer</h1><p class="lede">Every logged event of this run: action, observation, belief change, battery and decision. Click a row for the explanation and raw JSON.</p></div>
      <section class="panel"><div class="chips">${TF.map(([k, l]) => `<button class="chip" data-tf="${k}" aria-pressed="${traceFilter === k}">${l}</button>`).join("")}<span style="flex:1"></span><button class="btn" id="dlTrace">Download trace.jsonl</button></div>
      <div style="max-height:calc(100vh - 290px);overflow:auto"><table class="trace"><thead><tr><th>Step</th><th>Action</th><th>Observation</th><th>Uncertainty (bits)</th><th>Battery (change)</th><th>Decision</th></tr></thead><tbody>${rows}</tbody></table></div>
      <p class="small faint" style="margin:10px 0 0">Schema: phantom-trace/1. One JSON line per event in <span class="num">trace.jsonl</span>, plus header, report and evaluation lines.</p></section></div>`;
  }

  // ---------------- evaluation
  let benchCache = window.PHANTOM_BENCH || null;
  function pageEvaluation() {
    const B = benchCache;
    if (!B) return `<div class="page-inner"><h1>Evaluation</h1><section class="panel"><p class="empty">Not run yet, so no numbers are shown. Run <span class="num">python -m phantom bench --seeds 20</span> and reload.</p></section></div>`;
    const agents = B.agents, scen = Object.keys(B.scenarios);
    const AC = { phantom: C.cyan, phantom_no_dna: "#7aa7ff", sweep: C.amber, random: C.dim };
    // grouped bar chart with 95% ranges
    const cw = 920, ch = 260, L = 46, Bm = 34, gw = (cw - L - 10) / scen.length, bw = Math.min(26, (gw - 24) / agents.length);
    const Yv = (v) => 10 + (1 - v) * (ch - Bm - 10);
    let svg = `<svg viewBox="0 0 ${cw} ${ch}" style="width:100%;height:auto" role="img" aria-label="Success rate by scenario and agent">`;
    for (const v of [0, 0.25, 0.5, 0.75, 1]) svg += `<line x1="${L}" x2="${cw - 10}" y1="${Yv(v)}" y2="${Yv(v)}" stroke="rgba(120,180,220,0.12)"/><text x="${L - 8}" y="${Yv(v) + 4}" fill="${C.faint}" font-size="11" text-anchor="end" font-family="JetBrains Mono">${v * 100}%</text>`;
    scen.forEach((s, si) => {
      const x0 = L + si * gw + (gw - bw * agents.length - 6 * (agents.length - 1)) / 2;
      agents.forEach((a, ai) => {
        const row = B.summary.find((r) => r.scenario === s && r.agent === a); if (!row) return;
        const x = x0 + ai * (bw + 6), y = Yv(row.success_rate);
        svg += `<rect x="${x}" y="${y}" width="${bw}" height="${Yv(0) - y}" rx="3" fill="${AC[a]}" opacity="0.85"><title>${a} on ${s}: ${pct(row.success_rate)} (95% ${pct(row.success_ci95[0])}–${pct(row.success_ci95[1])})</title></rect>`;
        svg += `<line x1="${x + bw / 2}" x2="${x + bw / 2}" y1="${Yv(row.success_ci95[0])}" y2="${Yv(row.success_ci95[1])}" stroke="${C.text}" stroke-width="1.2" opacity="0.7"/>`;
      });
      svg += `<text x="${L + si * gw + gw / 2}" y="${ch - 12}" fill="${C.dim}" font-size="13" text-anchor="middle" font-family="Barlow">${s.replace("_", " ")}</text>`;
    });
    svg += `</svg>`;
    const rows = B.summary.map((s) => `<tr class="${s.agent === "phantom" ? "phantom" : ""}"><td>${esc(s.scenario.replace("_", " "))}</td><td>${esc(s.agent)}</td><td>${pct(s.success_rate)}</td><td class="faint">${pct(s.success_ci95[0])}–${pct(s.success_ci95[1])}</td><td>${s.mean_steps_when_found ?? "–"}</td><td>${s.mean_battery_spent}</td><td>${s.stranded}</td><td>${s.median_error_cells}</td><td>${s.mean_p_true_cell.toFixed(2)}</td><td>${pct(s.status_correct_rate)}</td></tr>`).join("");
    return `<div class="page-inner"><div><h1>Evaluation</h1><p class="lede">${B.seeds_per_scenario} randomly generated missions per scenario (seeds 1000–${999 + B.seeds_per_scenario}), ${scen.length} scenarios, ${agents.length} agents, all on the same missions with the same safety rules: ${B.seeds_per_scenario * scen.length * agents.length} runs in total. Ground truth is used only after each run, to score it.</p></div>
      ${panel("Success rate by scenario", `<div class="chips">${agents.map((a) => `<span class="chip" style="cursor:default;color:${AC[a]};border-color:${AC[a]}">${a}</span>`).join("")}<span class="small dim" style="align-self:center">Whiskers = 95% range (Wilson)</span></div>${svg}`)}
      ${panel("All metrics", `<div style="overflow-x:auto"><table class="bench"><thead><tr><th>Scenario</th><th>Agent</th><th>Success</th><th>95% range</th><th>Steps when found</th><th>Battery spent</th><th>Stranded</th><th>Median error (cells)</th><th>P(true cell)</th><th>Status right</th></tr></thead><tbody>${rows}</tbody></table></div>`)}
      <div class="grid-2">${panel("Metric definitions", `<div class="kv small"><span class="k">Success</span><span class="v" style="text-align:left">a local scan confirmed EVE at her true cell</span><span class="k">Steps when found</span><span class="v" style="text-align:left">average actions, successful runs only</span><span class="k">Stranded</span><span class="v" style="text-align:left">ended away from base below one move of battery</span><span class="k">Median error</span><span class="v" style="text-align:left">distance from final estimate to true cell</span><span class="k">P(true cell)</span><span class="v" style="text-align:left">final belief on the true cell (evaluation only)</span><span class="k">Status right</span><span class="v" style="text-align:left">inferred active/dormant matches truth</span></div>`)}
      ${panel("What this does and does not show", `<p class="small" style="margin-top:0">PHANTOM succeeded far more often than the sweep and random baselines and no agent was stranded. Signal DNA made a clear difference only in the decoy scenario; elsewhere the full agent and the no-DNA version are within noise.</p><p class="small dim" style="margin:0">With ${B.seeds_per_scenario} runs per row the ranges are wide, so small differences mean nothing. All scenarios use this simulator's sensor model and 24×18 maps; other configs may behave differently. Clean, noisy and decoy share maps per seed, so the baselines that ignore long-range signals repeat across them.</p>`)}</div>
      ${panel("Scenario settings", `<table class="bench"><thead><tr><th>Scenario</th><th></th><th>Relay reliability</th><th>Relay noise</th><th>Beacon noise</th><th>Decoy source</th><th>Battery</th></tr></thead><tbody>${scen.map((s) => { const p = B.scenarios[s]; return `<tr><td>${esc(s.replace("_", " "))}</td><td></td><td>${p.relay_rel}</td><td>±${p.relay_noise}°</td><td>±${p.beacon_noise}°</td><td>${p.decoy ? "yes" : "no"}</td><td>${p.battery}</td></tr>`; }).join("")}</tbody></table>`)}
    </div>`;
  }

  function wirePage() {
    document.querySelectorAll("[data-jump]").forEach((b) => b.onclick = () => { view = "replay"; stopPlay(); setStep(Number(b.dataset.jump)); });
    document.querySelectorAll("[data-goview]").forEach((b) => b.onclick = () => { view = b.dataset.goview; renderView(); });
    document.querySelectorAll("[data-tf]").forEach((b) => b.onclick = () => { traceFilter = b.dataset.tf; renderView(); });
    document.querySelectorAll("tr.row").forEach((r) => r.onclick = () => { const i = Number(r.dataset.row); openRows.has(i) ? openRows.delete(i) : openRows.add(i); renderView(); });
    const t = $("truthToggle"); if (t) t.onchange = () => { showTruth = t.checked; redraw(); toast(showTruth ? "Ground truth is now shown on the map as a red star (evaluation view)." : "Ground truth hidden."); };
    const dr = $("dlReport"); if (dr) dr.onclick = () => download("report.json", JSON.stringify(M.report, null, 2));
    const dt = $("dlTrace"); if (dt) dt.onclick = () => download("trace.jsonl", traceJSONL());
  }
  function traceJSONL() {
    const lines = [JSON.stringify({ type: "header", format: M.format, meta: M.meta, public: M.public }), ...E.map((e) => JSON.stringify(e))];
    if (M.report) lines.push(JSON.stringify({ type: "report", report: M.report }));
    if (M.evaluation) lines.push(JSON.stringify({ type: "evaluation", evaluation: M.evaluation }));
    return lines.join("\n") + "\n";
  }
  function download(name, text) { const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([text], { type: "application/octet-stream" })); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000); }

  // ================================================================ legend + jump list
  function buildLegend() {
    $("vramp").style.background = rampCSS("0deg");
    const k = [
      [`<svg viewBox="0 0 26 18"><circle cx="13" cy="9" r="7" fill="#04121e" stroke="${C.cyan}" stroke-width="1.8"/><rect x="10" y="6" width="6" height="6" fill="${C.cyan}"/></svg>`, "WALL-E (agent)"],
      [`<svg viewBox="0 0 26 18"><line x1="2" y1="9" x2="24" y2="9" stroke="${C.cyan}" stroke-width="2.2" stroke-dasharray="5 4"/></svg>`, "Route history"],
      [`<svg viewBox="0 0 26 18"><circle cx="13" cy="9" r="7" fill="#04121e" stroke="${C.cyan}" stroke-width="1.5"/><path d="M13 5l4 4h-1.5v4h-5v-4H9z" fill="${C.cyan}"/></svg>`, "Base / charger"],
      [`<svg viewBox="0 0 26 18"><rect x="3" y="3" width="20" height="12" fill="rgba(40,130,170,0.4)" stroke="rgba(56,225,240,0.5)"/></svg>`, "Explored area"],
      [`<svg viewBox="0 0 26 18"><rect x="3" y="3" width="20" height="12" fill="#0a1622" stroke="#24384a"/></svg>`, "Unexplored area"],
      [`<svg viewBox="0 0 26 18"><rect x="3" y="3" width="20" height="12" fill="#5f6f82"/><path d="M3 4h20" stroke="#b9c8d6"/></svg>`, "Rock (blocked)"],
      [`<svg viewBox="0 0 26 18"><rect x="3" y="3" width="20" height="12" fill="#22323f"/><circle cx="9" cy="8" r="1" fill="#9ab"/><circle cx="16" cy="11" r="1" fill="#9ab"/></svg>`, "Rough ground"],
      [`<svg viewBox="0 0 26 18"><path d="M13 3l6 6-6 6-6-6z" fill="#04121e" stroke="${C.cyan}" stroke-width="1.6"/></svg>`, "Clue found"],
      [`<svg viewBox="0 0 26 18"><circle cx="13" cy="9" r="7" fill="none" stroke="rgba(56,225,240,0.6)" stroke-dasharray="2 3"/></svg>`, "Local scan area"],
      [`<svg viewBox="0 0 26 18"><line x1="2" y1="9" x2="24" y2="9" stroke="${C.cyan}" stroke-width="2"/></svg>`, "Signal: trusted"],
      [`<svg viewBox="0 0 26 18"><line x1="2" y1="9" x2="24" y2="9" stroke="${C.amber}" stroke-width="2"/></svg>`, "Signal: doubtful"],
      [`<svg viewBox="0 0 26 18"><line x1="2" y1="9" x2="24" y2="9" stroke="${C.red}" stroke-width="2"/></svg>`, "Signal: conflicting"],
      [`<svg viewBox="0 0 26 18"><path d="M13 2l7 7-7 7-7-7z" fill="#140800" stroke="${C.amber}" stroke-width="1.6"/></svg>`, "Most likely EVE cell"],
    ];
    $("legendKeys").innerHTML = k.map(([s, t]) => `<li>${s}<span>${t}</span></li>`).join("") +
      D.sources.map((s, i) => `<li><svg viewBox="0 0 26 18"><line x1="2" y1="9" x2="24" y2="9" stroke="#9fd6e6" stroke-width="2" stroke-dasharray="${DASH[i % 3].join(" ")}"/></svg><span>${esc(s)} bearings</span></li>`).join("");
  }
  function buildJump() {
    const opts = D.keyEvents.filter((k) => ["detect", "clue", "safety", "label", "fail", "end"].includes(k.type));
    $("jumpEvent").innerHTML = `<option value="">Jump to event</option>` + opts.map((k) => `<option value="${k.step}">#${k.step} ${esc(k.text)}</option>`).join("");
  }

  // ================================================================ transport (replay or live)
  const SVG = { first: `<svg viewBox="0 0 24 24"><path d="M6 5v14M18 5l-9 7 9 7z" fill="currentColor"/></svg>`, prev: `<svg viewBox="0 0 24 24"><path d="M16 5l-9 7 9 7z" fill="currentColor"/></svg>`,
    play: `<svg viewBox="0 0 24 24"><path d="M7 4l13 8-13 8z" fill="currentColor"/></svg>`, pause: `<svg viewBox="0 0 24 24"><path d="M6 4h4v16H6zM14 4h4v16h-4z" fill="currentColor"/></svg>`,
    next: `<svg viewBox="0 0 24 24"><path d="M8 5l9 7-9 7z" fill="currentColor"/></svg>`, last: `<svg viewBox="0 0 24 24"><path d="M18 5v14M6 5l9 7-9 7z" fill="currentColor"/></svg>`,
    reset: `<svg viewBox="0 0 24 24"><path d="M4 12a8 8 0 1 0 3-6.2M4 4v5h5" fill="none" stroke="currentColor" stroke-width="2"/></svg>` };
  function renderTransport() {
    const t = $("transport");
    if (live.on) {
      t.innerHTML = `<button class="tbtn label" id="tReset" title="Reset: new mission with the same settings">${SVG.reset}Reset</button>
        <button class="tbtn label" id="tStep" title="Run exactly one action">${SVG.next}Step</button>
        <button class="tbtn primary" id="tPlay" aria-label="${live.running ? "Pause" : "Start"}" title="${live.running ? "Pause" : "Start"}">${live.running ? SVG.pause : SVG.play}</button>
        <button class="tbtn label" id="tEnd" title="Run to the end">${SVG.last}Run to end</button>
        <select id="speed" aria-label="Speed">${[[3, "Slow"], [8, "Normal"], [20, "Fast"]].map(([v, l]) => `<option value="${v}" ${v === speed.live ? "selected" : ""}>${l}</option>`).join("")}</select>`;
      $("tReset").onclick = () => startSession(live.req);
      $("tStep").onclick = () => { liveStop(); liveStep(1); };
      $("tPlay").onclick = () => (live.running ? liveStop() : liveStart());
      $("tEnd").onclick = () => { liveStop(); liveFinish(); };
      const done = live.done; ["tStep", "tPlay", "tEnd"].forEach((id) => { $(id).disabled = done; });
      $("modeNote").textContent = done ? "Mission finished · drag the timeline to review" : "Live simulation · controls drive the real agent";
      $("replayTitle").textContent = "Mission control";
    } else {
      t.innerHTML = `<button class="tbtn" id="tFirst" aria-label="Jump to start" title="Jump to start (Home)">${SVG.first}</button>
        <button class="tbtn" id="tPrev" aria-label="Previous action" title="Previous action (←)">${SVG.prev}</button>
        <button class="tbtn primary" id="tPlay" aria-label="${playing ? "Pause" : "Play"}" title="Play / pause (space)">${playing ? SVG.pause : SVG.play}</button>
        <button class="tbtn" id="tNext" aria-label="Next action" title="Next action (→)">${SVG.next}</button>
        <button class="tbtn" id="tLast" aria-label="Jump to end" title="Jump to end (End)">${SVG.last}</button>
        <select id="speed" aria-label="Speed">${[[2, "0.5×"], [5, "1×"], [12, "2×"], [25, "4×"]].map(([v, l]) => `<option value="${v}" ${v === speed.replay ? "selected" : ""}>${l}</option>`).join("")}</select>`;
      $("tFirst").onclick = () => { stopPlay(); setStep(0); };
      $("tPrev").onclick = () => { stopPlay(); setStep(cur - 1); };
      $("tNext").onclick = () => { stopPlay(); setStep(cur + 1); };
      $("tLast").onclick = () => { stopPlay(); setStep(E.length - 1); };
      $("tPlay").onclick = () => (playing ? stopPlay() : startPlay());
      $("modeNote").textContent = "Replaying a saved run · run python -m phantom serve for live control";
      $("replayTitle").textContent = "Mission replay";
    }
    syncCineCtl();
    $("speed").onchange = () => {
      speed[live.on ? "live" : "replay"] = Number($("speed").value);
      if (playing) { stopPlay(); startPlay(); } if (live.running) { liveStop(); liveStart(); }
    };
  }
  function syncCineCtl() {
    if (!$("cineCtl")) return;
    $("cPlay").innerHTML = playing ? SVG.pause : SVG.play; $("cPlay").setAttribute("aria-label", playing ? "Pause" : "Play");
    $("cSpeed").value = String(speed.replay);
    $("cFollow").setAttribute("aria-pressed", String(cine.follow));
    $("cRec").classList.toggle("on", !!cine.rec); $("cRec").lastChild.textContent = cine.rec ? "Recording…" : "Record";
    ["cPrev", "cPlay", "cNext", "cSpeed", "cFollow"].forEach((id) => { $(id).disabled = !!cine.rec; });
  }
  function recordReplay() {
    if (cine.rec) return;
    if (!canvas.captureStream || !window.MediaRecorder) { toast("This browser cannot record video. Use a recent Chrome, Edge or Firefox."); return; }
    stopPlay(); liveStop();
    const types = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
    const mime = types.find((t) => MediaRecorder.isTypeSupported(t)) || "";
    const chunks = [], stream = canvas.captureStream(30);
    const rec = new MediaRecorder(stream, mime ? { mimeType: mime, videoBitsPerSecond: 8_000_000 } : {});
    rec.ondataavailable = (e) => { if (e.data && e.data.size) chunks.push(e.data); };
    rec.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(chunks, { type: "video/webm" }), a = document.createElement("a");
      a.href = URL.createObjectURL(blob); a.download = `phantom-replay-PH-${String(M.meta.seed).padStart(4, "0")}-${M.meta.agent}.webm`; a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 4000);
      cine.rec = null; syncCineCtl(); toast("Replay video saved to your downloads.");
    };
    cine.rec = rec; selected = null; $("inspector").hidden = true;
    setStep(0); rec.start(250); syncCineCtl();
    setTimeout(() => startPlay(true), 1200);          // hold the opening frame briefly
  }
  function startPlay(fromRec) {
    if (cur >= E.length - 1) setStep(0);
    playing = true; renderTransport();
    playTimer = setInterval(() => {
      if (cur >= E.length - 1) { stopPlay(); if (cine.rec) setTimeout(() => cine.rec && cine.rec.stop(), 2000); return; }
      setStep(cur + 1);
    }, 1000 / speed.replay);
  }
  function stopPlay() { if (!playing) return; playing = false; clearInterval(playTimer); renderTransport(); redraw(); }
  $("cPlay").onclick = () => { liveStop(); playing ? stopPlay() : startPlay(); };
  $("cPrev").onclick = () => { stopPlay(); liveStop(); setStep(cur - 1); };
  $("cNext").onclick = () => { stopPlay(); liveStop(); setStep(cur + 1); };
  $("cSpeed").onchange = () => { speed.replay = Number($("cSpeed").value); if (playing) { stopPlay(); startPlay(); } };
  $("cFollow").onclick = () => { cine.follow = !cine.follow; if (cine.follow) cam.zoom = 1.9; else { cam.zoom = 1; cam.panX = 0; cam.panY = 0; } $("zoomRead").textContent = `Zoom ${Math.round(cam.zoom * 100)}%`; syncCineCtl(); redraw(); };
  $("cRec").onclick = recordReplay;

  // ================================================================ live API
  async function api(path, body) {
    const res = await fetch(path, body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {});
    const data = await res.json().catch(() => ({ error: `server returned ${res.status}` }));
    if (!res.ok) throw new Error(data.error || `server returned ${res.status}`);
    return data;
  }
  async function startSession(req) {
    liveStop();
    try {
      const d = await api("/api/session", req);
      live.session = d.session; live.req = req; live.done = false; live.savedTo = null;
      M = null; setTrace(d.trace); return true;
    } catch (e) { toast(`Could not start the mission: ${e.message}`); return false; }
  }
  function appendEvents(d) {
    const follow = cur >= E.length - 1;
    E.push(...d.events);
    if (d.done) { M.report = d.report; M.evaluation = d.evaluation; live.done = true; live.savedTo = d.saved_to; liveStop(); renderTransport(); }
    derive(); buildJump(); heatFor = -1; maskFor = -1;
    setStep(follow ? E.length - 1 : cur);
  }
  async function liveStep(n) {
    if (live.busy || live.done) return; live.busy = true;
    try { appendEvents(await api("/api/step", { session: live.session, count: n })); }
    catch (e) { liveStop(); toast(`Simulation error: ${e.message}`); }
    finally { live.busy = false; }
  }
  async function liveFinish() {
    if (live.busy || live.done) return; live.busy = true;
    try { appendEvents(await api("/api/finish", { session: live.session })); }
    catch (e) { toast(`Simulation error: ${e.message}`); } finally { live.busy = false; }
  }
  function liveStart() { if (live.done) return; live.running = true; renderTransport(); updateStatus(); live.timer = setInterval(() => liveStep(1), 1000 / speed.live); }
  function liveStop() { if (live.timer) clearInterval(live.timer); live.timer = null; if (live.running) { live.running = false; renderTransport(); updateStatus(); } }

  // ================================================================ modals
  function openHelp() {
    $("helpBody").innerHTML = `<h2 id="helpTitle">How to read the console</h2>
      <p class="dim">WALL-E is searching for EVE with noisy signals and a limited battery. Everything on screen comes from the simulation trace.</p>
      <ol><li><b>The glow is WALL-E's belief</b>, not the answer: warmer and brighter cells are where he thinks EVE probably is. The amber diamond marks his single best guess.</li>
        <li><b>Lines are signal bearings.</b> Cyan = trusted, amber = doubtful, red = conflicting. Click a signal source on the left to inspect it.</li>
        <li><b>${live.on ? "Start, Step and Run to end" : "Play, step or drag the timeline"}</b> to ${live.on ? "run the real agent" : "replay the mission"}. Click any map cell for its details.</li>
        <li><b>Investigation</b> shows every option WALL-E scored, Signal DNA and how the belief changed. <b>Report</b> has the final conclusions.</li></ol>
      <h3>Status meanings</h3><div class="kv small" style="max-width:560px"><span class="k cyan">Ready / Running</span><span class="v" style="text-align:left">mission created / agent acting</span><span class="k amber">Paused</span><span class="v" style="text-align:left">stopped, press Start to continue</span><span class="k green">Complete</span><span class="v" style="text-align:left">EVE confirmed by WALL-E's own scan</span><span class="k amber">Ended, not located</span><span class="v" style="text-align:left">mission over without confirmation</span><span class="k red">Failed</span><span class="v" style="text-align:left">stranded away from base</span></div>
      <h3>Keyboard</h3><div class="shortcuts"><kbd>Space</kbd><span>play / pause${live.on ? " (start / pause)" : ""}</span><kbd>←</kbd> <kbd>→</kbd><span>previous / next step</span><kbd>+</kbd> <kbd>−</kbd><span>zoom</span><kbd>C</kbd><span>center on WALL-E</span><kbd>1</kbd>–<kbd>6</kbd><span>switch screens</span><kbd>Esc</kbd><span>close dialogs</span></div>
      <h3>Run information</h3><div class="small dim"><span class="num">python -m phantom run configs/decoy_mission.yaml</span> runs a mission and refreshes the replay data.<br><span class="num">python -m phantom serve</span> starts this console in live mode at <span class="num">http://127.0.0.1:8765</span>.<br>Traces: <span class="num">outputs/&lt;config&gt;_&lt;agent&gt;/trace.jsonl</span> (schema phantom-trace/1). Python 3.10+, numpy, PyYAML. Mode now: <b>${live.on ? "live" : "replay"}</b>.</div>
      <div class="row-actions" style="margin-top:18px"><button class="btn primary" id="helpClose">Start exploring</button></div>`;
    $("helpModal").hidden = false; $("helpClose").focus();
    $("helpClose").onclick = () => { $("helpModal").hidden = true; try { localStorage.setItem("phantom-guide-v2", "1"); } catch (e) { /* storage unavailable */ } };
  }
  function openConfig() {
    if (!live.on) {
      $("configBody").innerHTML = `<h2 id="configTitle">Mission configuration</h2>
        <p>You are viewing a <b>saved run</b>. To configure and launch missions from here, start the live console:</p>
        <div class="formula" style="font-size:14px">python -m phantom serve<br># then open http://127.0.0.1:8765</div>
        <p class="dim" style="margin-top:14px">Or load another saved run (any <span class="num">outputs/*/mission.json</span>):</p>
        <div class="row-actions"><button class="btn primary" id="cfgLoad">Load mission.json…</button><button class="btn" id="cfgClose">Close</button></div>`;
      $("configModal").hidden = false;
      $("cfgLoad").onclick = () => $("fileInput").click(); $("cfgClose").onclick = () => { $("configModal").hidden = true; };
      return;
    }
    const P = live.info.presets, lim = live.info.limits, rq = live.req || {};
    const curP = P.find((p) => p.id === (rq.preset || P[0].id)) || P[0];
    $("configBody").innerHTML = `<h2 id="configTitle">Mission configuration</h2><p class="dim">Settings go straight into the real environment config. The same preset, seed and agent always produce the same mission.</p>
      <form class="form" id="cfgForm" novalidate>
        <div class="field" style="grid-column:1/-1"><label for="fPreset">Map / environment preset</label><select id="fPreset">${P.map((p) => `<option value="${p.id}" ${p.id === curP.id ? "selected" : ""}>${esc(p.label)}</option>`).join("")}</select><span class="hint" id="presetHint"></span></div>
        <div class="field"><label for="fSeed">Seed</label><input id="fSeed" type="number" min="${lim.seed[0]}" max="${lim.seed[1]}" step="1"><span class="hint" id="hSeed"></span></div>
        <div class="field"><label for="fAgent">Agent</label><select id="fAgent">${live.info.agents.map((a) => `<option ${a === (rq.agent || "phantom") ? "selected" : ""}>${a}</option>`).join("")}</select><span class="hint">phantom = full agent; others are baselines</span></div>
        <div class="field"><label for="fBat">Battery capacity</label><input id="fBat" type="number" min="${lim.battery_capacity[0]}" max="${lim.battery_capacity[1]}" step="1"><span class="hint" id="hBat"></span></div>
        <div class="field"><label for="fTime">Time budget (ticks)</label><input id="fTime" type="number" min="${lim.time_budget[0]}" max="${lim.time_budget[1]}" step="1"><span class="hint" id="hTime"></span></div>
      </form><div class="form-error" id="cfgErr" role="alert"></div>
      <div class="row-actions"><button class="btn primary" id="cfgGo">Launch mission</button><button class="btn" id="cfgDefaults">Reset to defaults</button><button class="btn" id="cfgClose">Cancel</button></div>`;
    const fill = (keep) => {
      const p = P.find((x) => x.id === $("fPreset").value), d = p.defaults;
      $("presetHint").textContent = p.kind === "file" ? "Hand-made map from configs/" : "Map, EVE, clues and signals generated from the seed";
      $("hSeed").textContent = `default ${d.seed} · ${lim.seed[0]}–${lim.seed[1]}`; $("hBat").textContent = `default ${d.battery_capacity} · ${lim.battery_capacity[0]}–${lim.battery_capacity[1]}`; $("hTime").textContent = `default ${d.time_budget} · ${lim.time_budget[0]}–${lim.time_budget[1]}`;
      if (!keep) { $("fSeed").value = d.seed; $("fBat").value = d.battery_capacity; $("fTime").value = d.time_budget; }
    };
    fill(false);
    if (rq.preset === curP.id) { if (rq.seed != null) $("fSeed").value = rq.seed; if (rq.battery_capacity != null) $("fBat").value = rq.battery_capacity; if (rq.time_budget != null) $("fTime").value = rq.time_budget; }
    $("fPreset").onchange = () => fill(false); $("cfgDefaults").onclick = () => fill(false);
    $("cfgClose").onclick = () => { $("configModal").hidden = true; };
    $("cfgGo").onclick = async () => {
      const f = [["fSeed", "Seed"], ["fBat", "Battery capacity"], ["fTime", "Time budget"]];
      for (const [id, name] of f) { const el = $(id); if (el.value === "" || !el.checkValidity() || !Number.isInteger(Number(el.value))) { $("cfgErr").textContent = `${name} must be a whole number between ${el.min} and ${el.max}.`; el.focus(); return; } }
      $("cfgErr").textContent = "";
      const req = { preset: $("fPreset").value, agent: $("fAgent").value, seed: Number($("fSeed").value), battery_capacity: Number($("fBat").value), time_budget: Number($("fTime").value) };
      try { await api("/api/session", req).then((d) => { live.session = d.session; live.req = req; live.done = false; live.savedTo = null; M = null; setTrace(d.trace); }); $("configModal").hidden = true; toast("Mission ready. Press Start."); }
      catch (e) { $("cfgErr").textContent = e.message; }
    };
    $("configModal").hidden = false; $("fPreset").focus();
  }

  let toastTimer = null;
  function toast(msg) { let t = document.querySelector(".toast"); if (!t) { t = document.createElement("div"); t.className = "toast"; t.setAttribute("role", "status"); document.body.appendChild(t); } t.textContent = msg; clearTimeout(toastTimer); toastTimer = setTimeout(() => t.remove(), 4500); }

  // ================================================================ wiring
  function cellAt(px, py) { const g = geom(), x = Math.floor((px - g.ox) / g.cs), y = Math.floor((py - g.oy) / g.cs); return x >= 0 && y >= 0 && x < W && y < H ? [x, y] : null; }
  let drag = null;
  function bandStep(e) {
    const r = canvas.getBoundingClientRect(), b = cine.band, x = e.clientX - r.left, y = e.clientY - r.top;
    if (!isCine() || !b || cine.rec || y < b.y0 || y > b.y1 || x < b.x0 - 8 || x > b.x1 + 8) return null;
    return Math.round(clamp((x - b.x0) / (b.x1 - b.x0), 0, 1) * b.last);
  }
  canvas.addEventListener("pointerdown", (e) => { const sb = bandStep(e); if (sb !== null) { cine.scrub = true; stopPlay(); liveStop(); setStep(sb); canvas.setPointerCapture(e.pointerId); return; } });
  canvas.addEventListener("pointerdown", (e) => { if (cine.scrub) return; drag = { x: e.clientX, y: e.clientY, px: cam.panX, py: cam.panY, moved: false }; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointermove", (e) => {
    if (cine.scrub) { const r = canvas.getBoundingClientRect(), b = cine.band; const st = Math.round(clamp((e.clientX - r.left - b.x0) / (b.x1 - b.x0), 0, 1) * b.last); if (st !== cur) setStep(st); return; }
    if (!drag) return; const dx = e.clientX - drag.x, dy = e.clientY - drag.y; if (Math.abs(dx) + Math.abs(dy) > 4) { drag.moved = true; canvas.classList.add("dragging"); } if (drag.moved) { cam.panX = drag.px + dx; cam.panY = drag.py + dy; redraw(); if (selected) renderInspector(); } });
  canvas.addEventListener("pointerup", (e) => { if (cine.scrub) { cine.scrub = false; return; } canvas.classList.remove("dragging"); if (drag && !drag.moved && M) { const r = canvas.getBoundingClientRect(); selected = cellAt(e.clientX - r.left, e.clientY - r.top); renderInspector(); redraw(); } drag = null; });
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); const r = canvas.getBoundingClientRect(); zoomAt(e.deltaY < 0 ? 1.15 : 1 / 1.15, e.clientX - r.left, e.clientY - r.top); if (selected) renderInspector(); }, { passive: false });
  $("zoomIn").onclick = () => zoomAt(1.25); $("zoomOut").onclick = () => zoomAt(1 / 1.25);
  $("centerAgent").onclick = () => M && centerOn(E[cur].position); $("fitMap").onclick = fit;
  document.querySelectorAll("[data-layer]").forEach((cb) => cb.addEventListener("change", () => { layers[cb.dataset.layer] = cb.checked; redraw(); }));
  document.addEventListener("click", (e) => { const m = $("layersMenu"); if (m.open && !m.contains(e.target)) m.open = false; });
  $("scrubber").oninput = (e) => { stopPlay(); liveStop(); setStep(Number(e.target.value)); };
  $("jumpEvent").onchange = (e) => { if (e.target.value !== "") { stopPlay(); liveStop(); setStep(Number(e.target.value)); } e.target.value = ""; };
  document.querySelectorAll(".tabs button").forEach((b) => b.onclick = () => { view = b.dataset.view; renderView(); });
  // delegated clicks inside the side columns
  for (const id of ["colLeft", "colRight"]) $(id).addEventListener("click", (e) => {
    const src = e.target.closest("[data-src]"); if (src) { selectedSource = src.dataset.src; renderView(); return; }
    const st = e.target.closest("[data-step]"); if (st) { stopPlay(); liveStop(); setStep(Number(st.dataset.step)); return; }
    const f = e.target.closest("[data-filter]"); if (f) { const k = f.dataset.filter; eventFilter.has(k) ? eventFilter.delete(k) : eventFilter.add(k); renderView(); return; }
    const gv = e.target.closest("[data-goview]"); if (gv) { view = gv.dataset.goview; renderView(); }
  });
  $("btnHelp").onclick = openHelp; $("btnConfig").onclick = openConfig;
  document.querySelectorAll(".overlay").forEach((o) => o.addEventListener("click", (e) => { if (e.target === o) o.hidden = true; }));
  $("fileInput").onchange = (e) => {
    const f = e.target.files[0]; if (!f) return; const rd = new FileReader();
    rd.onload = () => { try { stopPlay(); liveStop(); live.on = false; M = null; setTrace(JSON.parse(rd.result)); $("configModal").hidden = true; } catch (err) { toast("Could not read that file: " + err.message); } };
    rd.readAsText(f); e.target.value = "";
  };
  document.addEventListener("keydown", (e) => {
    if (["INPUT", "SELECT", "TEXTAREA"].includes(e.target.tagName)) return;
    if (e.key === "Escape") { document.querySelectorAll(".overlay").forEach((o) => (o.hidden = true)); return; }
    if (!M || !$("helpModal").hidden || !$("configModal").hidden) return;
    const views = ["mission", "investigation", "replay", "report", "trace", "evaluation"];
    if (/^[1-6]$/.test(e.key)) { view = views[Number(e.key) - 1]; renderView(); return; }
    if (e.key === " ") { e.preventDefault(); if (live.on) (live.running ? liveStop() : liveStart()); else (playing ? stopPlay() : startPlay()); }
    if (e.key === "ArrowRight") { stopPlay(); if (live.on && cur === E.length - 1 && !live.done) liveStep(1); else setStep(cur + 1); }
    if (e.key === "ArrowLeft") { stopPlay(); liveStop(); setStep(cur - 1); }
    if (e.key === "Home") { stopPlay(); setStep(0); }
    if (e.key === "End") { stopPlay(); setStep(E.length - 1); }
    if (e.key === "+" || e.key === "=") zoomAt(1.2);
    if (e.key === "-") zoomAt(1 / 1.2);
    if (e.key === "c" || e.key === "C") centerOn(E[cur].position);
  });
  window.addEventListener("resize", () => { redraw(); drawStrip(); if (M && ["mission", "investigation", "replay"].includes(view)) { drawConfChart(); drawSparks(); } });

  // ================================================================ start
  async function init() {
    requestAnimationFrame(loop);
    let seen = false; try { seen = localStorage.getItem("phantom-guide-v2") === "1"; } catch (e) { /* ignore */ }
    if (location.protocol.startsWith("http")) {
      try {
        const info = await api("/api/info");
        live.on = true; live.info = info;
        api("/api/benchmark").then((b) => { benchCache = b; }).catch(() => { /* no benchmark yet */ });
        const preset = info.presets.find((p) => p.id === "file:decoy_mission") || info.presets[0];
        await startSession({ preset: preset.id, agent: "phantom" });
      } catch (e) { live.on = false; }
    }
    if (!live.on) {
      if (window.PHANTOM_MISSION) setTrace(window.PHANTOM_MISSION);
      else { $("sysStatus").lastElementChild.textContent = "No data"; toast("No mission data found. Run python -m phantom run configs/decoy_mission.yaml, or start the live console with python -m phantom serve."); }
    }
    if (!seen) openHelp();
  }
  init();

  // test hook for automated checks (read-only state + step control)
  window.__phantom = { get step() { return cur; }, get events() { return E.length; }, get event() { return E[cur]; }, setStep, get selected() { return selected; },
    get zoom() { return cam.zoom; }, get layers() { return { ...layers }; }, get view() { return view; }, get live() { return { on: live.on, running: live.running, done: live.done, session: live.session }; },
    get showTruth() { return showTruth; }, get cine() { return { follow: cine.follow, recording: !!cine.rec, band: cine.band }; } };
})();
