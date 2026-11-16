"""Single-ink, monospace "console" dashboard for the camera fleet.

Served at / on the laptop server and /fleet on both servers. Everything comes
from the fleet heartbeats (/api/fleet) plus the existing registry endpoints
(/api/cameras/...) for stream, capture and the motion region. Manual and calibration
captures and the motion-region (ROI) editor live here; the previous dashboard
stays at /legacy for the full config form.
"""

CONSOLE_DASHBOARD_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Bird Cameras</title>
<style>
:root { --bg: #fafaf8; --ink: #111111; --muted: #5c5c5c; --rule: #b5b5b5; --img: #1a1a1a; --thumb: #e4e4e0;
  --mono: "JetBrains Mono", "Cascadia Mono", ui-monospace, Consolas, "Liberation Mono", monospace; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) { --bg: #121212; --ink: #e8e8e4; --muted: #a0a09a; --rule: #4a4a46; --img: #000000; --thumb: #2a2a28; }
}
:root[data-theme="dark"] { --bg: #121212; --ink: #e8e8e4; --muted: #a0a09a; --rule: #4a4a46; --img: #000000; --thumb: #2a2a28; }
* { box-sizing: border-box; }
html, body { margin: 0; background: var(--bg); color: var(--ink); font: 13px/1.5 var(--mono); }
a { color: var(--ink); }
button { font: inherit; font-size: 12.5px; min-height: 36px; padding: 0 10px; border: 1px solid var(--ink); border-radius: 0;
  background: transparent; color: var(--ink); cursor: pointer; }
button:hover:not(:disabled) { background: var(--ink); color: var(--bg); }
button.primary { background: var(--ink); color: var(--bg); font-weight: 700; }
button.dashed { border-style: dashed; }
button:disabled { opacity: .45; cursor: not-allowed; background: transparent; color: var(--ink); font-weight: 400; }
.wrap { max-width: 1360px; margin: 0 auto; padding: 0 20px; }
header .bar { display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px 20px; padding: 14px 0 10px; border-bottom: 1px solid var(--ink); }
h1 { margin: 0; font-size: 15px; }
h2, h3 { margin: 0; font-size: 12px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; }
nav { margin-left: auto; display: flex; flex-wrap: wrap; gap: 2px 16px; }
nav a { text-decoration: none; padding: 10px 0; }
nav a.on { font-weight: 700; text-decoration: underline; text-underline-offset: 4px; text-decoration-thickness: 2px; }
.muted { color: var(--muted); }
main { display: flex; flex-wrap: wrap; gap: 24px; align-items: flex-start; padding-top: 14px; padding-bottom: 40px; }
aside { flex: 1 1 300px; min-width: 0; display: flex; flex-direction: column; gap: 8px; }
section.cam { flex: 999 1 560px; min-width: 0; display: flex; flex-direction: column; gap: 18px; }
.sec-head { display: flex; align-items: baseline; gap: 8px; border-bottom: 1px solid var(--ink); padding-bottom: 3px; }
.sec-head .fill { flex: 1; }
.scroll { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
th { font-weight: 400; color: var(--muted); text-align: left; padding: 2px 6px 4px 0; }
td { padding: 6px 6px 6px 0; vertical-align: top; white-space: nowrap; }
td.r, th.r { text-align: right; }
tr.row { border-bottom: 1px dotted var(--rule); cursor: pointer; }
tr.row td:first-child { padding-left: 4px; }
tr.sel { background: var(--ink); color: var(--bg); }
tr.sel a { color: var(--bg); }
tr.row a { text-decoration: none; font-weight: 700; }
.small { font-size: 11.5px; }
.b { font-weight: 700; }
.u { text-decoration: underline; }
.note { font-size: 12px; color: var(--muted); border-top: 1px dashed var(--rule); padding-top: 6px; }
.warn-list { display: flex; flex-direction: column; gap: 4px; font-size: 12px; }
.pi-empty { border: 1px dashed var(--rule); padding: 10px; font-size: 12px; color: var(--muted); }
.camhead { display: flex; flex-wrap: wrap; gap: 8px 16px; align-items: flex-end; border-bottom: 1px solid var(--ink); padding-bottom: 10px; }
.camhead .id { display: flex; flex-direction: column; gap: 2px; min-width: 0; flex: 1 1 320px; }
.camhead .title { display: flex; flex-wrap: wrap; gap: 8px; align-items: baseline; }
.camhead h2 { font-size: 17px; text-transform: none; letter-spacing: 0; }
.actions { display: flex; flex-wrap: wrap; gap: 6px; }
.row2 { display: flex; flex-wrap: wrap; gap: 24px; align-items: flex-start; }
.col-a { flex: 1 1 380px; min-width: 0; display: flex; flex-direction: column; gap: 6px; }
.col-b { flex: 1 1 340px; min-width: 0; display: flex; flex-direction: column; gap: 6px; }
.col-wide { flex: 999 1 380px; min-width: 0; display: flex; flex-direction: column; gap: 6px; }
.frame { position: relative; width: 100%; aspect-ratio: 4 / 3; background: var(--img); overflow: hidden; }
.frame img { position: absolute; inset: 0; width: 100%; height: 100%; object-fit: contain; }
.frame .label { position: absolute; left: 8px; top: 6px; font-size: 12px; color: #cccccc; }
.roi { position: absolute; border: 1px dashed #ffffff; pointer-events: none; }
.roi span { position: absolute; top: -18px; left: 0; font-size: 11px; color: #ffffff; background: #000000; padding: 0 3px; white-space: nowrap; }
.roi.editing { border: 2px solid #ffffff; box-shadow: 0 0 0 9999px rgba(0,0,0,.45); }
.drag { position: absolute; inset: 0; cursor: crosshair; touch-action: none; }
.roi-bar { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; }
.roi-bar .vals { font-size: 12px; margin-right: auto; }
select { font: inherit; font-size: 12.5px; min-height: 36px; border: 1px solid var(--ink); border-radius: 0; background: var(--bg); color: var(--ink); padding: 0 6px; }
dl { margin: 0; display: flex; flex-direction: column; }
dl div { display: flex; gap: 8px; align-items: baseline; padding: 3px 0; border-bottom: 1px dotted var(--rule); }
dt { color: var(--muted); flex: none; }
dd { margin: 0 0 0 auto; text-align: right; overflow-wrap: anywhere; }
.crash-head { display: flex; flex-wrap: wrap; gap: 4px 12px; align-items: baseline; border-bottom: 2px solid var(--ink); padding-bottom: 3px; }
pre { margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; font: 12px/1.5 var(--mono); }
.thumbs { display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 8px; }
.thumbs figure { margin: 0; }
.thumbs .img { aspect-ratio: 4 / 3; background: var(--thumb); display: block; overflow: hidden; }
.thumbs img { width: 100%; height: 100%; object-fit: cover; display: block; }
.thumbs figcaption { display: flex; justify-content: space-between; gap: 6px; font-size: 11.5px; padding-top: 2px; }
.log { font-size: 12px; line-height: 1.6; max-height: 320px; overflow: auto; }
.log div { white-space: pre-wrap; overflow-wrap: anywhere; }
.empty { color: var(--muted); padding: 24px 0; }
#toast { position: fixed; left: 50%; bottom: 16px; transform: translateX(-50%); background: var(--ink); color: var(--bg);
  padding: 6px 12px; font-size: 12.5px; opacity: 0; transition: opacity .2s; pointer-events: none; }
#toast.show { opacity: 1; }
</style>
</head>
<body>
<header class="wrap">
  <div class="bar">
    <h1>bird-cameras</h1>
    <span class="muted" id="meta">loading…</span>
    <nav aria-label="Sections">
      <a href="/" class="on">overview</a>
      <a href="/legacy">legacy dashboard (full config)</a>
    </nav>
  </div>
</header>
<main class="wrap">
  <aside aria-label="Cameras">
    <div class="sec-head"><h2>Cameras</h2><span class="fill"></span></div>
    <div class="scroll">
      <table>
        <thead><tr><th scope="col">st</th><th scope="col">camera</th><th scope="col" class="r">rssi</th><th scope="col" class="r">up</th><th scope="col" class="r">motion</th></tr></thead>
        <tbody id="cam-rows"></tbody>
      </table>
    </div>
    <div class="note">ON online · LT late (&gt;40s) · -- offline (&gt;2m) · zz camera powered down</div>
    <div class="warn-list" id="warnings"></div>

    <div class="sec-head" style="margin-top: 18px;"><h2>Pi</h2><span class="fill"></span><span class="small muted">monitor</span></div>
    <div class="pi-empty" id="pi">Checking Pi connection…</div>
  </aside>

  <section class="cam" aria-label="Selected camera" id="cam">
    <p class="empty">No heartbeats yet. Cameras running the current firmware report here every 15 s.</p>
  </section>
</main>
<div id="toast" role="status" aria-live="polite"></div>

<script>
const S = { cameras: [], registry: {}, selected: null, captures: {}, roi: {}, streaming: false, built: null, roiEdit: null, calType: "normal" };
try { S.selected = localStorage.getItem("fleet.selected"); } catch (e) {}
if (location.hash.startsWith("#cam=")) S.selected = decodeURIComponent(location.hash.slice(5));

const $ = id => document.getElementById(id);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const dur = s => s == null ? "–" : s < 90 ? `${s}s` : s < 5400 ? `${Math.round(s/60)}m` : s < 172800 ? `${(s/3600).toFixed(1)}h` : `${Math.round(s/86400)}d`;
const kb = b => b == null ? "?" : `${Math.round(b/1024)}K`;
const hms = t => t ? new Date(t*1000).toLocaleTimeString([], {hour:"2-digit", minute:"2-digit", second:"2-digit", hour12:false}) : "";
const hm = iso => { const d = new Date(iso); return isNaN(d) ? "" : d.toLocaleTimeString([], {hour:"2-digit", minute:"2-digit", hour12:false}); };
const short = id => id.replace(/^esp32cam-/, "").slice(0, 4);
function toast(m) { const t = $("toast"); t.textContent = m; t.classList.add("show"); clearTimeout(toast.h); toast.h = setTimeout(() => t.classList.remove("show"), 3200); }
// The server allows one operation per camera and only notices a closed stream on
// its next write, so retry briefly on "busy" instead of failing straight away.
async function apiRetry(url, opts, tries = 8) {
  for (let i = 0; ; i++) {
    try { return await api(url, opts); }
    catch (e) { if (i >= tries - 1 || !/busy/i.test(e.message)) throw e; await new Promise(r => setTimeout(r, 600)); }
  }
}
async function api(url, opts) { const r = await fetch(url, Object.assign({cache: "no-store"}, opts)); const b = await r.json().catch(() => ({})); if (!r.ok) throw new Error(b.error || `HTTP ${r.status}`); return b; }

const hb = c => c.last_heartbeat || {};
const cam = c => hb(c).camera || {};
const label = c => c.name || `cam-${short(c.camera_id)}`;
const code = c => c.status === "online" ? (cam(c).powered === false ? "zz" : "ON") : c.status === "late" ? "LT" : "--";
const reg = c => S.registry[c.camera_id];
const current = () => S.cameras.find(c => c.camera_id === S.selected);

function warnings(c) {
  const w = [], h = hb(c), boot = h.boot || {}, wifi = h.wifi || {};
  if (c.status === "offline") w.push(`offline ${dur(c.seconds_since_seen)}`);
  if (c.crashes_24h) w.push(`${c.crashes_24h} crash${c.crashes_24h > 1 ? "es" : ""} in 24h`);
  else if (c.bad_restarts_24h) w.push(`${c.bad_restarts_24h} bad restart${c.bad_restarts_24h > 1 ? "s" : ""} in 24h`);
  if (boot.brownouts_since_power_on) w.push(`${boot.brownouts_since_power_on} brownout(s) since power-on`);
  if ((wifi.disconnects_this_boot || 0) > 3) w.push(`${wifi.disconnects_this_boot} wi-fi drops this boot`);
  if (wifi.rssi_dbm && wifi.rssi_dbm < -78) w.push(`weak wi-fi ${wifi.rssi_dbm} dBm`);
  if (h.chip_temperature_c >= 65) w.push(`hot: chip ${Math.round(h.chip_temperature_c)}C`);
  if ((h.memory || {}).min_free_heap < 20000) w.push(`low memory ${kb(h.memory.min_free_heap)}`);
  if (cam(c).last_error) w.push(`error: ${cam(c).last_error}`);
  return w;
}

function renderSidebar() {
  $("cam-rows").innerHTML = S.cameras.map(c => {
    const h = hb(c), wifi = h.wifi || {}, r = wifi.rssi_dbm;
    const rc = r && r < -78 ? "b u" : r && r < -70 ? "b" : "";
    const sm = cam(c).seconds_since_motion;
    return `<tr class="row ${c.camera_id === S.selected ? "sel" : ""}" data-id="${esc(c.camera_id)}">
      <td class="b">${code(c)}</td>
      <td><a href="#cam=${encodeURIComponent(c.camera_id)}">${esc(label(c))}</a><div class="small" style="opacity:.8">${esc(h.camera_model || "?")} · ${esc(c.ip || "")}</div></td>
      <td class="r ${rc}">${r ? r : "–"}</td>
      <td class="r">${c.status === "offline" ? "–" : dur(h.uptime_s)}</td>
      <td class="r">${sm == null ? "–" : dur(sm)}</td></tr>`;
  }).join("") || `<tr><td colspan="5" class="muted">no cameras yet</td></tr>`;
  const lines = S.cameras.flatMap(c => warnings(c).map(w => `<div><span class="b">!</span> ${esc(label(c))}: ${esc(w)}</div>`));
  $("warnings").innerHTML = lines.join("");
}

function healthRows(c) {
  const h = hb(c), boot = h.boot || {}, wifi = h.wifi || {}, mem = h.memory || {}, k = cam(c);
  const bad = new Set(["crash", "interrupt_watchdog", "task_watchdog", "other_watchdog", "brownout"]);
  const sinceBad = (boot.crashes_since_power_on || 0) + (boot.watchdog_resets_since_power_on || 0) + (boot.brownouts_since_power_on || 0);
  const rssi = wifi.rssi_dbm;
  return [
    ["mode", `${k.mode || "?"}/${k.state || "?"}`],
    ["camera power", k.powered === false ? "off (idle)" : k.powered ? "on" : "?"],
    ["last decision", k.last_decision || "–"],
    ["motion events", `${k.motion_events ?? 0}` + (k.seconds_since_motion != null ? ` (${dur(k.seconds_since_motion)} ago)` : "")],
    ["last upload", k.last_upload_ok ? "ok" : k.motion_events ? "! failed" : "–", !k.last_upload_ok && k.motion_events],
    ["uptime", dur(h.uptime_s)],
    ["last restart", (bad.has(boot.reset_reason) ? "! " : "") + (boot.reset_reason || "?"), bad.has(boot.reset_reason)],
    ["restarts 24h", `${c.restarts_24h} (${c.bad_restarts_24h} bad)`, c.bad_restarts_24h > 0],
    ["since power-on", `${boot.crashes_since_power_on ?? 0} crash ${boot.watchdog_resets_since_power_on ?? 0} wdt ${boot.brownouts_since_power_on ?? 0} bo`, sinceBad > 0],
    ["wi-fi", rssi ? `${rssi} dBm ${rssi >= -60 ? "good" : rssi >= -70 ? "ok" : rssi >= -78 ? "weak" : "very weak"}` : "–", rssi && rssi < -70],
    ["wi-fi drops", `${wifi.disconnects_this_boot ?? 0} this boot` + (wifi.last_disconnect_reason_name ? ` (${wifi.last_disconnect_reason_name})` : "")],
    ["heap free/low", `${kb(mem.free_heap)}/${kb(mem.min_free_heap)}`, mem.min_free_heap < 20000],
    ["chip temp", h.chip_temperature_c != null ? `${Math.round(h.chip_temperature_c)}C (uncal.)` : "–", h.chip_temperature_c >= 65],
  ].map(([k, v, hot]) => `<div><dt>${esc(k)}</dt><dd class="${hot ? "b" : ""}">${esc(v)}</dd></div>`).join("");
}

function buildMain(c) {
  S.built = c.camera_id; S.streaming = false; S.roiEdit = null;
  $("cam").innerHTML = `
    <div class="camhead">
      <div class="id"><div class="title"><h2 id="c-name"></h2><span id="c-tags"></span></div>
        <span class="small muted" id="c-meta" style="overflow-wrap:anywhere"></span></div>
      <div class="actions" id="c-actions"></div>
    </div>
    <div class="row2">
      <div class="col-a">
        <div class="sec-head"><h3>Preview</h3><span class="fill"></span><span class="small muted" id="p-cap"></span></div>
        <div class="frame" id="frame"><span class="label" id="p-label">[no image yet]</span></div>
        <div class="roi-bar" id="roi-bar"></div>
      </div>
      <div class="col-b"><div class="sec-head"><h3>Health</h3></div><dl id="health"></dl></div>
    </div>
    <div id="crash"></div>
    <div class="row2">
      <div class="col-wide"><div class="sec-head"><h3>Latest captures</h3><span class="fill"></span><span class="small muted" id="cap-count"></span></div><div class="thumbs" id="thumbs"></div></div>
      <div class="col-b"><div class="sec-head"><h3>Event log</h3><span class="fill"></span><a class="small" id="report" target="_blank" rel="noopener">full report</a></div><div class="log" id="events"></div></div>
    </div>`;
  loadRoi(c); loadCaptures(c);
}

function setPreview(src, caption) {
  const f = $("frame"); let img = f.querySelector("img");
  // Clearing src is what makes the browser drop an MJPEG stream; removing the
  // element alone can leave the download (and the server's camera lock) running.
  if (!src) { if (img) { img.removeAttribute("src"); img.remove(); } $("p-label").style.display = ""; $("p-cap").textContent = caption || ""; return; }
  if (!img) { img = document.createElement("img"); img.alt = "Camera image"; f.prepend(img); }
  if (img.getAttribute("src") !== src) img.src = src;
  $("p-label").style.display = "none"; $("p-cap").textContent = caption || "";
}

function drawRoi(c) {
  const f = $("frame"); if (!f) return; f.querySelector(".roi")?.remove();
  const r = S.roi[c.camera_id]; if (!r || (r.x === 0 && r.y === 0 && r.width >= 1 && r.height >= 1)) return;
  const d = document.createElement("div"); d.className = "roi";
  Object.assign(d.style, {left: `${r.x*100}%`, top: `${r.y*100}%`, width: `${r.width*100}%`, height: `${r.height*100}%`});
  d.innerHTML = "<span>roi</span>"; f.append(d);
}

function renderMain() {
  const c = current();
  if (!c) { if (!S.cameras.length) return; S.selected = S.cameras[0].camera_id; return renderAll(); }
  if (S.built !== c.camera_id) buildMain(c);
  const h = hb(c), k = cam(c), r = reg(c);
  $("c-name").textContent = label(c);
  $("c-tags").textContent = `[${c.status}${c.status !== "online" ? " " + dur(c.seconds_since_seen) : ""}] [${k.motion_enabled ? "motion armed" : "idle"}] [camera ${k.powered === false ? "off" : "on"}]`;
  $("c-meta").textContent = `${h.camera_model || "?"} ${c.camera_id} ${c.ip || ""} fw:${(h.firmware_elf_sha256 || "").slice(0, 8)}`;
  const can = !!r, off = c.status === "offline";
  $("c-actions").innerHTML = (can ? "" : `<button type="button" data-act="register" title="Add this camera to the registry so the server can stream and capture from it">register</button>`) +
    (can ? `<button type="button" class="${r.needs_name ? "primary" : ""}" data-act="name" title="Give this camera a name">${r.needs_name ? "name this camera" : "rename"}</button>` : "") +
    `<button type="button" class="${S.streaming ? "" : "primary"}" data-act="stream" ${can && !off ? "" : "disabled"}>${S.streaming ? "stop stream" : "stream"}</button>
     <button type="button" data-act="capture" ${can && !off ? "" : "disabled"}>capture</button>
     <button type="button" data-act="${k.motion_enabled ? "idle" : "motion"}" aria-pressed="${!!k.motion_enabled}" ${off ? "disabled" : ""}>motion [${k.motion_enabled ? "x" : " "}] ${k.motion_enabled ? "on" : "off"}</button>
     <button type="button" class="dashed" data-act="reboot">reboot</button>
     <button type="button" class="dashed" data-act="remove" title="Delete this camera and its history from the dashboard">remove</button>`;
  $("health").innerHTML = healthRows(c);
  const crash = (c.crashes || []).slice(-1)[0];
  if (crash) {
    const d = crash.decoded || {};
    const frames = (d.frames && d.frames.length) ? d.frames.map(f => "  " + f.replace(/ at \S*[\/\\]/, " at ")) : (crash.backtrace || []).map(a => "  " + a);
    const note = {firmware_mismatch: "firmware rebuilt since this crash: can't decode", no_addr2line: "toolchain not found: can't decode", no_firmware_elf: "firmware.elf not found: can't decode"}[d.status];
    $("crash").innerHTML = `<div class="crash-head"><h3>!! Last crash</h3><span>${esc(hms(crash.received))} ${esc(crash.task)} ${esc(crash.cause_name)} @${esc(crash.pc)}${crash.backtrace_corrupted ? " (backtrace corrupted)" : ""}</span>
      <span style="flex:1"></span><span class="small muted">${c.crashes_24h} in 24h</span></div><pre>${esc(frames.join("\n"))}${note ? "\n  (" + esc(note) + ")" : ""}</pre>`;
  } else $("crash").innerHTML = "";
  const cur = (h.boot || {}).number_since_power_on;
  $("events").innerHTML = (c.events || []).slice().reverse().map(e => {
    const cls = /CRASH|HEAP|failed|rebooting|rejected|watchdog/i.test(e.text) ? "b" : e.boot !== cur ? "muted" : "";
    const t = e.approx_time ? hms(e.approx_time) : `b${e.boot} +${dur(Math.round((e.t_ms || 0) / 1000))}`;
    return `<div class="${cls}"><span class="muted">${esc(t.padEnd(8))}</span> ${esc(e.text)}</div>`;
  }).join("") || `<div class="muted">no events yet</div>`;
  $("report").href = `/api/fleet/${encodeURIComponent(c.camera_id)}/report.txt`;
  if (!S.roiEdit) renderRoiBar();
  if (S.roiEdit) return;
  if (!S.streaming) {
    const latest = (S.captures[c.camera_id] || [])[0];
    setPreview(latest ? `/api/fleet/captures/${encodeURIComponent(latest.filename)}` : null,
               latest ? `last capture ${hm(latest.received_at)}` : "no captures yet");
  }
  drawRoi(c);
}

const CAL_TYPES = ["normal", "dark_frame", "flat_field", "gray_card", "color_chart", "resolution_target"];  // must match capture_storage.CalibrationType
function contentBox() {
  // The image is letterboxed with object-fit: contain; map pointer coords to the picture itself.
  const f = $("frame"), img = f.querySelector("img"), r = f.getBoundingClientRect();
  if (!img || !img.naturalWidth) return {left: 0, top: 0, width: r.width, height: r.height};
  const s = Math.min(r.width / img.naturalWidth, r.height / img.naturalHeight);
  const w = img.naturalWidth * s, h = img.naturalHeight * s;
  return {left: (r.width - w) / 2, top: (r.height - h) / 2, width: w, height: h};
}
function roiVals(r) { return `roi x ${r.x.toFixed(3)} y ${r.y.toFixed(3)} w ${r.width.toFixed(3)} h ${r.height.toFixed(3)}`; }
function renderRoiBar() {
  const c = current(); if (!c || !$("roi-bar")) return;
  const r = reg(c), off = c.status === "offline", can = !!r && !off;
  if (S.roiEdit) {
    $("roi-bar").innerHTML = `<span class="vals b" id="roi-vals">${esc(roiVals(S.roiEdit.roi))}</span>
      <button type="button" class="primary" data-act="roi-save">save roi</button>
      <button type="button" data-act="roi-full">full frame</button>
      <button type="button" data-act="roi-frame">new frame</button>
      <button type="button" data-act="roi-cancel">cancel</button>`;
  } else {
    const cur = S.roi[c.camera_id];
    $("roi-bar").innerHTML = `<span class="vals muted">${cur ? esc(roiVals(cur)) : "roi: unknown"}</span>
      <button type="button" data-act="roi-edit" ${can ? "" : "disabled"}>edit roi</button>
      <label class="small muted" for="cal-type">capture as</label><select id="cal-type">${CAL_TYPES.map(t => `<option value="${t}" ${t === S.calType ? "selected" : ""}>${t.replace(/_/g, " ")}</option>`).join("")}</select>`;
  }
}
function drawRoiEdit() {
  const f = $("frame"); f.querySelector(".roi")?.remove(); f.querySelector(".drag")?.remove();
  const b = contentBox(), r = S.roiEdit.roi;
  const d = document.createElement("div"); d.className = "roi editing";
  Object.assign(d.style, {left: `${b.left + r.x * b.width}px`, top: `${b.top + r.y * b.height}px`, width: `${r.width * b.width}px`, height: `${r.height * b.height}px`});
  d.innerHTML = "<span>drag to draw the motion region</span>"; f.append(d);
  const drag = document.createElement("div"); drag.className = "drag"; f.append(drag);
  const pt = e => { const fr = f.getBoundingClientRect(), bx = contentBox();
    return {x: Math.min(1, Math.max(0, (e.clientX - fr.left - bx.left) / bx.width)), y: Math.min(1, Math.max(0, (e.clientY - fr.top - bx.top) / bx.height))}; };
  drag.addEventListener("pointerdown", e => { S.roiEdit.start = pt(e); drag.setPointerCapture(e.pointerId); e.preventDefault(); });
  drag.addEventListener("pointermove", e => {
    const a = S.roiEdit.start; if (!a) return; const p = pt(e);
    S.roiEdit.roi = {x: Math.min(a.x, p.x), y: Math.min(a.y, p.y), width: Math.abs(p.x - a.x), height: Math.abs(p.y - a.y)};
    const bx = contentBox(), rr = S.roiEdit.roi;
    Object.assign(d.style, {left: `${bx.left + rr.x * bx.width}px`, top: `${bx.top + rr.y * bx.height}px`, width: `${rr.width * bx.width}px`, height: `${rr.height * bx.height}px`});
    $("roi-vals").textContent = roiVals(rr);
  });
  const end = () => { if (!S.roiEdit.start) return; S.roiEdit.start = null;
    if (S.roiEdit.roi.width < 0.02 || S.roiEdit.roi.height < 0.02) { toast("region too small; drag a larger box"); } };
  drag.addEventListener("pointerup", end); drag.addEventListener("pointercancel", end);
}
async function startRoiEdit(c) {
  const r = reg(c); if (!r) return;
  if (S.streaming) { S.streaming = false; setPreview(null); await new Promise(res => setTimeout(res, 400)); }
  S.roiEdit = {roi: Object.assign({x: 0, y: 0, width: 1, height: 1}, S.roi[c.camera_id] || {}), start: null};
  renderRoiBar();
  toast("grabbing a fresh frame… (a powered-down camera takes ~2 s)");
  for (let attempt = 0; attempt < 4 && S.roiEdit; attempt++) {
    const ok = await new Promise((resolve) => {
      setPreview(`/api/cameras/${encodeURIComponent(r.id)}/capture?t=${Date.now()}`, "live frame (not saved)");
      const img = $("frame").querySelector("img"); img.onload = () => resolve(true); img.onerror = () => resolve(false);
    });
    if (ok) break;
    if (attempt === 3) toast("couldn't get a frame from the camera; drawing on the last image");
    else await new Promise(res => setTimeout(res, 800));  // camera may still be busy closing a stream
  }
  if (S.roiEdit) drawRoiEdit();
}
function stopRoiEdit() {
  S.roiEdit = null; const f = $("frame"); f?.querySelector(".drag")?.remove(); f?.querySelector(".roi")?.remove();
  renderRoiBar(); renderMain();
}

function renderCaptures() {
  const c = current(); if (!c || !$("thumbs")) return;
  const list = S.captures[c.camera_id] || [];
  $("cap-count").textContent = list.length ? `newest ${list.length}` : "";
  $("thumbs").innerHTML = list.map(x => {
    const tag = x.classification ? (x.classification.is_bird ? `BIRD ${(x.classification.confidence ?? 0).toFixed(2).replace(/^0/, "")}` : "none") : (x.source || "");
    return `<figure><a class="img" href="/api/fleet/captures/${encodeURIComponent(x.filename)}" target="_blank" rel="noopener"><img loading="lazy" alt="Capture ${esc(x.received_at)}" src="/api/fleet/captures/${encodeURIComponent(x.filename)}"></a>
      <figcaption><span class="muted">${esc(hm(x.received_at))}</span><span class="${x.classification?.is_bird ? "b" : "muted"}">${esc(tag)}</span></figcaption></figure>`;
  }).join("") || `<div class="muted small">nothing saved from this camera yet</div>`;
}

async function loadCaptures(c) {
  try { S.captures[c.camera_id] = (await api(`/api/fleet/${encodeURIComponent(c.camera_id)}/captures?limit=6`)).captures; renderCaptures(); renderMain(); } catch (e) {}
}
async function loadRoi(c) {
  const r = reg(c); if (!r) return;
  try { const cfg = await api(`/api/cameras/${encodeURIComponent(r.id)}/config`); if (cfg.motion_roi) { S.roi[c.camera_id] = cfg.motion_roi; if (!S.roiEdit) { drawRoi(c); renderRoiBar(); } } } catch (e) {}
}

function renderAll() {
  const online = S.cameras.filter(c => c.status === "online").length;
  const crashes = S.cameras.reduce((n, c) => n + (c.crashes_24h || 0), 0);
  $("meta").textContent = `${online}/${S.cameras.length} online · ${crashes} crash/24h${S.clock && !S.clock.trusted ? " · HUB CLOCK UNVERIFIED" : ""} · updated ${new Date().toLocaleTimeString([], {hour12: false})}`;
  renderSidebar(); renderMain();
}

// The hub may have no battery clock: offer this device's time as a reference every 5 minutes.
let lastTimeOffer = 0;
function offerTime() {
  if (Date.now() - lastTimeOffer < 300000) return;
  lastTimeOffer = Date.now();
  fetch("/api/time", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({client_unix_ms: Date.now()})}).catch(() => {});
}

async function refresh() {
  refreshPi();
  try {
    const [fleet, registry] = await Promise.all([api("/api/fleet"), api("/api/cameras").catch(() => ({cameras: []}))]);
    S.cameras = fleet.cameras || []; S.clock = fleet.hub_clock || null;
    offerTime();
    S.registry = {}; for (const r of registry.cameras || []) if (r.hardware_camera_id) S.registry[r.hardware_camera_id] = r;
    renderAll();
  } catch (e) { $("meta").textContent = "server unreachable – retrying"; }
}

let piLoading = false, piCheckedAt = 0;
async function refreshPi() {
  if (piLoading || Date.now() - piCheckedAt < 10000) return;
  piLoading = true; piCheckedAt = Date.now();
  try {
    let config = null, local = false;
    const response = await fetch('/api/pi/config', {cache:'no-store'});
    if (response.status === 404) local = true;
    else if (!response.ok) throw Error('Pi monitor configuration unavailable');
    else config = await response.json();
    if (!local && !config.pi_url) { $("pi").innerHTML = '<a href="/pi">Connect Pi and open collection monitor</a>'; return; }
    const h = await api(local ? '/health' : '/api/pi/health');
    const services = Object.values(h.motion_services || {});
    const failures = services.reduce((n,s) => n + (s.failures || 0), 0);
    $("pi").innerHTML = `<strong>${h.collection_mode ? 'Collection · detector only' : 'Receiver online'}</strong><br>` +
      `${esc(local ? 'This hub' : config.pi_url)}<br>${services.length} cameras · ${failures} capture failures · queue ${h.classification_queue_depth ?? '–'}<br>` +
      `<a href="${local ? '/' : '/pi'}">Open collection monitor &amp; review</a>`;
  } catch (e) { $("pi").innerHTML = `Pi unavailable · ${esc(e.message)}<br><a href="/pi">Connection settings</a>`; }
  finally { piLoading = false; }
}

function select(id) {
  if (!id || id === S.selected) return;
  S.selected = id; S.built = null;
  try { localStorage.setItem("fleet.selected", id); } catch (e) {}
  history.replaceState(null, "", `#cam=${encodeURIComponent(id)}`);
  renderAll();
}

document.addEventListener("click", async e => {
  const row = e.target.closest("tr.row");
  if (row) { e.preventDefault(); select(row.dataset.id); return; }
  const b = e.target.closest("button[data-act]"); if (!b) return;
  const c = current(); if (!c) return;
  const id = encodeURIComponent(c.camera_id), r = reg(c), act = b.dataset.act;
  if (act === "reboot" && !confirm(`Reboot ${label(c)}?`)) return;
  if (act === "remove" && !confirm(`Remove ${label(c)} from the dashboard?\n\nIts saved images are kept. A camera that is still powered on will add itself again within about 15 seconds.`)) return;
  b.disabled = true;
  try {
    if (act === "reboot") { const x = await api(`/api/fleet/${id}/reboot`, {method: "POST"}); toast(x.result === "rebooting" ? "rebooting now" : "no answer; will reboot on next heartbeat"); }
    else if (act === "motion" || act === "idle") { await api(`/api/fleet/${id}/mode`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({mode: act})}); toast(`motion ${act === "motion" ? "on" : "off"}`); }
    else if (act === "register") { await api("/api/cameras", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({name: label(c), address: c.ip})}); toast("registered"); }
    else if (act === "remove") {
      if (S.streaming) { S.streaming = false; setPreview(null); }
      // Both records go: the registry entry (stream/capture target) and the fleet history.
      if (r) await api(`/api/cameras/${encodeURIComponent(r.id)}`, {method: "DELETE"}).catch(() => {});
      await api(`/api/fleet/${id}`, {method: "DELETE"}).catch(() => {});
      S.selected = null; S.built = null;
      try { localStorage.removeItem("fleet.selected"); } catch (e) {}
      toast("removed");
    }
    else if (act === "name") {
      const n = (prompt("Name for this camera (for example: Back garden feeder)", r.needs_name ? "" : r.name) || "").trim();
      if (n) { await api(`/api/cameras/${encodeURIComponent(r.id)}`, {method: "PATCH", headers: {"Content-Type": "application/json"}, body: JSON.stringify({name: n})}); toast("named"); }
    }
    else if (act === "capture") {
      // The server allows one operation per camera; a running stream would make this a 409.
      if (S.streaming) { S.streaming = false; setPreview(null); await new Promise(res => setTimeout(res, 400)); }
      const type = $("cal-type")?.value || "normal";
      toast("capturing… (a powered-down camera takes ~2 s to wake)");
      const saved = await apiRetry(`/api/cameras/${encodeURIComponent(r.id)}/captures`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({calibration_type: type})});
      toast(`saved ${saved.filename || ""}${type !== "normal" ? " (" + type.replace(/_/g, " ") + ")" : ""}`);
      await loadCaptures(c);
    }
    else if (act === "roi-edit") { b.disabled = false; await startRoiEdit(c); return; }
    else if (act === "roi-cancel") { stopRoiEdit(); return; }
    else if (act === "roi-full") { S.roiEdit.roi = {x: 0, y: 0, width: 1, height: 1}; drawRoiEdit(); $("roi-vals").textContent = roiVals(S.roiEdit.roi); b.disabled = false; return; }
    else if (act === "roi-frame") { const keep = S.roiEdit.roi; b.disabled = false; await startRoiEdit(c); if (S.roiEdit) { S.roiEdit.roi = keep; drawRoiEdit(); $("roi-vals").textContent = roiVals(keep); } return; }
    else if (act === "roi-save") {
      const q = v => Math.round(v * 1000) / 1000, x = S.roiEdit.roi;
      const roi = {x: q(x.x), y: q(x.y), width: q(Math.min(x.width, 1 - x.x)), height: q(Math.min(x.height, 1 - x.y))};
      if (roi.width < 0.02 || roi.height < 0.02) { toast("region too small; drag a larger box"); b.disabled = false; return; }
      const cfg = await apiRetry(`/api/cameras/${encodeURIComponent(r.id)}/config`, {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({motion_roi: roi})});
      S.roi[c.camera_id] = cfg.motion_roi || roi; toast("roi saved to camera; motion reference will relearn"); stopRoiEdit();
    }
    else if (act === "stream") {
      if (S.roiEdit) stopRoiEdit();
      S.streaming = !S.streaming;
      if (S.streaming) setPreview(`/api/cameras/${encodeURIComponent(r.id)}/stream?t=${Date.now()}`, "live stream");
      else { setPreview(null); }
    }
  } catch (err) { toast(err.message || "request failed"); }
  b.disabled = false; await refresh();
});

document.addEventListener("change", e => { if (e.target.id === "cal-type") S.calType = e.target.value; });
refresh(); setInterval(refresh, 5000);
setInterval(() => { const c = current(); if (c) loadCaptures(c); }, 20000);
</script>
</body>
</html>
"""
