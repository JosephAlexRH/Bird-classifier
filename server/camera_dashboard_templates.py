"""Multi-camera tabbed dashboards shared by the laptop and Raspberry Pi servers."""

from __future__ import annotations


LAPTOP_TABBED_DASHBOARD_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ESP32-CAM multi-camera dashboard</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #101418; color: #edf2f7; }
    header, main { width: min(1180px, calc(100% - 2rem)); margin: auto; }
    header { padding: 1.5rem 0 .75rem; }
    h1, h2 { margin: 0 0 .75rem; }
    .muted { color: #aab5c0; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(330px, 1fr)); gap: 1rem; padding-bottom: 2rem; }
    section { background: #192128; border: 1px solid #2d3943; border-radius: 12px; padding: 1rem; }
    img { display: block; width: 100%; min-height: 240px; object-fit: contain; background: #080a0c; border-radius: 8px; }
    button, select, input { padding: .6rem .75rem; margin: .25rem .25rem .25rem 0; }
    button { cursor: pointer; }
    button:disabled { cursor: not-allowed; opacity: .55; }
    label { display: block; margin-top: .65rem; }
    label input, label select { display: block; width: 100%; box-sizing: border-box; }
    textarea { width: 100%; min-height: 210px; box-sizing: border-box; font: 13px monospace; }
    pre { min-height: 5rem; max-height: 24rem; overflow: auto; white-space: pre-wrap; }
    .controls { margin: .6rem 0; }
    .badge { display: inline-block; padding: .25rem .55rem; border-radius: 999px; background: #303b44; }
    .badge.ok { background: #174b31; } .badge.running { background: #173f61; }
    .badge.error { background: #632828; }
    .ok { color: #73d99f; } .error { color: #ff8b8b; } .running { color: #78bfff; }
    .tabs { display: flex; gap: .4rem; overflow-x: auto; padding: .25rem 0 .85rem; }
    .tab { flex: 0 0 auto; border: 1px solid #3b4954; border-radius: 999px; background: #202a32; color: inherit; }
    .tab[aria-selected="true"] { background: #14608b; border-color: #42b8f4; }
    .tab.offline { border-color: #8a3b3b; }
    .tab.disabled { opacity: .55; }
    .camera-heading { display: flex; flex-wrap: wrap; justify-content: space-between; gap: .5rem; align-items: baseline; }
    .camera-heading p { margin: 0; }
    .roi-stage { position: relative; width: 100%; background: #080a0c; border-radius: 8px; overflow: hidden; }
    .roi-stage img { min-height: 0; height: auto; object-fit: fill; }
    .roi-stage canvas { position: absolute; inset: 0; width: 100%; height: 100%; touch-action: none; cursor: crosshair; }
    .empty { padding: 1rem; border: 1px dashed #4b5964; border-radius: 10px; }
  </style>
</head>
<body>
  <header>
    <h1>ESP32-CAM laptop dashboard</h1>
    <p class="muted">Each camera tab has independent controls, stream state, configuration, ROI, and results.</p>
    <div id="camera-tabs" class="tabs" role="tablist" aria-label="Registered cameras"></div>
    <div class="camera-heading">
      <div>
        <h2 id="active-camera-name">No camera selected</h2>
        <p id="active-camera-details" class="muted">Add a camera below.</p>
      </div>
      <p>Camera state: <strong id="camera-state" class="badge">Not checked</strong></p>
    </div>
  </header>
  <main>
    <p id="no-camera" class="empty muted">No enabled camera is selected. Add or enable a camera in Camera fleet.</p>
    <div id="camera-workspace" class="grid" hidden>
      <section>
        <h2>Camera preview</h2>
        <img id="live" alt="ESP32-CAM preview is stopped">
        <div class="controls">
          <button id="start" class="camera-action" onclick="startLive()">Start stream</button>
          <button id="stop" onclick="stopLive()" disabled>Stop stream</button>
          <button class="camera-action" onclick="setCameraMode('motion')">Arm motion (on-board)</button>
          <button class="camera-action" onclick="setCameraMode('idle')">Idle</button>
        </div>
        <p id="live-message" class="muted">Preview stopped.</p>
      </section>
      <section>
        <h2>Latest saved capture for this camera</h2>
        <img id="latest" alt="No saved capture received for this camera">
        <p id="upload-message" class="muted">No saved capture received for this camera yet.</p>
      </section>
      <section>
        <h2>Manual / calibration capture</h2>
        <label>Capture type
          <select id="calibration-type">
            <option value="normal">Normal</option>
            <option value="dark_frame">Dark frame (cover the lens)</option>
            <option value="flat_field">Flat field</option>
            <option value="gray_card">Gray card</option>
            <option value="color_chart">Color chart</option>
            <option value="resolution_target">Resolution target</option>
          </select>
        </label>
        <label>Label<input id="capture-label" maxlength="120" placeholder="Optional dataset label"></label>
        <label>Note<input id="capture-note" maxlength="500" placeholder="Lighting, target, distance, etc."></label>
        <button id="capture" class="camera-action" onclick="saveCapture()">Capture and save</button>
        <button class="camera-action" onclick="saveProfile()">Save camera profile</button>
        <p class="muted">The capture and profile are tagged with the active camera ID and name.</p>
      </section>
      <section>
        <h2>Motion region of interest</h2>
        <div class="roi-stage">
          <img id="roi-preview" data-roi-preview alt="Refresh this camera preview, then drag to select a motion ROI">
          <canvas id="roi-overlay" aria-label="Drag to select a rectangular motion region"></canvas>
        </div>
        <p id="roi-values" class="muted">x 0.000, y 0.000, width 1.000, height 1.000</p>
        <div class="controls">
          <button class="camera-action" onclick="refreshRoiPreview()">Refresh ROI preview</button>
          <button class="camera-action" onclick="loadMotionRoi()">Load current ROI</button>
          <button class="camera-action" onclick="resetMotionRoi()">Reset full frame</button>
          <button class="camera-action" onclick="saveMotionRoi()">Save ROI</button>
        </div>
        <p class="muted">Save sends only <code>motion_roi</code> to the active camera.</p>
      </section>
      <section>
        <h2>Camera checks</h2>
        <button class="camera-action" onclick="cameraGet('status')">Status details</button>
        <button class="camera-action" onclick="cameraGet('motion')">Consume motion event</button>
        <button class="camera-action" onclick="cameraGet('metadata')">Metadata</button>
        <button class="camera-action" onclick="loadConfig()">Load config</button>
        <button class="camera-action" onclick="pollActiveCamera()">Poll registry</button>
        <pre id="result">Choose an action.</pre>
      </section>
      <section>
        <h2>Camera configuration</h2>
        <textarea id="config">{
  "onboard_motion_detection": false,
  "motion_enabled": false
}</textarea>
        <button class="camera-action" onclick="saveConfig()">Save config to this camera</button>
        <p class="muted"><code>onboard_motion_detection: false</code> leaves motion detection to the laptop/Pi service. Set it to <code>true</code> to use the ESP32 detector.</p>
      </section>
    </div>
    <section>
      <h2>Camera fleet</h2>
      <p class="muted">Adding a camera creates a tab. The legacy default is no longer used by dashboard actions.</p>
      <input id="camera-name" placeholder="Camera name (optional)">
      <input id="camera-address" placeholder="192.168.1.84">
      <button onclick="addCamera()">Add camera</button>
      <button onclick="loadCameras()">Refresh cameras</button>
      <div id="camera-list" class="muted">Loading cameras...</div>
    </section>
  </main>
  <script>
    'use strict';
    const fullFrame = {x: 0, y: 0, width: 1, height: 1};
    const tabState = new Map();
    let cameras = [];
    let activeCameraId = localStorage.getItem('esp32-active-camera') || null;
    let roi = {...fullFrame};
    let dragStart = null;
    const liveImage = document.getElementById('live');
    const result = document.getElementById('result');

    function activeCamera() { return cameras.find(camera => camera.id === activeCameraId) || null; }
    function stateFor(id) {
      if (!tabState.has(id)) tabState.set(id, {liveRunning: false, result: 'Choose an action.', resultError: false, config: '', roi: {...fullFrame}});
      return tabState.get(id);
    }
    function apiPath(endpoint = '') {
      if (!activeCameraId) throw new Error('Select a camera tab first.');
      return `/api/cameras/${encodeURIComponent(activeCameraId)}${endpoint ? `/${endpoint}` : ''}`;
    }
    function clamp(value) { return Math.max(0, Math.min(1, value)); }
    function normalized(value) {
      if (!value || typeof value !== 'object') return {...fullFrame};
      const next = {};
      for (const key of ['x', 'y', 'width', 'height']) {
        if (typeof value[key] !== 'number' || !Number.isFinite(value[key])) return {...fullFrame};
        next[key] = clamp(value[key]);
      }
      next.width = Math.min(next.width, 1 - next.x);
      next.height = Math.min(next.height, 1 - next.y);
      return next;
    }
    function pretty(text) { try { return JSON.stringify(JSON.parse(text), null, 2); } catch (_) { return text; } }
    function showResult(text, failed = false) {
      const state = activeCameraId ? stateFor(activeCameraId) : null;
      if (state) { state.result = pretty(String(text)); state.resultError = failed; }
      result.className = failed ? 'error' : 'ok';
      result.textContent = pretty(String(text));
    }
    function showError(error) { showResult(String(error), true); }
    window.showCameraResult = showResult;

    function setState(label, style = '') {
      const badge = document.getElementById('camera-state');
      badge.textContent = label.replaceAll('_', ' ');
      badge.className = `badge ${style}`;
    }
    function saveVisibleState() {
      if (!activeCameraId) return;
      const state = stateFor(activeCameraId);
      state.config = document.getElementById('config').value;
      state.roi = {...roi};
    }
    function updateControls() {
      const camera = activeCamera();
      const liveRunning = camera ? stateFor(camera.id).liveRunning : false;
      document.getElementById('start').disabled = !camera || !camera.enabled || liveRunning;
      document.getElementById('stop').disabled = !liveRunning;
      document.querySelectorAll('.camera-action').forEach(button => {
        if (button.id !== 'start') button.disabled = !camera || !camera.enabled || liveRunning;
      });
    }
    function renderTabs() {
      const tabs = document.getElementById('camera-tabs');
      tabs.replaceChildren();
      cameras.forEach(camera => {
        const button = document.createElement('button');
        button.className = `tab${camera.reachable === false ? ' offline' : ''}${camera.enabled ? '' : ' disabled'}`;
        button.type = 'button';
        button.role = 'tab';
        button.ariaSelected = camera.id === activeCameraId ? 'true' : 'false';
        button.textContent = `${camera.name}${camera.streaming ? ' • streaming' : ''}`;
        button.onclick = () => selectCamera(camera.id);
        tabs.appendChild(button);
      });
    }
    function renderFleet() {
      const list = document.getElementById('camera-list');
      list.replaceChildren();
      if (!cameras.length) { list.textContent = 'No cameras registered.'; return; }
      cameras.forEach(camera => {
        const row = document.createElement('p');
        const sensor = camera.sensor?.model || camera.status?.camera_sensor_model || 'unknown sensor';
        row.append(`${camera.name} — ${camera.url} — ${sensor} — ${camera.enabled ? 'enabled' : 'disabled'} `);
        const select = document.createElement('button'); select.textContent = 'Open tab'; select.onclick = () => selectCamera(camera.id);
        const toggle = document.createElement('button'); toggle.textContent = camera.enabled ? 'Disable' : 'Enable'; toggle.onclick = () => updateCamera(camera.id, {enabled: !camera.enabled});
        const poll = document.createElement('button'); poll.textContent = 'Poll'; poll.onclick = () => pollCamera(camera.id);
        const remove = document.createElement('button'); remove.textContent = 'Remove'; remove.onclick = () => removeCamera(camera.id);
        const rename = document.createElement('button'); rename.textContent = camera.needs_name ? 'Name this camera' : 'Rename'; rename.onclick = () => renameCamera(camera.id);
        row.append(rename, select, toggle, poll, remove); list.appendChild(row);
      });
    }
    function renderActiveCamera() {
      const camera = activeCamera();
      document.getElementById('no-camera').hidden = Boolean(camera);
      document.getElementById('camera-workspace').hidden = !camera;
      if (!camera) {
        document.getElementById('active-camera-name').textContent = 'No camera selected';
        document.getElementById('active-camera-details').textContent = 'Add or enable a camera below.';
        setState('not checked'); updateControls(); return;
      }
      const sensor = camera.sensor?.model || camera.status?.camera_sensor_model || 'unknown sensor';
      document.getElementById('active-camera-name').textContent = camera.name;
      document.getElementById('active-camera-details').textContent = `${camera.url} | ${sensor} | ${camera.enabled ? 'enabled' : 'disabled'}`;
      const state = stateFor(camera.id);
      document.getElementById('config').value = state.config || '{\n  "onboard_motion_detection": false,\n  "motion_enabled": false\n}';
      result.textContent = state.result; result.className = state.resultError ? 'error' : '';
      roi = normalized(state.roi); drawRoi();
      document.getElementById('latest').src = `${apiPath('latest.jpg')}?t=${Date.now()}`;
      document.getElementById('upload-message').textContent = `Latest saved image tagged to ${camera.name}.`;
      if (camera.status?.runtime_state) {
        const runtime = camera.status.runtime_state;
        setState(runtime, runtime === 'error' ? 'error' : runtime === 'motion_armed' ? 'ok' : '');
      } else if (camera.reachable === false) setState('unreachable', 'error');
      else if (camera.reachable === true) setState('reachable', 'ok');
      else setState('not checked');
      updateControls();
    }
    function selectCamera(id) {
      if (id === activeCameraId) return;
      saveVisibleState();
      if (activeCameraId && stateFor(activeCameraId).liveRunning) stopLive();
      activeCameraId = id;
      localStorage.setItem('esp32-active-camera', id);
      renderTabs(); renderActiveCamera();
      refreshCameraStatus().catch(showError);
    }
    async function loadCameras() {
      saveVisibleState();
      const response = await fetch('/api/cameras', {cache: 'no-store'});
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.error || 'Could not load cameras.');
      cameras = payload.cameras || [];
      if (!cameras.some(camera => camera.id === activeCameraId)) {
        activeCameraId = (cameras.find(camera => camera.enabled) || cameras[0] || {}).id || null;
      }
      if (activeCameraId) localStorage.setItem('esp32-active-camera', activeCameraId);
      renderTabs(); renderFleet(); renderActiveCamera();
    }
    async function addCamera() {
      const address = document.getElementById('camera-address').value.trim();
      const name = document.getElementById('camera-name').value.trim() || null;
      const response = await fetch('/api/cameras', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({address, name})});
      const payload = await response.json();
      if (!response.ok) return showError(payload.error || 'Could not add camera.');
      document.getElementById('camera-address').value = ''; document.getElementById('camera-name').value = '';
      activeCameraId = payload.camera.id; await loadCameras();
    }
    async function updateCamera(id, values) {
      const response = await fetch(`/api/cameras/${encodeURIComponent(id)}`, {method: 'PATCH', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(values)});
      const payload = await response.json(); if (!response.ok) showError(payload.error || 'Could not update camera.');
      await loadCameras();
    }
    async function renameCamera(id) { const name = (window.prompt('Name for this camera (for example: Back garden feeder)') || '').trim(); if (name) await updateCamera(id, {name}); }
    async function pollCamera(id) { await fetch(`/api/cameras/${encodeURIComponent(id)}/poll`, {method: 'POST'}); await loadCameras(); }
    async function pollActiveCamera() { if (activeCameraId) await pollCamera(activeCameraId); }
    async function removeCamera(id) {
      await fetch(`/api/cameras/${encodeURIComponent(id)}`, {method: 'DELETE'});
      tabState.delete(id); if (activeCameraId === id) activeCameraId = null; await loadCameras();
    }
    function startLive() {
      const camera = activeCamera(); if (!camera || !camera.enabled) return;
      const state = stateFor(camera.id); state.liveRunning = true; updateControls(); setState('streaming', 'running');
      const message = document.getElementById('live-message'); message.className = 'running'; message.textContent = `Connecting to ${camera.name}...`;
      liveImage.src = `${apiPath('stream')}?t=${Date.now()}`;
    }
    function stopLive() {
      if (activeCameraId) stateFor(activeCameraId).liveRunning = false;
      liveImage.removeAttribute('src'); updateControls();
      const message = document.getElementById('live-message'); message.className = 'muted'; message.textContent = 'Stream stopped.';
    }
    liveImage.addEventListener('load', () => { if (activeCameraId && stateFor(activeCameraId).liveRunning) document.getElementById('live-message').textContent = 'MJPEG stream running for this tab.'; });
    liveImage.addEventListener('error', () => { if (activeCameraId && stateFor(activeCameraId).liveRunning) { stateFor(activeCameraId).liveRunning = false; updateControls(); setState('stream error', 'error'); } });
    async function scopedRequest(endpoint, options = {}) {
      const response = await fetch(apiPath(endpoint), options); const text = await response.text();
      if (!response.ok) throw new Error(text); return text;
    }
    async function refreshCameraStatus() {
      if (!activeCameraId || stateFor(activeCameraId).liveRunning) return;
      try {
        const status = JSON.parse(await scopedRequest('status', {cache: 'no-store'}));
        const runtime = status.runtime_state || status.selected_mode || 'reachable';
        setState(runtime, runtime === 'error' ? 'error' : runtime === 'motion_armed' ? 'ok' : '');
      } catch (error) { setState('unreachable', 'error'); }
    }
    async function cameraGet(endpoint) { try { const text = await scopedRequest(endpoint, {cache: 'no-store'}); showResult(text); return text; } catch (error) { showError(error); throw error; } }
    async function setCameraMode(mode) {
      try { showResult(await scopedRequest('mode', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({mode})})); await refreshCameraStatus(); }
      catch (error) { showError(error); }
    }
    async function saveCapture() {
      const button = document.getElementById('capture'); button.disabled = true;
      try {
        const payload = {calibration_type: document.getElementById('calibration-type').value, label: document.getElementById('capture-label').value, note: document.getElementById('capture-note').value};
        showResult(await scopedRequest('captures', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(payload)}));
        document.getElementById('latest').src = `${apiPath('latest.jpg')}?t=${Date.now()}`;
      } catch (error) { showError(error); } finally { button.disabled = false; }
    }
    async function saveProfile() { try { showResult(await scopedRequest('profiles', {method: 'POST'})); } catch (error) { showError(error); } }
    async function loadConfig() {
      try {
        const text = await scopedRequest('config', {cache: 'no-store'}); const config = JSON.parse(text);
        document.getElementById('config').value = JSON.stringify(config, null, 2); stateFor(activeCameraId).config = document.getElementById('config').value;
        roi = normalized(config.motion_roi); stateFor(activeCameraId).roi = {...roi}; drawRoi(); showResult(text);
      } catch (error) { showError(error); }
    }
    async function saveConfig() {
      try {
        const config = JSON.parse(document.getElementById('config').value);
        const text = await scopedRequest('config', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(config)});
        stateFor(activeCameraId).config = document.getElementById('config').value; showResult(text); await loadCameras();
      } catch (error) { showError(error); }
    }
    function drawRoi() {
      const layer = document.getElementById('roi-overlay'); const image = document.getElementById('roi-preview');
      if (!layer || !image) return; const width = Math.max(1, Math.round(image.clientWidth)); const height = Math.max(1, Math.round(image.clientHeight));
      if (layer.width !== width) layer.width = width; if (layer.height !== height) layer.height = height;
      const context = layer.getContext('2d'); context.clearRect(0, 0, width, height); context.fillStyle = 'rgba(0,0,0,.38)'; context.fillRect(0, 0, width, height);
      const x = roi.x * width, y = roi.y * height, w = roi.width * width, h = roi.height * height;
      context.clearRect(x, y, w, h); context.fillStyle = 'rgba(51,196,255,.16)'; context.fillRect(x, y, w, h); context.strokeStyle = '#33c4ff'; context.lineWidth = 3; context.strokeRect(x, y, w, h);
      document.getElementById('roi-values').textContent = `x ${roi.x.toFixed(3)}, y ${roi.y.toFixed(3)}, width ${roi.width.toFixed(3)}, height ${roi.height.toFixed(3)}`;
    }
    function roiPoint(event) { const bounds = document.getElementById('roi-overlay').getBoundingClientRect(); return {x: clamp((event.clientX - bounds.left) / Math.max(1, bounds.width)), y: clamp((event.clientY - bounds.top) / Math.max(1, bounds.height))}; }
    const roiCanvas = document.getElementById('roi-overlay');
    roiCanvas.addEventListener('pointerdown', event => { if (event.button !== undefined && event.button !== 0) return; dragStart = roiPoint(event); roiCanvas.setPointerCapture(event.pointerId); roi = {x: dragStart.x, y: dragStart.y, width: 0, height: 0}; drawRoi(); event.preventDefault(); });
    roiCanvas.addEventListener('pointermove', event => { if (!dragStart) return; const point = roiPoint(event); roi = {x: Math.min(dragStart.x, point.x), y: Math.min(dragStart.y, point.y), width: Math.abs(point.x - dragStart.x), height: Math.abs(point.y - dragStart.y)}; drawRoi(); event.preventDefault(); });
    roiCanvas.addEventListener('pointerup', event => { if (!dragStart) return; dragStart = null; stateFor(activeCameraId).roi = {...roi}; event.preventDefault(); });
    roiCanvas.addEventListener('pointercancel', () => { dragStart = null; });
    document.getElementById('roi-preview').addEventListener('load', drawRoi); window.addEventListener('resize', drawRoi);
    function resetMotionRoi() { roi = {...fullFrame}; if (activeCameraId) stateFor(activeCameraId).roi = {...roi}; drawRoi(); }
    async function loadMotionRoi() { await loadConfig(); }
    async function saveMotionRoi() { try { showResult(await scopedRequest('config', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({motion_roi: normalized(roi)})})); stateFor(activeCameraId).roi = {...roi}; } catch (error) { showError(error); } }
    function refreshRoiPreview() { if (activeCameraId) document.getElementById('roi-preview').src = `${apiPath('capture')}?t=${Date.now()}`; }
    window.addEventListener('beforeunload', stopLive);
    loadCameras().catch(showError);
    setInterval(() => { loadCameras().catch(() => {}); refreshCameraStatus().catch(() => {}); }, 5000);
  </script>
</body>
</html>
"""


PI_TABBED_DASHBOARD_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ESP32-CAM Raspberry Pi multi-camera dashboard</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #101418; color: #edf2f7; }
    header, main { width: min(1080px, calc(100% - 2rem)); margin: auto; }
    header { padding: 1.5rem 0 .75rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 1rem; padding-bottom: 1rem; }
    section { background: #192128; border: 1px solid #2d3943; border-radius: 12px; padding: 1rem; margin-bottom: 1rem; }
    img { display: block; width: 100%; min-height: 240px; object-fit: contain; background: #080a0c; border-radius: 8px; }
    button, input { padding: .6rem .75rem; margin: .25rem .25rem .25rem 0; cursor: pointer; }
    button:disabled { opacity: .55; cursor: not-allowed; }
    textarea { width: 100%; min-height: 210px; box-sizing: border-box; font: 13px monospace; }
    pre { min-height: 5rem; max-height: 22rem; overflow: auto; white-space: pre-wrap; }
    .muted { color: #aab5c0; } .ok { color: #73d99f; } .error { color: #ff8b8b; }
    .tabs { display: flex; gap: .4rem; overflow-x: auto; padding-bottom: .8rem; }
    .tab { flex: 0 0 auto; border: 1px solid #3b4954; border-radius: 999px; background: #202a32; color: inherit; }
    .tab[aria-selected="true"] { background: #14608b; border-color: #42b8f4; }
    .tab.offline { border-color: #8a3b3b; } .tab.disabled { opacity: .55; }
    .roi-stage { position: relative; width: 100%; background: #080a0c; border-radius: 8px; overflow: hidden; }
    .roi-stage img { min-height: 0; height: auto; object-fit: fill; }
    .roi-stage canvas { position: absolute; inset: 0; width: 100%; height: 100%; touch-action: none; cursor: crosshair; }
  </style>
</head>
<body>
  <header>
    <h1>ESP32-CAM Raspberry Pi dashboard</h1>
    <p class="muted">Every enabled camera has its own tab and independent Pi motion worker/background.</p>
    <div id="camera-tabs" class="tabs" role="tablist" aria-label="Registered cameras"></div>
    <h2 id="active-camera-name">No camera selected</h2>
    <p id="active-camera-details" class="muted">Add a camera below.</p>
  </header>
  <main>
    <div id="camera-workspace" class="grid" hidden>
      <section>
        <h2>Camera preview and motion ROI</h2>
        <div class="roi-stage">
          <img id="roi-preview" data-roi-preview alt="Refresh this camera preview, then drag to select a motion ROI">
          <canvas id="roi-overlay" aria-label="Drag to select a rectangular motion region"></canvas>
        </div>
        <p id="roi-values" class="muted">x 0.000, y 0.000, width 1.000, height 1.000</p>
        <button class="camera-action" onclick="refreshRoiPreview()">Refresh ROI preview</button>
        <button class="camera-action" onclick="loadMotionRoi()">Load current ROI</button>
        <button class="camera-action" onclick="resetMotionRoi()">Reset full frame</button>
        <button class="camera-action" onclick="saveMotionRoi()">Save ROI</button>
      </section>
      <section>
        <h2>Camera controls</h2>
        <button class="camera-action" onclick="setMode('motion')">Arm motion (on-board)</button>
        <button class="camera-action" onclick="setMode('idle')">Idle</button>
        <button class="camera-action" onclick="cameraGet('status')">Refresh status</button>
        <button class="camera-action" onclick="cameraGet('metadata')">Metadata</button>
        <button class="camera-action" onclick="pollActiveCamera()">Poll registry</button>
        <pre id="result">Choose an action.</pre>
      </section>
      <section>
        <h2>Configuration for this camera</h2>
        <textarea id="config">{
  "onboard_motion_detection": false,
  "motion_enabled": false
}</textarea>
        <button class="camera-action" onclick="loadConfig()">Load config</button>
        <button class="camera-action" onclick="saveConfig()">Save config</button>
        <p class="muted">Keep <code>onboard_motion_detection</code> false when the Pi/laptop workers should own motion detection.</p>
      </section>
      <section>
        <h2>Latest classified upload for this camera</h2>
        <img id="latest-image" alt="No classified upload for this camera yet">
        <pre id="inference">Waiting for this camera's first saved event.</pre>
      </section>
    </div>
    <section>
      <h2>Camera fleet</h2>
      <p class="muted">Cameras add themselves when they check in and follow their own IP changes; just give new ones a name. Adding by IP address remains as a fallback. Enabled cameras receive independent motion services.</p>
      <input id="fleet-name" placeholder="Camera name (optional)">
      <input id="fleet-address" placeholder="192.168.1.84">
      <button onclick="addFleetCamera()">Add camera</button>
      <button onclick="loadFleet()">Refresh cameras</button>
      <div id="fleet-list" class="muted">Loading cameras...</div>
    </section>
  </main>
  <script>
    'use strict';
    const fullFrame = {x: 0, y: 0, width: 1, height: 1};
    const cameraState = new Map();
    let cameras = [], activeCameraId = localStorage.getItem('esp32-pi-active-camera') || null;
    let roi = {...fullFrame}, dragStart = null;
    const result = document.getElementById('result');
    const activeCamera = () => cameras.find(camera => camera.id === activeCameraId) || null;
    function stateFor(id) { if (!cameraState.has(id)) cameraState.set(id, {roi: {...fullFrame}, config: '', result: 'Choose an action.', failed: false}); return cameraState.get(id); }
    function apiPath(endpoint = '') { if (!activeCameraId) throw new Error('Select a camera tab first.'); return `/api/cameras/${encodeURIComponent(activeCameraId)}${endpoint ? `/${endpoint}` : ''}`; }
    function pretty(text) { try { return JSON.stringify(JSON.parse(text), null, 2); } catch (_) { return text; } }
    function showCameraResult(text, failed = false) { const state = activeCameraId ? stateFor(activeCameraId) : null; if (state) { state.result = pretty(String(text)); state.failed = failed; } result.className = failed ? 'error' : 'ok'; result.textContent = pretty(String(text)); }
    window.showCameraResult = showCameraResult;
    const clamp = value => Math.max(0, Math.min(1, value));
    function normalized(value) { if (!value || typeof value !== 'object') return {...fullFrame}; const next = {}; for (const key of ['x','y','width','height']) { if (typeof value[key] !== 'number' || !Number.isFinite(value[key])) return {...fullFrame}; next[key] = clamp(value[key]); } next.width = Math.min(next.width, 1-next.x); next.height = Math.min(next.height, 1-next.y); return next; }
    function renderTabs() { const tabs = document.getElementById('camera-tabs'); tabs.replaceChildren(); cameras.forEach(camera => { const button = document.createElement('button'); button.className = `tab${camera.reachable === false ? ' offline' : ''}${camera.enabled ? '' : ' disabled'}`; button.role = 'tab'; button.ariaSelected = camera.id === activeCameraId ? 'true' : 'false'; button.textContent = `${camera.name} • motion ${camera.motion_service?.running ? 'on' : 'off'}`; button.onclick = () => selectCamera(camera.id); tabs.appendChild(button); }); }
    function renderFleet() { const list = document.getElementById('fleet-list'); list.replaceChildren(); if (!cameras.length) { list.textContent = 'No cameras registered.'; return; } cameras.forEach(camera => { const row = document.createElement('p'); const sensor = camera.sensor?.model || camera.status?.camera_sensor_model || 'unknown sensor'; row.append(`${camera.name} — ${camera.url} — ${sensor} — ${camera.enabled ? 'enabled' : 'disabled'} `); const open = document.createElement('button'); open.textContent = 'Open tab'; open.onclick = () => selectCamera(camera.id); const toggle = document.createElement('button'); toggle.textContent = camera.enabled ? 'Disable' : 'Enable'; toggle.onclick = () => updateFleetCamera(camera.id, {enabled: !camera.enabled}); const poll = document.createElement('button'); poll.textContent = 'Poll'; poll.onclick = () => pollFleetCamera(camera.id); const remove = document.createElement('button'); remove.textContent = 'Remove'; remove.onclick = () => removeFleetCamera(camera.id); const rename = document.createElement('button'); rename.textContent = camera.needs_name ? 'Name this camera' : 'Rename'; rename.onclick = () => renameFleetCamera(camera.id); row.append(rename, open, toggle, poll, remove); list.appendChild(row); }); }
    function saveVisibleState() { if (!activeCameraId) return; const state = stateFor(activeCameraId); state.roi = {...roi}; state.config = document.getElementById('config').value; }
    function renderActive() { const camera = activeCamera(); document.getElementById('camera-workspace').hidden = !camera; if (!camera) { document.getElementById('active-camera-name').textContent = 'No camera selected'; document.getElementById('active-camera-details').textContent = 'Add or enable a camera below.'; return; } const state = stateFor(camera.id); roi = normalized(state.roi); document.getElementById('config').value = state.config || '{\n  "onboard_motion_detection": false,\n  "motion_enabled": false\n}'; result.textContent = state.result; result.className = state.failed ? 'error' : ''; const sensor = camera.sensor?.model || camera.status?.camera_sensor_model || 'unknown sensor'; document.getElementById('active-camera-name').textContent = camera.name; document.getElementById('active-camera-details').textContent = `${camera.url} | ${sensor} | Pi motion ${camera.motion_service?.running ? 'running' : 'stopped'}`; document.querySelectorAll('.camera-action').forEach(button => button.disabled = !camera.enabled); document.getElementById('latest-image').src = `${apiPath('latest.jpg')}?t=${Date.now()}`; drawRoi(); refreshInference(); }
    function selectCamera(id) { if (id === activeCameraId) return; saveVisibleState(); activeCameraId = id; localStorage.setItem('esp32-pi-active-camera', id); renderTabs(); renderActive(); }
    async function loadFleet() { saveVisibleState(); const response = await fetch('/api/cameras', {cache:'no-store'}); const payload = await response.json(); if (!response.ok) throw new Error(payload.error || 'Could not load cameras.'); cameras = payload.cameras || []; if (!cameras.some(camera => camera.id === activeCameraId)) activeCameraId = (cameras.find(camera => camera.enabled) || cameras[0] || {}).id || null; renderTabs(); renderFleet(); renderActive(); }
    async function addFleetCamera() { const address = document.getElementById('fleet-address').value.trim(); const name = document.getElementById('fleet-name').value.trim() || null; const response = await fetch('/api/cameras', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({address,name})}); const payload = await response.json(); if (!response.ok) return showCameraResult(payload.error || 'Could not add camera.', true); activeCameraId = payload.camera.id; document.getElementById('fleet-address').value = ''; document.getElementById('fleet-name').value = ''; await loadFleet(); }
    async function updateFleetCamera(id, values) { const response = await fetch(`/api/cameras/${encodeURIComponent(id)}`, {method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify(values)}); const payload = await response.json(); if (!response.ok) showCameraResult(payload.error || 'Could not update camera.', true); await loadFleet(); }
    // Offer this device's clock to the hub (it has no battery clock of its own).
    fetch('/api/time', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({client_unix_ms:Date.now()})}).catch(() => {});
    async function renameFleetCamera(id) { const name = (window.prompt('Name for this camera (for example: Back garden feeder)') || '').trim(); if (name) await updateFleetCamera(id, {name}); }
    async function pollFleetCamera(id) { await fetch(`/api/cameras/${encodeURIComponent(id)}/poll`, {method:'POST'}); await loadFleet(); }
    async function pollActiveCamera() { if (activeCameraId) await pollFleetCamera(activeCameraId); }
    async function removeFleetCamera(id) { await fetch(`/api/cameras/${encodeURIComponent(id)}`, {method:'DELETE'}); cameraState.delete(id); if (activeCameraId === id) activeCameraId = null; await loadFleet(); }
    async function scopedRequest(endpoint, options = {}) { const response = await fetch(apiPath(endpoint), options); const text = await response.text(); if (!response.ok) throw new Error(text); return text; }
    async function cameraGet(endpoint) { try { const text = await scopedRequest(endpoint, {cache:'no-store'}); showCameraResult(text); return text; } catch (error) { showCameraResult(String(error), true); throw error; } }
    async function setMode(mode) { try { showCameraResult(await scopedRequest('mode', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({mode})})); } catch (error) { showCameraResult(String(error), true); } }
    async function loadConfig() { try { const text = await scopedRequest('config', {cache:'no-store'}); const config = JSON.parse(text); document.getElementById('config').value = JSON.stringify(config, null, 2); stateFor(activeCameraId).config = document.getElementById('config').value; roi = normalized(config.motion_roi); stateFor(activeCameraId).roi = {...roi}; drawRoi(); showCameraResult(text); } catch (error) { showCameraResult(String(error), true); } }
    async function saveConfig() { try { const config = JSON.parse(document.getElementById('config').value); showCameraResult(await scopedRequest('config', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(config)})); stateFor(activeCameraId).config = document.getElementById('config').value; await loadFleet(); } catch (error) { showCameraResult(String(error), true); } }
    function drawRoi() { const layer = document.getElementById('roi-overlay'), image = document.getElementById('roi-preview'); if (!layer || !image) return; const width = Math.max(1, Math.round(image.clientWidth)), height = Math.max(1, Math.round(image.clientHeight)); layer.width = width; layer.height = height; const context = layer.getContext('2d'); context.fillStyle = 'rgba(0,0,0,.38)'; context.fillRect(0,0,width,height); const x=roi.x*width,y=roi.y*height,w=roi.width*width,h=roi.height*height; context.clearRect(x,y,w,h); context.fillStyle='rgba(51,196,255,.16)'; context.fillRect(x,y,w,h); context.strokeStyle='#33c4ff'; context.lineWidth=3; context.strokeRect(x,y,w,h); document.getElementById('roi-values').textContent = `x ${roi.x.toFixed(3)}, y ${roi.y.toFixed(3)}, width ${roi.width.toFixed(3)}, height ${roi.height.toFixed(3)}`; }
    function roiPoint(event) { const bounds = document.getElementById('roi-overlay').getBoundingClientRect(); return {x:clamp((event.clientX-bounds.left)/Math.max(1,bounds.width)),y:clamp((event.clientY-bounds.top)/Math.max(1,bounds.height))}; }
    const roiCanvas = document.getElementById('roi-overlay');
    roiCanvas.addEventListener('pointerdown', event => { if (event.button !== undefined && event.button !== 0) return; dragStart=roiPoint(event); roiCanvas.setPointerCapture(event.pointerId); roi={x:dragStart.x,y:dragStart.y,width:0,height:0}; drawRoi(); event.preventDefault(); });
    roiCanvas.addEventListener('pointermove', event => { if (!dragStart) return; const point=roiPoint(event); roi={x:Math.min(dragStart.x,point.x),y:Math.min(dragStart.y,point.y),width:Math.abs(point.x-dragStart.x),height:Math.abs(point.y-dragStart.y)}; drawRoi(); event.preventDefault(); });
    roiCanvas.addEventListener('pointerup', event => { if (!dragStart) return; dragStart=null; stateFor(activeCameraId).roi={...roi}; event.preventDefault(); });
    roiCanvas.addEventListener('pointercancel', () => { dragStart=null; }); document.getElementById('roi-preview').addEventListener('load', drawRoi); window.addEventListener('resize', drawRoi);
    function resetMotionRoi() { roi={...fullFrame}; if (activeCameraId) stateFor(activeCameraId).roi={...roi}; drawRoi(); }
    async function loadMotionRoi() { await loadConfig(); }
    async function saveMotionRoi() { try { showCameraResult(await scopedRequest('config', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({motion_roi:normalized(roi)})})); stateFor(activeCameraId).roi={...roi}; } catch (error) { showCameraResult(String(error), true); } }
    function refreshRoiPreview() { if (activeCameraId) document.getElementById('roi-preview').src = `${apiPath('capture')}?t=${Date.now()}`; }
    async function refreshInference() { if (!activeCameraId) return; try { const response = await fetch(apiPath('latest'), {cache:'no-store'}); const payload = await response.json(); if (!response.ok) throw new Error(payload.error || 'No saved event yet.'); document.getElementById('latest-image').src = `${apiPath('latest.jpg')}?t=${Date.now()}`; document.getElementById('inference').textContent = JSON.stringify(payload, null, 2); } catch (error) { document.getElementById('inference').textContent = String(error); } }
    loadFleet().catch(error => showCameraResult(String(error), true)); setInterval(() => { loadFleet().catch(() => {}); refreshInference(); }, 5000);
  </script>
</body>
</html>
"""