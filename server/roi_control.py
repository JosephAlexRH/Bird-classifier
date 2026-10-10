"""Shared browser-side motion ROI editor used by both Flask dashboards."""

from __future__ import annotations


ROI_CONTROL_JS = r"""
(() => {
  'use strict';
  const fullFrame = {x: 0, y: 0, width: 1, height: 1};
  let roi = {...fullFrame};
  let dragStart = null;

  const clamp = value => Math.max(0, Math.min(1, value));
  const canvas = () => document.getElementById('roi-overlay');
  const preview = () => document.querySelector('[data-roi-preview]');

  function report(text, failed = false) {
    if (typeof window.showCameraResult === 'function') {
      window.showCameraResult(text, failed);
      return;
    }
    const output = document.getElementById('result');
    if (output) {
      output.className = failed ? 'error' : 'ok';
      output.textContent = text;
    }
  }

  function normalized(value) {
    if (!value || typeof value !== 'object') return {...fullFrame};
    const result = {};
    for (const key of ['x', 'y', 'width', 'height']) {
      if (typeof value[key] !== 'number' || !Number.isFinite(value[key])) {
        return {...fullFrame};
      }
      result[key] = clamp(value[key]);
    }
    result.width = Math.min(result.width, 1 - result.x);
    result.height = Math.min(result.height, 1 - result.y);
    return result;
  }

  function describe() {
    const output = document.getElementById('roi-values');
    if (output) {
      output.textContent = `x ${roi.x.toFixed(3)}, y ${roi.y.toFixed(3)}, ` +
        `width ${roi.width.toFixed(3)}, height ${roi.height.toFixed(3)}`;
    }
  }

  function draw() {
    const layer = canvas();
    const image = preview();
    if (!layer || !image) return;
    const width = Math.max(1, Math.round(image.clientWidth));
    const height = Math.max(1, Math.round(image.clientHeight));
    if (layer.width !== width) layer.width = width;
    if (layer.height !== height) layer.height = height;
    const context = layer.getContext('2d');
    context.clearRect(0, 0, width, height);
    context.fillStyle = 'rgba(0, 0, 0, .38)';
    context.fillRect(0, 0, width, height);
    const x = roi.x * width;
    const y = roi.y * height;
    const w = roi.width * width;
    const h = roi.height * height;
    context.clearRect(x, y, w, h);
    context.fillStyle = 'rgba(51, 196, 255, .16)';
    context.fillRect(x, y, w, h);
    context.strokeStyle = '#33c4ff';
    context.lineWidth = 3;
    context.strokeRect(x, y, w, h);
    describe();
  }

  function point(event) {
    const bounds = canvas().getBoundingClientRect();
    return {
      x: clamp((event.clientX - bounds.left) / Math.max(1, bounds.width)),
      y: clamp((event.clientY - bounds.top) / Math.max(1, bounds.height))
    };
  }

  function start(event) {
    if (event.button !== undefined && event.button !== 0) return;
    dragStart = point(event);
    canvas().setPointerCapture(event.pointerId);
    roi = {x: dragStart.x, y: dragStart.y, width: 0, height: 0};
    draw();
    event.preventDefault();
  }

  function move(event) {
    if (!dragStart) return;
    const current = point(event);
    roi = {
      x: Math.min(dragStart.x, current.x),
      y: Math.min(dragStart.y, current.y),
      width: Math.abs(current.x - dragStart.x),
      height: Math.abs(current.y - dragStart.y)
    };
    draw();
    event.preventDefault();
  }

  function finish(event) {
    if (!dragStart) return;
    move(event);
    dragStart = null;
    event.preventDefault();
  }

  window.resetMotionRoi = () => {
    roi = {...fullFrame};
    draw();
  };

  window.loadMotionRoi = async () => {
    try {
      const response = await fetch('/camera/config', {cache: 'no-store'});
      const text = await response.text();
      if (!response.ok) throw new Error(text);
      const config = JSON.parse(text);
      roi = normalized(config.motion_roi);
      draw();
      const editor = document.getElementById('config');
      if (editor) editor.value = JSON.stringify(config, null, 2);
      report(text);
    } catch (error) {
      report(String(error), true);
    }
  };

  window.saveMotionRoi = async () => {
    try {
      const payload = {motion_roi: normalized(roi)};
      const response = await fetch('/camera/config', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      const text = await response.text();
      if (!response.ok) throw new Error(text);
      report(text);
    } catch (error) {
      report(String(error), true);
    }
  };

  window.refreshRoiPreview = () => {
    const image = preview();
    if (image) image.src = `/camera/capture?t=${Date.now()}`;
  };

  function initialize() {
    const layer = canvas();
    const image = preview();
    if (!layer || !image) return;
    layer.addEventListener('pointerdown', start);
    layer.addEventListener('pointermove', move);
    layer.addEventListener('pointerup', finish);
    layer.addEventListener('pointercancel', () => { dragStart = null; });
    image.addEventListener('load', draw);
    window.addEventListener('resize', draw);
    draw();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize);
  } else {
    initialize();
  }
})();
"""