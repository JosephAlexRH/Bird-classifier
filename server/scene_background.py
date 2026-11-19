"""Explicit empty-scene calibration and conservative background adaptation."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import uuid
import time
from datetime import datetime, timezone

import numpy as np

try:
    from .motion_crop import RunningMedianBackground, normalized_roi_bounds, _blur, _binary_morphology, _components
except ImportError:
    from motion_crop import RunningMedianBackground, normalized_roi_bounds, _blur, _binary_morphology, _components


class SceneBackground(RunningMedianBackground):
    required_samples = 5
    maximum_learning_samples = 15

    def __init__(self, config, *, camera_id, cache_path: Path):
        super().__init__(config, camera_id=camera_id)
        self.cache_path = cache_path
        self.signature = None
        self.phase = 'warming'
        self.reason = 'waiting_for_confident_empty_frames_or_manual_learning'
        self.origin = 'automatic_confident_empty'
        self.session = None
        self.evidence = []
        self.learning_frames = []
        self.reference_frame = None
        self.broad_changes = 0
        self.last_learning_at = None
        self.pixel_thresholds = None
        self.learning_diagnostics = {}
        self.learning_attempts = 0
        self.noise_map_sha256 = None
        self.event_log_error = None

    def state(self):
        return {'phase': self.phase, 'reason': self.reason, 'origin': self.origin,
                'ready': self.ready, 'samples': self.sample_count,
                'learning_samples': len(self.learning_frames), 'required_samples': self.required_samples,
                'session_id': self.session, 'evidence_capture_ids': list(self.evidence),
                'learning_attempts': self.learning_attempts, 'maximum_learning_samples': self.maximum_learning_samples,
                'calibration_method': 'temporal_median_noise_v2' if self.pixel_thresholds is not None else 'legacy_or_automatic',
                'noise_map_sha256': self.noise_map_sha256,
                'event_log_error': self.event_log_error,
                'diagnostics': dict(self.learning_diagnostics)}

    def record_transition(self, event):
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            entry = {'time_utc': datetime.now(timezone.utc).isoformat(), 'event': event,
                     'camera_id': self.camera_id, 'scene_signature': self.signature, **self.state()}
            with self.cache_path.with_suffix('.events.jsonl').open('a', encoding='utf-8') as output:
                output.write(json.dumps(entry) + '\n')
            self.event_log_error = None
        except OSError as error:
            self.event_log_error = str(error)

    def configure_scene(self, signature):
        shape = tuple(signature[2])
        signature = json.dumps(signature, sort_keys=True, separators=(',', ':'))
        if self.signature == signature:
            return
        if self.signature is not None:
            self.invalidate('camera_boot_geometry_roi_or_settings_changed')
        else:
            try:
                with np.load(self.cache_path, allow_pickle=False) as saved:
                    info = json.loads(str(saved['metadata'].item()))
                    frames = saved['frames']
                    if (info['signature'] == signature and frames.dtype == np.uint8 and frames.ndim == 3
                            and tuple(frames.shape[1:]) == shape
                            and self.config.minimum_background_frames <= len(frames) <= self.config.background_frames):
                        for frame in frames:
                            self.admit(frame, force=True)
                        self.reference_frame = self.median().copy() if self.ready else None
                        self.origin = info['origin']
                        self.session = info.get('session_id')
                        self.evidence = info.get('evidence_capture_ids', [])
                        if 'pixel_thresholds' in saved.files:
                            thresholds = saved['pixel_thresholds']
                            if thresholds.shape != shape or thresholds.dtype != np.uint8:
                                raise ValueError('Invalid reference noise map')
                            self.pixel_thresholds = thresholds.copy()
                            self.noise_map_sha256 = hashlib.sha256(thresholds.tobytes()).hexdigest()
                        self.learning_diagnostics = info.get('diagnostics', {})
                        self.phase, self.reason = 'ready', 'restored_matching_reference'
                    else:
                        self.invalidate('stored_reference_does_not_match_camera_scene')
            except FileNotFoundError:
                pass
            except (OSError, ValueError, KeyError, TypeError, EOFError):
                self.invalidate('stored_reference_unusable')
        self.signature = signature

    def invalidate(self, reason):
        self.reset()
        self.phase, self.reason = 'needs_learning', reason
        self.learning_frames.clear()
        self.reference_frame = None
        self.pixel_thresholds = None
        self.noise_map_sha256 = None
        self.broad_changes = 0
        self.origin = 'automatic_confident_empty'
        self.cache_path.unlink(missing_ok=True)
        self.record_transition('reference_invalidated')

    def begin_learning(self):
        self.invalidate('user_confirmed_empty_scene')
        self.phase = 'learning'
        self.origin = 'user_confirmed_empty_scene'
        self.session = uuid.uuid4().hex
        self.evidence = []
        self.last_learning_at = None
        self.learning_attempts = 0
        self.learning_diagnostics = {}
        self.record_transition('learning_started')

    def learn(self, gray, roi, capture_id, *, now=None):
        now = time.monotonic() if now is None else now
        if self.last_learning_at is not None and now - self.last_learning_at < 1:
            return
        self.last_learning_at = now
        self.evidence.append(capture_id)
        self.learning_attempts += 1
        self.learning_frames.append(gray.copy())
        self.learning_frames = self.learning_frames[-self.required_samples:]
        if len(self.learning_frames) < self.required_samples:
            return
        left, top, right, bottom = normalized_roi_bounds(roi, gray.shape[1], gray.shape[0])
        # Assess the complete temporal window, rather than rejecting the second
        # JPEG against a possibly noisy/poorly exposed first image.
        stack = np.stack(self.learning_frames).astype(np.float32)
        levels = np.median(stack[:, top:bottom, left:right], axis=(1, 2))
        shifts = levels - levels[-1]
        normalized = np.clip(stack - shifts[:, None, None], 0, 255).astype(np.uint8)
        blurred = np.stack([_blur(frame) for frame in normalized]).astype(np.float32)
        center = np.median(blurred, axis=0)
        noise = np.percentile(np.abs(blurred - center), 90, axis=0)
        roi_noise = noise[top:bottom, left:right]
        changed = _binary_morphology(roi_noise >= self.config.minimum_threshold)
        components = _components(changed)
        # Thin swaying feeder edges/foliage are different from a substantial
        # compact foreground object. Do not teach that object as normal noise.
        compact_area = max(self.config.minimum_component_pixels, int(roi_noise.size * .005))
        compact = [item for item in components if item['area'] >= compact_area
                   and min(item['right']-item['left'], item['bottom']-item['top']) /
                       max(item['right']-item['left'], item['bottom']-item['top']) >= .35
                   and item['area'] / ((item['right']-item['left']) * (item['bottom']-item['top'])) >= .5]
        fraction = float(np.mean(roi_noise >= self.config.minimum_threshold))
        exposure_span = float(np.max(levels) - np.min(levels))
        self.learning_diagnostics = {'variable_fraction': fraction, 'allowed_variable_fraction': .15,
            'exposure_span': exposure_span, 'allowed_exposure_span': 40,
            'largest_compact_change_pixels': max((item['area'] for item in compact), default=0),
            'noise_percentile_95': float(np.percentile(roi_noise, 95))}
        reason = ('exposure_still_settling' if exposure_span > 40 else
                  'large_foreground_movement' if compact else
                  'too_much_scene_variation' if fraction > .15 else None)
        if reason:
            self.reason = 'waiting_for_stable_window_' + reason
            if reason == 'large_foreground_movement' or self.learning_attempts >= self.maximum_learning_samples:
                self.phase, self.reason = 'failed', 'calibration_' + reason
                self.record_transition('learning_failed')
            return
        for frame in normalized:
            self.admit(frame, force=True)
        self.reference_frame = self.median().copy()
        # Per-pixel thresholds account for the observed background noise. No
        # pixels are silently masked out; substantial changes can still trigger.
        self.pixel_thresholds = np.clip(np.ceil(noise + 8), self.config.minimum_threshold,
                                        max(96, self.config.minimum_threshold)).astype(np.uint8)
        self.noise_map_sha256 = hashlib.sha256(self.pixel_thresholds.tobytes()).hexdigest()
        self.phase, self.reason = 'ready', 'manual_empty_scene_calibrated'
        self.save_reference()
        self.record_transition('learning_completed')
        self.learning_frames.clear()

    def save_reference(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        info = {**self.state(), 'signature': self.signature,
                'note': 'Calibration assertion is not a per-image training label.'}
        temporary = self.cache_path.with_suffix('.tmp')
        try:
            with temporary.open('wb') as stream:
                arrays = {'frames': np.stack(tuple(self._frames)), 'metadata': json.dumps(info)}
                if self.pixel_thresholds is not None:
                    arrays['pixel_thresholds'] = self.pixel_thresholds
                np.savez_compressed(stream, **arrays)
            temporary.replace(self.cache_path)
        finally:
            temporary.unlink(missing_ok=True)
        # Human-readable provenance is kept beside the exact reference samples.
        self.cache_path.with_suffix('.json').write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')

    def adapt(self, gray, motion, now):
        reason = motion['detection'].get('reason')
        shift = float(motion['detection'].get('global_shift', 0))
        if reason in ('broad_scene_change', 'excessive_noise') or abs(shift) > 32:
            self.broad_changes += 1
            if self.broad_changes >= 3:
                # Broad change can be a bird occupying the ROI. Preserve the
                # empty reference and freeze learning until the view is quiet.
                if self.phase != 'attention':
                    self.phase, self.reason = 'attention', 'large_subject_or_scene_change_reference_frozen'
                    self.record_transition('reference_frozen')
            return False
        self.broad_changes = 0
        if reason != 'no_changed_blob' or not self.ready:
            return False
        if self.phase == 'attention':
            self.phase, self.reason = 'ready', 'quiet_scene_recovered_reference_preserved'
            self.record_transition('reference_recovered')
        background = self.median().astype(np.float32)
        delta = gray.astype(np.float32) - background - shift
        # Foreground pixels are frozen, even if morphology discarded a small
        # component. Slow local drift is bounded against the calibrated scene.
        quiet = np.abs(delta) < self.config.minimum_threshold / 2
        adapted = background + shift
        if self.reference_frame is not None:
            anchor = self.reference_frame.astype(np.float32)
            global_total = float(np.median(background - anchor))
            step = np.where(quiet, np.clip(delta, -1, 1), 0)
            local = np.clip(background + step - anchor - global_total, -6, 6)
            adapted = anchor + global_total + shift + local
        return self.admit(np.clip(adapted, 0, 255).astype(np.uint8), now=now)
