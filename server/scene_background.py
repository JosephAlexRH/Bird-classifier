"""Explicit empty-scene calibration and conservative background adaptation."""
from __future__ import annotations

import json
from pathlib import Path
import uuid
import time

import numpy as np

try:
    from .motion_crop import RunningMedianBackground, normalized_roi_bounds
except ImportError:
    from motion_crop import RunningMedianBackground, normalized_roi_bounds


class SceneBackground(RunningMedianBackground):
    required_samples = 5

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

    def state(self):
        return {'phase': self.phase, 'reason': self.reason, 'origin': self.origin,
                'ready': self.ready, 'samples': self.sample_count,
                'learning_samples': len(self.learning_frames), 'required_samples': self.required_samples,
                'session_id': self.session, 'evidence_capture_ids': list(self.evidence)}

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
        self.broad_changes = 0
        self.origin = 'automatic_confident_empty'
        self.cache_path.unlink(missing_ok=True)

    def begin_learning(self):
        self.invalidate('user_confirmed_empty_scene')
        self.phase = 'learning'
        self.origin = 'user_confirmed_empty_scene'
        self.session = uuid.uuid4().hex
        self.evidence = []
        self.last_learning_at = None

    def learn(self, gray, roi, capture_id, *, now=None):
        now = time.monotonic() if now is None else now
        if self.last_learning_at is not None and now - self.last_learning_at < 1:
            return
        self.last_learning_at = now
        self.evidence.append(capture_id)
        if self.learning_frames:
            first = self.learning_frames[0]
            left, top, right, bottom = normalized_roi_bounds(roi, gray.shape[1], gray.shape[0])
            difference = gray[top:bottom, left:right].astype(np.int16) - first[top:bottom, left:right].astype(np.int16)
            shift = float(np.median(difference))
            fraction = float(np.mean(np.abs(difference - shift) >= self.config.minimum_threshold))
            if abs(shift) > 12 or fraction > .01:
                self.invalidate('calibration_unstable_check_for_birds_motion_or_exposure_changes')
                self.phase = 'failed'
                return
        self.learning_frames.append(gray.copy())
        if len(self.learning_frames) >= self.required_samples:
            for frame in self.learning_frames:
                self.admit(frame, force=True)
            self.reference_frame = self.median().copy()
            self.phase, self.reason = 'ready', 'manual_empty_scene_calibrated'
            self.save_reference()
            self.learning_frames.clear()

    def save_reference(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        info = {**self.state(), 'signature': self.signature,
                'note': 'Calibration assertion is not a per-image training label.'}
        temporary = self.cache_path.with_suffix('.tmp')
        try:
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, frames=np.stack(tuple(self._frames)), metadata=json.dumps(info))
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
                self.invalidate('persistent_scene_or_lighting_change_relearn_empty_scene')
            return False
        self.broad_changes = 0
        if reason != 'no_changed_blob' or not self.ready:
            return False
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
