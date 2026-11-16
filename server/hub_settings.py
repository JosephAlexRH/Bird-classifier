"""Validated, persistent hub controls, independent of firmware motion mode."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

try:
    from .collection import write_review_record
except ImportError:
    from collection import write_review_record


class HubSettings:
    BOOL_FIELDS = {"capture_enabled", "collection_mode", "detector_enabled", "species_enabled", "motion_localization_enabled", "motion_triggered"}
    NUMERIC_LIMITS = {'poll_seconds': (.25, 3600), 'detector_threshold': (0, 1),
                      'motion_threshold': (1, 128), 'motion_minimum_area': (1, 1000000),
                      'empty_sample_seconds': (0, 86400)}

    def __init__(self, path: Path, defaults: dict):
        self.path = path
        self.value = {"schema_version": 1, "revision": 0, "capture_enabled": True,
            "collection_mode": False, "detector_enabled": True, "species_enabled": False,
            "motion_localization_enabled": True, "poll_seconds": 2.0,
            "detector_threshold": .5, "per_camera": {}, **defaults}
        self.value.update({'motion_triggered': True, 'motion_threshold': 20,
                           'motion_minimum_area': 100, 'empty_sample_seconds': 300})
        self.value.update(defaults)
        if path.is_file():
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict) or loaded.get("schema_version") != 1:
                raise ValueError(f"Unsupported hub settings file: {path}")
            patch = {k: v for k, v in loaded.items() if k not in {"schema_version", "revision", "per_camera"}}
            proposed = self.propose(patch, validate_combination=False)
            cameras = loaded.get("per_camera", {})
            if not isinstance(cameras, dict) or any(not isinstance(k, str) or not isinstance(v, dict)
                or set(v) - {"capture_enabled"} or type(v.get("capture_enabled")) is not bool for k, v in cameras.items()):
                raise ValueError("Invalid per-camera capture settings")
            revision = loaded.get("revision", 0)
            if type(revision) is not int or revision < 0:
                raise ValueError("Invalid hub settings revision")
            proposed.update(revision=revision, per_camera=cameras)
            self.value = proposed

    def snapshot(self):
        return copy.deepcopy(self.value)

    def propose(self, patch, *, camera_id=None, validate_combination=True):
        if not isinstance(patch, dict) or not patch:
            raise ValueError("Supply settings to change.")
        result = self.snapshot()
        if camera_id is not None:
            if set(patch) != {"capture_enabled"} or type(patch["capture_enabled"]) is not bool:
                raise ValueError("Per-camera settings accept capture_enabled: true or false.")
            result["per_camera"][camera_id] = dict(patch)
        else:
            allowed = self.BOOL_FIELDS | set(self.NUMERIC_LIMITS)
            if set(patch) - allowed:
                raise ValueError("Unknown hub settings: " + ", ".join(sorted(set(patch) - allowed)))
            for key, value in patch.items():
                if key in self.BOOL_FIELDS:
                    if type(value) is not bool:
                        raise ValueError(f"{key} must be true or false.")
                else:
                    low, high = self.NUMERIC_LIMITS[key]
                    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                        raise ValueError(f"{key} must be between {low} and {high}.")
                    if key in ('motion_threshold', 'motion_minimum_area') and int(value) != value:
                        raise ValueError(f'{key} must be a whole number.')
                result[key] = value
        if validate_combination:
            if result['motion_triggered'] and not result['collection_mode'] and not result['motion_localization_enabled']:
                raise ValueError('Enable Pi motion localization to require motion, or turn off Require motion before bird detection.')
            if result["species_enabled"] and not result["detector_enabled"]:
                raise ValueError("Enable bird detection before enabling species inference.")
            if not result["collection_mode"] and not result["detector_enabled"]:
                raise ValueError("Choose Save all originals before turning analysis off, so captures can be retained.")
        result["revision"] += 1
        return result

    def save(self, proposed):
        write_review_record(self.path, proposed)
        self.value = copy.deepcopy(proposed)

    def camera_running(self, camera_id):
        return self.value["capture_enabled"] and self.value["per_camera"].get(camera_id, {}).get("capture_enabled", True)
