import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from server.capture_storage import CaptureStore
from server.capture_storage import capture_files
from server.motion_crop import MotionConfig
from server.pi_motion import MotionService
from server.scene_background import SceneBackground
from test.test_pi_motion import jpeg, metadata, FakeEngine


class SceneBackgroundTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.now = 0
        self.frame = np.full((120, 160, 3), 60, np.uint8)
        self.header = metadata()
        self.engine = FakeEngine([{'status': 'complete', 'is_bird': True, 'bird_probability': .55}] * 30)
        self.engine.bird_threshold = .5
        self.service = self.make_service()

    def tearDown(self):
        self.temp.cleanup()

    def make_service(self, **kwargs):
        return MotionService('http://camera', CaptureStore(self.root), self.engine,
            camera_registry_id='camera-one', config=MotionConfig(admission_seconds=0),
            fetcher=lambda *_a, **_k: (jpeg(self.frame), self.header),
            monotonic=lambda: self.now, motion_triggered=True, **kwargs)

    def poll(self):
        self.now += 2
        return self.service.process_once()

    def calibrate(self):
        self.poll()
        self.service.background_command('learn', confirmed_empty=True)
        for _ in range(5):
            self.poll()
        self.assertTrue(self.service.background.ready)

    def test_manual_learning_bypasses_feeder_scores_and_keeps_evidence_unreviewed(self):
        initial = self.poll()
        self.assertTrue(initial['persisted'])
        self.assertTrue(initial['motion']['capture_policy']['detector_fallback_no_background'])
        with self.assertRaises(ValueError):
            self.service.background_command('learn')
        self.service.background_command('learn', confirmed_empty=True)
        for _ in range(5):
            self.poll()
        self.assertTrue(self.service.background.ready)
        self.assertEqual(len(self.engine.calls), 1)
        self.assertEqual(len(capture_files(self.root)), 6)
        self.assertEqual(len(self.service.background.evidence), 5)
        for record in capture_files(self.root, '*.inference.json'):
            value = json.loads(record.read_text())
            if value.get('status') == 'complete':
                continue
            self.assertEqual(value['reason'], 'background_calibration')
            self.assertIsNone(value['is_bird'])
        self.assertEqual(capture_files(self.root, '*.review.json'), [])

    def test_quiet_scene_skips_detector_and_saving_after_calibration(self):
        self.calibrate()
        before = len(self.engine.calls)
        for _ in range(8):
            result = self.poll()
            self.assertFalse(result['persisted'])
            self.assertEqual(result['inference']['reason'], 'no_motion')
        self.assertEqual(len(self.engine.calls), before)

    def test_local_movement_runs_detector_and_stationary_subject_not_absorbed(self):
        self.calibrate()
        baseline = self.service.background.median().copy()
        self.frame[40:65, 60:85] = 200
        for _ in range(8):
            result = self.poll()
            self.assertTrue(result['persisted'])
            self.assertFalse(result['background_admitted'])
        np.testing.assert_array_equal(self.service.background.median(), baseline)

    def test_uniform_lighting_shift_is_not_a_bird_event(self):
        self.calibrate()
        self.frame[:] = 70
        for _ in range(6):
            self.assertFalse(self.poll()['persisted'])
        self.assertTrue(self.service.background.ready)

    def test_reference_restores_after_worker_restart_and_roi_change_invalidates(self):
        self.calibrate()
        self.service = self.make_service()
        self.assertFalse(self.poll()['persisted'])
        self.assertEqual(self.service.background.reason, 'restored_matching_reference')
        value = json.loads(metadata())
        value['camera']['requested']['motion']['roi'] = {'x': 0, 'y': 0, 'width': .5, 'height': 1}
        self.header = json.dumps(value)
        self.assertTrue(self.poll()['persisted'])
        self.assertFalse(self.service.background.ready)
        self.assertIn('changed', self.service.background.reason)

    def test_calibration_motion_rejected_without_learning_subject(self):
        self.poll()
        self.service.background_command('learn', confirmed_empty=True)
        self.poll()
        self.frame[20:60, 20:60] = 200
        for _ in range(4):
            self.poll()
        self.assertEqual(self.service.background.phase, 'failed')
        self.assertFalse(self.service.background.ready)
        self.assertFalse(self.service.background.cache_path.exists())

    def test_periodic_sample_is_unreviewed_not_confirmed_negative(self):
        self.service = self.make_service(empty_sample_seconds=30)
        self.calibrate()
        self.now += 30
        result = self.poll()
        self.assertTrue(result['persisted'])
        self.assertTrue(result['motion']['capture_policy']['periodic_background_sample'])
        self.assertEqual(result['capture']['capture_policy']['training_label'], 'unreviewed')
        self.assertEqual(capture_files(self.root, '*.review.json'), [])

    def test_other_camera_cannot_restore_reference_and_reset_is_explicit(self):
        self.calibrate()
        self.service.background_command('reset')
        self.assertFalse(self.service.background.ready)
        self.assertEqual(self.service.background.reason, 'user_reset')
        self.assertFalse(self.service.background.cache_path.exists())
        self.assertGreater(len(capture_files(self.root)), 0)

    def test_background_command_rejects_concurrent_analysis(self):
        self.poll()
        with self.service._process_lock:
            with self.assertRaises(RuntimeError):
                self.service.background_command('learn', confirmed_empty=True)

    def test_large_subject_preserves_reference_and_runs_detector(self):
        self.calibrate()
        baseline = self.service.background.median().copy()
        self.frame[10:110, 10:150] = 140
        for _ in range(3):
            result = self.poll()
            self.assertTrue(result['persisted'])
            self.assertTrue(result['motion']['capture_policy']['uncertain_scene'])
        self.assertTrue(self.service.background.ready)
        self.assertTrue(self.service.background.cache_path.exists())
        self.assertEqual(self.service.background.phase, 'attention')
        np.testing.assert_array_equal(self.service.background.median(), baseline)
        self.frame[:] = 60
        self.assertFalse(self.poll()['persisted'])
        self.assertEqual(self.service.background.phase, 'ready')
        history = self.service.background.cache_path.with_suffix('.events.jsonl').read_text()
        self.assertIn('reference_frozen', history)
        self.assertIn('reference_recovered', history)

    def test_missing_reference_preserves_detected_bird_and_logs_fallback(self):
        result = self.poll()
        self.assertTrue(result['persisted'])
        self.assertTrue(result['capture']['capture_policy']['detector_fallback_no_background'])
        self.assertEqual(self.service.describe()['capture_policy_state'], 'detector_fallback_no_background')
        self.assertFalse(self.service.background.ready)

    def test_swaying_thin_feeder_edges_calibrate_and_do_not_trigger_events(self):
        self.poll()
        self.service.background_command('learn', confirmed_empty=True)
        for i in range(5):
            self.frame[:] = 60
            self.frame[10:105, 70+i%2:76+i%2] = 190
            self.poll()
        self.assertTrue(self.service.background.ready)
        self.assertIsNotNone(self.service.background.pixel_thresholds)
        for i in range(6):
            self.frame[:] = 60
            self.frame[10:105, 70+i%2:76+i%2] = 190
            self.assertFalse(self.poll()['persisted'])
        self.frame[50:75, 110:135] = 200
        result = self.poll()
        self.assertTrue(result['persisted'])
        self.assertTrue(result['motion']['capture_policy']['motion_candidate'])

    def test_exposure_can_settle_instead_of_failing_on_second_frame(self):
        self.poll()
        self.service.background_command('learn', confirmed_empty=True)
        for level in (30, 110, 80, 100, 100):
            self.frame[:] = level
            self.poll()
        self.assertEqual(self.service.background.phase, 'learning')
        for _ in range(4):
            self.frame[:] = 100
            self.poll()
        self.assertTrue(self.service.background.ready)

    def test_persistent_exposure_instability_times_out_with_diagnostics(self):
        self.poll()
        self.service.background_command('learn', confirmed_empty=True)
        for i in range(15):
            self.frame[:] = 20 if i%2 else 180
            self.poll()
        self.assertFalse(self.service.background.ready)
        self.assertEqual(self.service.background.phase, 'failed')
        self.assertEqual(self.service.background.reason, 'calibration_exposure_still_settling')
        self.assertGreater(self.service.background.learning_diagnostics['exposure_span'], 40)

    def test_calibrated_noise_map_restores_with_reference(self):
        self.calibrate()
        thresholds = self.service.background.pixel_thresholds.copy()
        self.service = self.make_service()
        self.poll()
        np.testing.assert_array_equal(self.service.background.pixel_thresholds, thresholds)


if __name__ == '__main__':
    unittest.main()
