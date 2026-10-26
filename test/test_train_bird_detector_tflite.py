from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from Models.Multi_Object_Detector.train_bird_detector_tflite import (
    CONFIDENCE_GRID,
    MAX_DETECTIONS,
    TARGET_BIRD_COUNT,
    TARGET_OBJECTNESS,
    RunPaths,
    boxes_in_crop,
    configuration_fingerprint_payload,
    encode_targets_numpy,
    is_native_crop,
    make_crop_records,
    select_representative_records,
    select_datasets,
    write_decoder_metadata,
)


class MotionCropDetectorTrainingTests(unittest.TestCase):
    def scene(
        self,
        sample_id: str,
        width: int,
        height: int,
        boxes: list[list[float]],
        split: str = "test",
        eligible: bool = False,
    ) -> dict:
        return {
            "sample_id": sample_id,
            "source_dataset": "esp32_capture",
            "image_path": f"{sample_id}.jpg",
            "width": width,
            "height": height,
            "boxes": np.asarray(boxes, np.float32).reshape((-1, 4)),
            "model_split": split,
            "eligible_for_training": eligible,
            "annotation_trust": "human_verified" if boxes else "verified_negative",
            "scene_geometry": "native_crop" if is_native_crop(width, height) else "full_scene",
        }

    def test_native_160x120_is_excluded_from_all_detector_datasets(self) -> None:
        scenes = [
            self.scene("native", 160, 120, [[10, 10, 140, 100]], split="train", eligible=True),
            self.scene("native-validation", 160, 120, [[10, 10, 140, 100]], split="validation"),
            self.scene("native-test", 160, 120, [[10, 10, 140, 100]], split="test"),
            self.scene("scene", 640, 480, [[200, 100, 260, 180]]),
        ]
        datasets = select_datasets(scenes)
        self.assertFalse(
            any(record["scene_geometry"] == "native_crop" for records in datasets.values() for record in records)
        )
        self.assertNotIn("validation_full_scene", datasets)
        self.assertNotIn("test_full_scene", datasets)

    def test_full_scene_bird_becomes_motion_crop_with_local_box(self) -> None:
        records = make_crop_records(
            [self.scene("scene", 640, 480, [[300, 200, 340, 240]])],
            negative_crops_per_scene=1,
        )
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["evaluation_geometry"], "simulated_motion_crop")
        self.assertEqual((record["width"], record["height"]), (160, 160))
        np.testing.assert_allclose(record["boxes"], [[60, 60, 100, 100]])

    def test_scene_union_evaluation_uses_one_crop_and_counts_each_bird_once(self) -> None:
        records = make_crop_records(
            [self.scene("scene", 640, 480, [[100, 100, 140, 140], [180, 120, 220, 160]])],
            negative_crops_per_scene=1,
            positive_crop_mode="scene_union",
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(len(records[0]["boxes"]), 2)

    def test_boxes_in_crop_keeps_birds_whose_centres_are_inside(self) -> None:
        boxes = np.asarray([[0, 0, 20, 20], [90, 90, 129, 129]], np.float32)
        result = boxes_in_crop(boxes, (10, 10, 110, 110))
        np.testing.assert_allclose(result, [[0, 0, 10, 10], [80, 80, 100, 100]])

    def test_one_bird_is_positive_in_nine_cells_but_counted_once(self) -> None:
        anchors = np.asarray([[0.1, 0.1], [0.2, 0.2], [0.4, 0.4]], np.float32)
        target = encode_targets_numpy(np.asarray([[0.45, 0.45, 0.55, 0.55]], np.float32), anchors, 8)
        self.assertEqual(int(target[..., TARGET_OBJECTNESS].sum()), 9)
        self.assertTrue(np.all(target[..., TARGET_BIRD_COUNT] == 1))

    def test_edge_bird_uses_only_valid_neighbour_cells(self) -> None:
        anchors = np.asarray([[0.1, 0.1], [0.2, 0.2], [0.4, 0.4]], np.float32)
        target = encode_targets_numpy(np.asarray([[0.0, 0.0, 0.1, 0.1]], np.float32), anchors, 8)
        self.assertEqual(int(target[..., TARGET_OBJECTNESS].sum()), 4)

    def test_two_birds_share_cells_without_overwriting_unique_count(self) -> None:
        anchors = np.asarray([[0.08, 0.08], [0.12, 0.12], [0.2, 0.2]], np.float32)
        boxes = np.asarray([[0.45, 0.45, 0.53, 0.53], [0.46, 0.46, 0.58, 0.58]], np.float32)
        target = encode_targets_numpy(boxes, anchors, 8)
        self.assertEqual(int(target[..., TARGET_OBJECTNESS].sum()), 18)
        self.assertTrue(np.all(target[..., TARGET_BIRD_COUNT] == 2))

    def test_calibration_prefers_balanced_simulated_motion_crops(self) -> None:
        records = []
        for index in range(8):
            records.append({
                **self.scene(f"native-{index}", 160, 120, [[10, 10, 100, 100]], split="train", eligible=True),
                "sample_id": f"native-{index}::native",
                "evaluation_geometry": "native_crop",
            })
        for index in range(4):
            records.append({
                **self.scene(f"positive-{index}", 160, 160, [[20, 20, 80, 80]], split="train", eligible=True),
                "sample_id": f"positive-{index}::motion",
                "evaluation_geometry": "simulated_motion_crop",
            })
            records.append({
                **self.scene(f"negative-{index}", 160, 160, [], split="train", eligible=True),
                "sample_id": f"negative-{index}::motion",
                "evaluation_geometry": "simulated_motion_crop",
            })
        selected = select_representative_records(records, count=6)
        self.assertTrue(all(record["evaluation_geometry"] == "simulated_motion_crop" for record in selected))
        self.assertEqual(sum(bool(len(record["boxes"])) for record in selected), 3)
        self.assertEqual(sum(not len(record["boxes"]) for record in selected), 3)

    def test_decoder_metadata_declares_selected_motion_crop_geometry(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            export = root / "exports"
            export.mkdir()
            uint8_model = export / "bird_detector_motion_crop_uint8.tflite"
            uint8_model.write_bytes(b"test-model")
            paths = RunPaths(
                source_run=root,
                artifact_dir=root / "artifacts",
                checkpoint_dir=root / "checkpoints",
                export_dir=export,
                head_weights=root / "head.weights.h5",
                final_weights=root / "final.weights.h5",
                anchors=root / "anchors.json",
                float_tflite=export / "bird_detector_motion_crop_float32.tflite",
                uint8_tflite=uint8_model,
            )
            details = {
                "uint8": {
                    "inputs": [{"name": "image", "shape": [1, 288, 288, 3], "dtype": "uint8", "quantization": [1.0, 0]}],
                    "outputs": [{"name": "raw", "shape": [1, 36, 36, 3, 5], "dtype": "uint8", "quantization": [0.1, 128]}],
                    "operations": ["CONV_2D"],
                }
            }
            metadata_path = write_decoder_metadata(
                paths,
                details,
                np.asarray([[0.1, 0.1], [0.2, 0.2], [0.4, 0.4]], np.float32),
                0.25,
            )
            metadata = __import__("json").loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["input"]["geometry"], "selected_crop_then_letterbox")
            self.assertEqual(metadata["input"]["crop_source"], "capture_motion_record.selection.final_bounds_px")
            self.assertEqual(metadata["postprocess"]["deployment_score_threshold"], 0.25)

    def test_threshold_grid_is_fine_above_point_nine_and_crop_detection_is_capped(self) -> None:
        high = [value for value in CONFIDENCE_GRID if value >= 0.9]
        self.assertEqual(high[0], 0.9)
        self.assertEqual(high[-1], 0.995)
        np.testing.assert_allclose(np.diff(high), 0.005)
        self.assertEqual(MAX_DETECTIONS, 5)

    def test_configuration_fingerprint_covers_loss_and_threshold_policy(self) -> None:
        payload = configuration_fingerprint_payload("manifest-hash", 288)
        self.assertEqual(payload["positive_objectness_weight"], 0.25)
        self.assertEqual(payload["negative_objectness_weight"], 0.75)
        self.assertEqual(payload["objectness_normalizer"], "unique_birds_in_batch")
        self.assertTrue(payload["native_160x120_excluded"])
        self.assertEqual(payload["max_detections"], 5)
        self.assertIn(0.995, payload["confidence_grid"])


if __name__ == "__main__":
    unittest.main()