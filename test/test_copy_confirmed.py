"""Tests for copying manually confirmed captures without a manifest."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import copy_confirmed


class CopyConfirmedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.destination = self.root / "destination"
        self.source.mkdir()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def write_capture(
        self,
        directory: Path,
        stem: str,
        *,
        bird_present: bool | None,
        filename: str | None = None,
        include_hash: bool = True,
    ) -> tuple[Path, Path]:
        directory.mkdir(parents=True, exist_ok=True)
        image_name = filename or f"{stem}.jpg"
        image_path = directory / image_name
        image_bytes = f"image bytes for {stem}".encode()
        image_path.write_bytes(image_bytes)
        data: dict[str, object] = {"filename": image_name}
        if include_hash:
            data["sha256"] = hashlib.sha256(image_bytes).hexdigest()
        if bird_present is not None:
            data["manual_label"] = {
                "bird_present": bird_present,
                "species": "Robin" if bird_present else None,
            }
        sidecar_path = directory / f"{stem}.json"
        sidecar_path.write_text(json.dumps(data), encoding="utf-8")
        return image_path, sidecar_path

    def test_discovers_both_bird_and_no_bird_labels_recursively(self) -> None:
        bird_image, bird_sidecar = self.write_capture(
            self.source / "day-one", "bird", bird_present=True
        )
        no_bird_image, no_bird_sidecar = self.write_capture(
            self.source / "day-two", "empty", bird_present=False
        )
        self.write_capture(self.source, "unreviewed", bird_present=None)
        (self.source / "broken.json").write_text("{not JSON", encoding="utf-8")

        result = copy_confirmed.discover_confirmed(self.source)

        self.assertEqual(
            {(item.image_path, item.sidecar_path) for item in result.captures},
            {(bird_image, bird_sidecar), (no_bird_image, no_bird_sidecar)},
        )
        self.assertEqual(result.sidecars_scanned, 4)
        self.assertEqual(result.unlabeled_sidecars, 1)
        self.assertEqual(result.invalid_sidecars, 1)
        self.assertEqual(result.missing_images, [])

    def test_uses_filename_field_and_reports_a_labeled_missing_image(self) -> None:
        image, sidecar = self.write_capture(
            self.source, "metadata-name", bird_present=True, filename="renamed.jpg"
        )
        missing = self.source / "missing.json"
        missing.write_text(
            json.dumps({"filename": "absent.jpg", "manual_label": {"bird_present": False}}),
            encoding="utf-8",
        )

        result = copy_confirmed.discover_confirmed(self.source)

        self.assertEqual(result.captures[0].image_path, image)
        self.assertEqual(result.captures[0].sidecar_path, sidecar)
        self.assertEqual(result.missing_images, [missing])

    def test_destination_tree_can_be_excluded_from_discovery(self) -> None:
        source_image, _ = self.write_capture(self.source, "source", bird_present=True)
        self.write_capture(self.destination, "old-copy", bird_present=True)

        result = copy_confirmed.discover_confirmed(self.source.parent, self.destination)

        self.assertEqual([capture.image_path for capture in result.captures], [source_image])

    @mock.patch("copy_confirmed.subprocess.run")
    def test_dry_run_queues_image_and_sidecar_without_creating_manifest(
        self, run: mock.Mock
    ) -> None:
        image, sidecar = self.write_capture(
            self.source / "nested", "capture", bird_present=True
        )
        run.return_value = subprocess.CompletedProcess([], 1)

        with mock.patch(
            "copy_confirmed.sys.argv",
            ["copy_confirmed.py", str(self.source), str(self.destination), "--dry-run"],
        ):
            copy_confirmed.main()

        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(command[:3], ["robocopy", str(image.parent), str(self.destination / "nested")])
        self.assertIn(image.name, command)
        self.assertIn(sidecar.name, command)
        self.assertIn("/L", command)
        self.assertFalse((self.destination / "manifest_confirmed.csv").exists())

    def test_verify_hashes_destination_image(self) -> None:
        image, _ = self.write_capture(self.source, "capture", bird_present=True)
        destination_image = self.destination / image.name
        self.destination.mkdir()
        destination_image.write_bytes(image.read_bytes())

        self.assertEqual(copy_confirmed.sha256_of(destination_image), copy_confirmed.sha256_of(image))
        destination_image.write_bytes(b"changed")
        self.assertNotEqual(copy_confirmed.sha256_of(destination_image), copy_confirmed.sha256_of(image))

    @mock.patch("copy_confirmed.subprocess.run")
    def test_copies_selected_files_preserving_directories_and_verifies(
        self, run: mock.Mock
    ) -> None:
        first = self.source / "species-one" / "first.jpg"
        second = self.source / "species-two" / "second.jpg"
        first.parent.mkdir()
        second.parent.mkdir()
        first.write_bytes(b"first image")
        second.write_bytes(b"second image")

        def copy_files(command: list[str], *, check: bool) -> subprocess.CompletedProcess:
            self.assertFalse(check)
            source_dir = Path(command[1])
            destination_dir = Path(command[2])
            destination_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_dir / command[3], destination_dir / command[3])
            return subprocess.CompletedProcess(command, 1)

        run.side_effect = copy_files
        result = copy_confirmed.copy_selected_files(
            self.source,
            self.destination,
            [Path("species-one/first.jpg"), Path("species-two/second.jpg")],
            verify=True,
        )

        self.assertEqual(result.files_queued, 2)
        self.assertEqual(result.robocopy_calls, 2)
        self.assertEqual(result.verified_files, 2)
        self.assertEqual((self.destination / "species-one" / "first.jpg").read_bytes(), b"first image")
        self.assertEqual((self.destination / "species-two" / "second.jpg").read_bytes(), b"second image")

    def test_selected_copy_rejects_path_traversal(self) -> None:
        with self.assertRaisesRegex(ValueError, "remain below"):
            copy_confirmed.copy_selected_files(
                self.source, self.destination, [Path("..") / "outside.jpg"]
            )

    @mock.patch("copy_confirmed.subprocess.run")
    def test_verify_copies_and_checks_pair_when_sidecar_has_no_hash(
        self, run: mock.Mock
    ) -> None:
        image, sidecar = self.write_capture(
            self.source, "capture", bird_present=False, include_hash=False
        )

        def copy_files(command: list[str], *, check: bool) -> subprocess.CompletedProcess:
            self.assertFalse(check)
            source_dir = Path(command[1])
            destination_dir = Path(command[2])
            destination_dir.mkdir(parents=True, exist_ok=True)
            for name in (image.name, sidecar.name):
                shutil.copy2(source_dir / name, destination_dir / name)
            return subprocess.CompletedProcess(command, 1)

        run.side_effect = copy_files
        with mock.patch(
            "copy_confirmed.sys.argv",
            ["copy_confirmed.py", str(self.source), str(self.destination), "--verify"],
        ):
            copy_confirmed.main()

        self.assertEqual((self.destination / image.name).read_bytes(), image.read_bytes())
        self.assertEqual(
            json.loads((self.destination / sidecar.name).read_text(encoding="utf-8")),
            json.loads(sidecar.read_text(encoding="utf-8")),
        )


if __name__ == "__main__":
    unittest.main()