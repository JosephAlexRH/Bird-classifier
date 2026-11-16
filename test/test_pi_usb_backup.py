import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.backup_pi_usb import backup, check_mount, digest, main


class UsbBackupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.destination = self.root / 'usb'

    def tearDown(self):
        self.temporary.cleanup()

    def test_nested_files_verified_and_unchanged_files_not_recopied(self):
        (self.source / 'crops').mkdir()
        (self.source / 'crops/a.png').write_bytes(b'crop')
        (self.source / 'a.json').write_bytes(b'{ "original": true }\n')
        first = backup(self.source, self.destination)
        self.assertTrue(first['verified'])
        self.assertEqual(first['copied'], 2)
        self.assertFalse(first['consistent_snapshot'])
        second = backup(self.source, self.destination)
        self.assertEqual(second['copied'], 0)
        for entry in second['files']:
            self.assertEqual(entry['sha256'], digest(self.destination / 'bird_images' / entry['path']))

    def test_updates_preserve_previous_bytes_and_source_deletions_do_not_propagate(self):
        original = self.source / 'review.json'
        original.write_bytes(b'first')
        backup(self.source, self.destination)
        first_hash = digest(original)
        original.write_bytes(b'corrected')
        self.assertTrue(backup(self.source, self.destination)['verified'])
        self.assertEqual((self.destination / 'previous_versions' / first_hash).read_bytes(), b'first')
        original.unlink()
        backup(self.source, self.destination)
        self.assertEqual((self.destination / 'bird_images/review.json').read_bytes(), b'corrected')

    def test_changing_source_keeps_previous_backup_and_reports_failure(self):
        original = self.source / 'log'
        original.write_bytes(b'first')
        backup(self.source, self.destination)
        original.write_bytes(b'second')
        import scripts.backup_pi_usb as module
        real_copy = module.shutil.copyfileobj

        def changing_copy(incoming, output, length):
            real_copy(incoming, output, length)
            original.write_bytes(b'third')

        with patch.object(module.shutil, 'copyfileobj', side_effect=changing_copy):
            result = backup(self.source, self.destination)
        self.assertFalse(result['verified'])
        self.assertEqual((self.destination / 'bird_images/log').read_bytes(), b'first')
        self.assertFalse(list(self.destination.rglob('.copy-*')))

    def test_missing_or_wrong_mount_rejected(self):
        with patch('scripts.backup_pi_usb.subprocess.check_output', return_value='OTHER\n'):
            with self.assertRaises(RuntimeError):
                check_mount(Path('/mnt/bird-backup'), 'FC1B-D5B3')
        with patch('scripts.backup_pi_usb.subprocess.check_output', return_value='fc1b-d5b3\n'):
            check_mount(Path('/mnt/bird-backup'), 'FC1B-D5B3')

    def test_startup_guard_checks_mount_without_copying(self):
        with patch('sys.argv', ['backup_pi_usb.py', '--check-only', '--mount',
                                str(self.root), '--uuid', 'FC1B-D5B3']), \
                patch('scripts.backup_pi_usb.check_mount') as check, \
                patch('scripts.backup_pi_usb.backup') as copy:
            self.assertEqual(main(), 0)
            check.assert_called_once_with(self.root.resolve(), 'FC1B-D5B3')
            copy.assert_not_called()

    def test_overlapping_trees_and_missing_source_rejected(self):
        with self.assertRaises(ValueError):
            backup(self.source, self.source / 'nested')
        with self.assertRaises(FileNotFoundError):
            backup(self.root / 'missing', self.destination)

    def test_copy_failure_recorded_without_changing_source(self):
        original = self.source / 'a.jpg'
        original.write_bytes(b'original')
        with patch('scripts.backup_pi_usb.copy_stable', side_effect=OSError('No space left')):
            result = backup(self.source, self.destination)
        self.assertFalse(result['verified'])
        self.assertEqual(original.read_bytes(), b'original')
        report = json.loads(next((self.destination / 'reports').glob('*.json')).read_text())
        self.assertIn('No space left', report['errors'][0])


if __name__ == '__main__':
    unittest.main()
