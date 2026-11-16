import tempfile
import unittest
from pathlib import Path

from scripts.archive_pi_deployment import build_inventory
from scripts.clean_redeploy_pi import check_tree, check_previous


class CleanRedeployTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name).resolve()
        (self.root / 'capture.jpg').write_bytes(b'bird')
        (self.root / 'crops').mkdir()
        (self.root / 'crops/a.png').write_bytes(b'crop')
        self.inventory = build_inventory(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def test_original_tree_verified_without_modification(self):
        self.assertEqual(check_tree(self.root, self.inventory),
                         {entry['sha256'] for entry in self.inventory['files']})
        self.assertEqual((self.root / 'capture.jpg').read_bytes(), b'bird')

    def test_new_capture_blocks_erasure(self):
        (self.root / 'new.jpg').write_bytes(b'new')
        with self.assertRaises(ValueError):
            check_tree(self.root, self.inventory)

    def test_changed_capture_blocks_erasure(self):
        (self.root / 'capture.jpg').write_bytes(b'fish')
        with self.assertRaises(ValueError):
            check_tree(self.root, self.inventory)

    def test_missing_file_blocks_erasure(self):
        (self.root / 'capture.jpg').unlink()
        with self.assertRaises(ValueError):
            check_tree(self.root, self.inventory)

    def test_traversal_and_duplicate_paths_rejected(self):
        self.inventory['files'][0]['path'] = '../outside'
        with self.assertRaises(ValueError):
            check_tree(self.root, self.inventory)

    def test_unarchived_history_refused(self):
        with self.assertRaises(ValueError):
            check_previous(self.root, {}, set())
        self.assertTrue((self.root / 'capture.jpg').exists())

    def test_separately_archived_history_accepted(self):
        check_previous(self.root, {str(self.root): self.inventory}, set())

    def test_history_changed_after_archiving_refused(self):
        (self.root / 'capture.jpg').write_bytes(b'changed')
        with self.assertRaises(ValueError):
            check_previous(self.root, {str(self.root): self.inventory}, set())
        self.inventory = build_inventory(self.root)
        self.inventory['files'].append(self.inventory['files'][0])
        with self.assertRaises(ValueError):
            check_tree(self.root, self.inventory)


if __name__ == '__main__':
    unittest.main()
