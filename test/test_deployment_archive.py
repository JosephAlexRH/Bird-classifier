from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.archive_pi_deployment import build_inventory, verify, inventory_source, rescue_http, sha256


class DeploymentArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.source=self.root/'source'
        self.source.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def test_exact_raw_copy_and_nested_logs_verify(self):
        (self.source/'capture_a.jpg').write_bytes(b'original jpeg bytes')
        (self.source/'capture_a.json').write_bytes(b'{ "schema_version": 3, "manual_label": {"bird_present":false}}\r\n')
        (self.source/'runlog').mkdir()
        (self.source/'runlog/run.jsonl').write_bytes(b'{}\n')
        inventory=build_inventory(self.source)
        destination=self.root/'copy'
        shutil.copytree(self.source,destination)
        self.assertTrue(verify(destination,inventory)['verified'])
        (destination/'capture_a.jpg').write_bytes(b'corrupted')
        report=verify(destination,inventory)
        self.assertFalse(report['verified'])
        self.assertEqual(report['errors'][0]['error'],'size_or_sha256_mismatch')

    def test_missing_extra_and_traversal_are_not_verified(self):
        (self.source/'a').write_bytes(b'a')
        inventory=build_inventory(self.source)
        destination=self.root/'copy'
        destination.mkdir()
        (destination/'extra').write_bytes(b'extra')
        report=verify(destination,inventory)
        self.assertEqual({r['error'] for r in report['errors']},{'missing','not_in_source_inventory'})
        inventory['files'][0]['path']='../escape'
        with self.assertRaises(ValueError):verify(destination,inventory)

    @unittest.skipUnless(os.name == 'nt', 'Windows filesystem path case')
    def test_windows_merged_model_directories_keep_hash_verification(self):
        destination = self.root / 'models-copy'
        (destination / 'Models').mkdir(parents=True)
        (destination / 'Models/a.tflite').write_bytes(b'a')
        (destination / 'Models/b.tflite').write_bytes(b'b')
        inventory = build_inventory(destination)
        inventory['files'][1]['path'] = 'models/b.tflite'
        self.assertTrue(verify(destination, inventory)['verified'])
        (destination / 'Models/b.tflite').write_bytes(b'corrupt')
        self.assertFalse(verify(destination, inventory)['verified'])
        inventory['files'].append(dict(inventory['files'][0], path='models/a.tflite'))
        inventory['file_count'] += 1
        with self.assertRaises(ValueError):
            verify(destination, inventory)

    def test_inventory_preserves_source_and_records_model_and_code(self):
        (self.source/'capture_a.jpg').write_bytes(b'original')
        source_hash=sha256(self.source/'capture_a.jpg')
        project=self.root/'project'
        (project/'server').mkdir(parents=True)
        (project/'Models/detector').mkdir(parents=True)
        (project/'server/main.py').write_bytes(b'print(1)\n')
        (project/'Models/detector/model.tflite').write_bytes(b'model')
        output=self.root/'audit'
        inventory_source(self.source,output,project)
        self.assertEqual(sha256(self.source/'capture_a.jpg'),source_hash)
        audit=json.loads((output/'audit_inventory.json').read_text())
        self.assertTrue(verify(output,audit)['verified'])
        self.assertTrue((output/'deployed_assets/Models/detector/model.tflite').is_file())
        self.assertEqual(json.loads((output/'deployment_context.json').read_text())['source_inventory_sha256'],sha256(output/'source_inventory.json'))
        with self.assertRaises(FileExistsError):inventory_source(self.source,output,project)

    def test_legacy_http_rescue_is_verified_but_never_claims_full_sidecar_backup(self):
        cid='capture_20261116T162436_323766Z_afb502b0'
        image=b'unchanged JPEG'
        capture={'capture_id':cid,'sha256':hashlib.sha256(image).hexdigest(),
                 'inference':{'status':'complete','guess':'wrong species'},'motion':{'schema_version':1}}
        responses={
            'http://pi.local/health':json.dumps({'latest_image':cid+'.jpg'}).encode(),
            'http://pi.local/api/captures?limit=100&after=':json.dumps({'captures':[capture],'has_more':False}).encode(),
            f'http://pi.local/api/captures/{cid}/image':image}
        class Response:
            def __init__(self,data):self.data=data
            def __enter__(self):return self
            def __exit__(self,*_):pass
            def read(self,*_):return self.data
        output=self.root/'rescue'
        with patch('scripts.archive_pi_deployment.urlopen',side_effect=lambda url,**_:Response(responses[url])):
            rescue_http('http://pi.local',output)
        report=json.loads((output/'rescue_report.json').read_text())
        self.assertEqual(report['images_verified'],1)
        self.assertFalse(report['complete_raw_archive'])
        self.assertEqual(report['missing_original_acquisition_sidecars'],[cid])
        self.assertFalse((output/'captures'/f'{cid}.json').exists())
        self.assertEqual((output/'captures'/f'{cid}.jpg').read_bytes(),image)
        self.assertTrue((output/'_api'/f'{cid}.inference.json').is_file())


if __name__=='__main__':unittest.main()
