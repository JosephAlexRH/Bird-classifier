from __future__ import annotations

import csv
import hashlib
import io
import json
import tempfile
import threading
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
from flask import Flask

from server.bird_inference import BirdInferenceEngine
from server.capture_storage import CaptureStore, capture_files, locate_capture_file
from server.collection import export_bird_crops, manifest_csv, MANIFEST_FIELDS
from server.pi_bridge import register_pi_bridge
from server.pi_motion import MotionService
from server.raspberry_pi_server import create_app
from scripts.bbox_data import DETECTOR_MANIFEST_FIELDS
from test import test_bird_inference as inference_fixtures
from test.test_pi_motion import jpeg, metadata


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def frame(self):
        rgb = np.arange(40*30*3, dtype=np.uint16).reshape(30,40,3).astype(np.uint8)
        image = self.root / "capture_20261116T150000_000001Z_abc.jpg"
        image.write_bytes(jpeg(rgb))
        return image

    def test_detector_only_never_loads_species_or_labels(self):
        detector, species, contract = inference_fixtures.DetectorInferenceTests().write_bundle(self.root)
        species.unlink()
        species.with_name("labels.txt").unlink()
        factory = inference_fixtures.DetectorInferenceTests.fake_interpreter()
        engine = BirdInferenceEngine(None, None, detector_model=detector,
                                     detector_contract_file=contract, interpreter_factory=factory)
        path = self.frame()
        result = engine.infer(path, selected_crop={"left":0,"top":0,"right":40,"bottom":30},
                              crop_source="capture_motion_record.selection.final_bounds_px")
        self.assertIsNone(engine.describe()["species"])
        self.assertEqual(list(factory.instances), [detector.name])
        self.assertEqual(result["timing_ms"]["species"], 0)
        self.assertIsNone(result["guess"])
        self.assertNotIn("label", result["classification"])

    def test_tight_crops_clip_round_and_keep_native_pixels(self):
        from server.bird_inference import decode_rgb
        path = self.frame()
        original = path.read_bytes()
        result = export_bird_crops(path, {"status":"complete", "detector":{"threshold":.5,
            "model_sha256":"detector", "detections":[
                {"score":.9,"box":{"left":-2.1,"top":3.8,"right":10.2,"bottom":12.1}},
                {"score":.2,"box":{"left":0,"top":0,"right":10,"bottom":10}},
                {"score":.9,"box":{"left":20,"top":0,"right":10,"bottom":10}}]}})
        self.assertEqual(len(result["crops"]), 1)
        crop = result["crops"][0]
        self.assertEqual((crop["width"], crop["height"]), (11,10))
        with Image.open(self.root/crop["filename"]) as actual:
            np.testing.assert_array_equal(np.array(actual), decode_rgb(path)[3:13,0:11])
        self.assertEqual(path.read_bytes(),original)
        self.assertIsNone(crop["species_label"])

    def test_capture_continues_while_inference_is_blocked(self):
        entered, release = threading.Event(), threading.Event()
        class Engine:
            bird_threshold = .5
            contract = None
            def infer(self, path, **kwargs):
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("Test release missing")
                return {"status":"complete","is_bird":False,"bird_probability":.1}
        frame = jpeg(np.full((30,40,3),50,np.uint8))
        service = MotionService("http://camera",CaptureStore(self.root),Engine(),collection_mode=True,
                                fetcher=lambda *_a,**_k:(frame,metadata()), deployment_id="feeder-a")
        service._collection_capture_once()
        first = service._captured.get_nowait()
        worker = threading.Thread(target=service.process_once,args=(first,))
        worker.start()
        try:
            self.assertTrue(entered.wait(2))
            service._collection_capture_once()
            service._collection_capture_once()
            self.assertEqual(len(capture_files(self.root, "capture_*.jpg")),3)
            records=[json.loads(p.read_text()) for p in capture_files(self.root, "*.inference.json")]
            self.assertEqual(sum(r["status"]=="skipped" for r in records),1)
            self.assertEqual(service.inference_skipped,1)
            self.assertEqual(service._captured.qsize(),1)
        finally:
            release.set()
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(service.frames_polled,1)
        self.assertEqual(service.frames_captured,3)

    def test_reviews_preserve_original_and_export_confirmed_empties(self):
        app=create_app(self.root, collection_mode=True)
        client=app.test_client()
        raw=jpeg(np.full((30,40,3),70,np.uint8))
        capture=CaptureStore(self.root).save_capture(raw,source="test",extra={"camera_id":"camera-1","deployment_id":"scene-a"})
        cid=capture["capture_id"]
        original=locate_capture_file(self.root, capture["filename"]).read_bytes()
        acquisition=locate_capture_file(self.root, capture["sidecar"]).read_bytes()
        try:
            self.assertEqual(client.post(f"/api/captures/{cid}/review",json={"decision":"boxed","boxes":[]}).status_code,400)
            invalid={"left":0,"top":0,"right":100,"bottom":20}
            self.assertEqual(client.post(f"/api/captures/{cid}/review",json={"decision":"boxed","boxes":[invalid]}).status_code,400)
            box={"left":1.2,"top":2.8,"right":10,"bottom":12}
            response=client.post(f"/api/captures/{cid}/review",json={"decision":"boxed","boxes":[box],"species_name":"House Sparrow"})
            self.assertEqual(response.status_code,200)
            crop=client.get(f"/api/captures/{cid}/review-crops/0/image")
            self.assertEqual(crop.status_code,200)
            crop.close()
            rows=list(csv.DictReader(io.StringIO(''.join(manifest_csv(self.root)))))
            self.assertEqual(rows[0]["review_state"],"boxed")
            self.assertEqual(rows[0]["species_name"],"House Sparrow")
            self.assertEqual(rows[0]["xmin"],"1.2")
            client.post(f"/api/captures/{cid}/review",json={"decision":"no_bird","note":"Empty, checked entire original"})
            rows=list(csv.DictReader(io.StringIO(''.join(manifest_csv(self.root)))))
            self.assertEqual(rows[0]["review_state"],"no_bird")
            self.assertEqual(rows[0]["annotation_trust"],"human_verified")
            self.assertEqual(rows[0]["eligible_for_training"],"True")
            self.assertEqual(rows[0]["is_Bird"],"FALSE")
            self.assertEqual(rows[0]["bird_count"],"0")
            self.assertEqual(rows[0]["xmin"],"")
            self.assertEqual(rows[0]["species_name"],"")
            self.assertEqual(rows[0]["split_group"],"camera-1:scene-a")
            self.assertEqual(locate_capture_file(self.root, capture["filename"]).read_bytes(),original)
            self.assertEqual(locate_capture_file(self.root, capture["sidecar"]).read_bytes(),acquisition)
            self.assertEqual(len(json.loads(locate_capture_file(self.root, f"{cid}.review.json").read_text())["history"]),2)
            self.assertEqual(client.get(f"/api/captures/{cid}/review-crops/0/image").status_code,404)
            self.assertEqual(client.get('/api/captures?order=desc').json['captures'][0]['review']['decision'],'no_bird')
        finally:
            app.config['CLEANUP']()
            app.config['HUB_STATUS_LOGGER'].stop()

    def test_unreviewed_and_skipped_frames_never_become_training_negatives(self):
        path=self.frame()
        path.with_suffix('.inference.json').write_text(json.dumps({"status":"skipped","is_bird":None}))
        row=list(csv.DictReader(io.StringIO(''.join(manifest_csv(self.root)))))[0]
        self.assertEqual(row['eligible_for_training'],'False')
        self.assertEqual(row['is_Bird'],'')
        self.assertEqual(row['review_state'],'unreviewed')
        self.assertEqual(MANIFEST_FIELDS,DETECTOR_MANIFEST_FIELDS)

    def test_model_proposals_do_not_become_reviews_in_existing_bbox_editor(self):
        from scripts.bbox_data import load_detector_manifest
        image=self.frame()
        image.with_suffix('.inference.json').write_text(json.dumps({'status':'complete',
            'is_bird':True,'bird_probability':.95,'detector':{'threshold':.5,'detections':[
                {'score':.95,'box':{'left':1,'top':2,'right':20,'bottom':25}}]}}))
        csv_path=self.root/'deployment.csv'
        csv_path.write_text(''.join(manifest_csv(self.root)),encoding='utf-8')
        row=list(csv.DictReader(io.StringIO(csv_path.read_text())))[0]
        self.assertEqual(row['xmin'],'')
        self.assertEqual(row['is_Bird'],'')
        self.assertEqual(row['eligible_for_training'],'False')
        dataset,reviews=load_detector_manifest(csv_path,image_root=self.root)
        self.assertEqual(reviews,{})

    def test_newest_first_pagination_and_original_metadata(self):
        app=create_app(self.root)
        client=app.test_client()
        store=CaptureStore(self.root)
        records=[store.save_capture(jpeg(np.full((30,40,3),n,np.uint8)),source='test') for n in range(3)]
        try:
            page=client.get('/api/captures?order=desc&limit=1').json
            self.assertEqual(page['captures'][0]['capture_id'],records[-1]['capture_id'])
            second=client.get('/api/captures?order=desc&limit=1&before='+page['next_before']).json
            self.assertEqual(second['captures'][0]['capture_id'],records[-2]['capture_id'])
            response=client.get('/api/captures/'+records[0]['capture_id']+'/metadata')
            self.assertEqual(response.data,locate_capture_file(self.root, records[0]['sidecar']).read_bytes())
            response.close()
        finally:
            app.config['CLEANUP']()
            app.config['HUB_STATUS_LOGGER'].stop()

    def test_collection_keeps_uncertain_non_bird_original(self):
        class Engine:
            bird_threshold=.5
            contract=None
            def infer(self,*_a,**_k):
                return {'status':'complete','is_bird':False,'bird_probability':.4}
        service=MotionService('http://camera',CaptureStore(self.root),Engine(),collection_mode=True,
            fetcher=lambda *_a,**_k:(jpeg(np.full((30,40,3),50,np.uint8)),metadata()))
        result=service.process_once()
        self.assertTrue(result['persisted'])
        self.assertEqual(len(capture_files(self.root, "capture_*.jpg")),1)
        self.assertFalse(result['background_admitted'])
        self.assertEqual(result['inference']['collection']['crops'],[])
        row=list(csv.DictReader(io.StringIO(''.join(manifest_csv(self.root)))))[0]
        self.assertEqual(row['eligible_for_training'],'False')
        self.assertEqual(row['is_Bird'],'')

    def test_laptop_pi_configuration_and_allowlisted_routes(self):
        app=Flask(__name__)
        config=self.root/'pi.json'
        register_pi_bridge(app,config)
        client=app.test_client()
        self.assertEqual(client.get('/api/pi/health').status_code,503)
        self.assertEqual(client.post('/api/pi/config',json={'pi_url':'http://user:secret@host'}).status_code,400)
        self.assertEqual(client.post('/api/pi/config',json={'pi_url':'http://pi.local:5000'}).status_code,200)
        self.assertEqual(json.loads(config.read_text())['pi_url'],'http://pi.local:5000')
        class Upstream:
            status=200
            headers=Message()
            headers['Content-Type']='application/json'
            def __enter__(self):return self
            def __exit__(self,*_):pass
            def read(self,*_):return b'{"status":"ok"}'
        with patch('server.pi_bridge.urlopen',return_value=Upstream()) as opener:
            self.assertEqual(client.get('/api/pi/health').json,{'status':'ok'})
            self.assertEqual(opener.call_args.args[0].full_url,'http://pi.local:5000/health')
            client.get('/api/pi/captures?order=desc&limit=30&before=capture_1&host=evil')
            self.assertNotIn('evil',opener.call_args.args[0].full_url)
            self.assertIn('order=desc',opener.call_args.args[0].full_url)
            self.assertEqual(client.get('/api/pi/captures/bad%20id/image').status_code,400)

    def test_laptop_monitor_does_not_start_competing_camera_poller(self):
        from server.laptop_test_server import create_app as laptop_app
        from server.camera_registry import CameraRegistry
        registry=CameraRegistry(self.root/'registry.json')
        registry.add('Pi-owned camera','http://camera.local')
        with patch.object(registry,'start') as start:
            app=laptop_app(self.root/'laptop',camera_registry=registry,
                           start_registry_poller=True,pi_url='http://pi.local:5000')
            try:
                start.assert_not_called()
                self.assertTrue(app.config['PI_MONITOR_ONLY'])
            finally:
                app.config['CLEANUP']()
                app.config['HUB_STATUS_LOGGER'].stop()

    def test_pi_manual_calibration_capture_and_profile_without_laptop(self):
        app=create_app(self.root)
        registry=app.config['CAMERA_REGISTRY']
        camera=registry.add('Feeder','http://camera.local')
        cid=camera['id']
        client=app.test_client()
        class Upstream:
            status=200
            def __init__(self,data,kind):
                self.data=data
                self.headers=Message()
                self.headers['Content-Type']=kind
                self.headers['X-ESP32-Camera-Metadata']=metadata()
            def __enter__(self):return self
            def __exit__(self,*_):pass
            def read(self,*_):return self.data
        try:
            with patch('server.raspberry_pi_server.urlopen',return_value=Upstream(jpeg(np.full((30,40,3),60,np.uint8)),'image/jpeg')):
                response=client.post(f'/api/cameras/{cid}/captures',json={'calibration_type':'gray_card','note':'Field calibration'})
            self.assertEqual(response.status_code,200)
            capture=response.json
            self.assertEqual(capture['calibration_type'],'gray_card')
            self.assertEqual(capture['note'],'Field calibration')
            inference=json.loads(locate_capture_file(self.root, capture['capture_id']+'.inference.json').read_text())
            self.assertEqual(inference['reason'],'calibration_frame')
            self.assertEqual(app.config['CLASSIFICATION_QUEUE'].unfinished_tasks,0)
            with patch('server.raspberry_pi_server.urlopen',return_value=Upstream(b'{}','application/json')):
                self.assertEqual(client.post(f'/api/cameras/{cid}/profiles').status_code,200)
            self.assertEqual(len(list((self.root/'profiles').glob('*.json'))),1)
        finally:
            app.config['CLEANUP']()
            app.config['HUB_STATUS_LOGGER'].stop()

    def test_pi_stream_releases_camera_for_collection_on_close(self):
        app=create_app(self.root)
        registry=app.config['CAMERA_REGISTRY']
        cid=registry.add('Feeder','http://camera.local')['id']
        class Stream:
            status=200
            headers={'Content-Type':'multipart/x-mixed-replace; boundary=frame'}
            closed=False
            count=0
            def read(self,*_):
                self.count+=1
                return b'--frame\r\n' if self.count==1 else b''
            def close(self):self.closed=True
        stream=Stream()
        try:
            with patch('server.camera_tools.urlopen',return_value=stream):
                response=app.test_client().get(f'/api/cameras/{cid}/stream',buffered=False)
                self.assertEqual(response.status_code,200)
                self.assertFalse(registry.operation_lock(cid).acquire(blocking=False))
                response.close()
            self.assertTrue(stream.closed)
            self.assertTrue(registry.operation_lock(cid).acquire(blocking=False))
            registry.operation_lock(cid).release()
            self.assertFalse(registry.get(cid)['streaming'])
        finally:
            app.config['CLEANUP']()
            app.config['HUB_STATUS_LOGGER'].stop()


if __name__=='__main__':
    unittest.main()
