from __future__ import annotations

import io
import json
import tempfile
import threading
import unittest
from unittest.mock import patch
from html.parser import HTMLParser
from pathlib import Path

import numpy as np
from PIL import Image

from server.bird_inference import BirdInferenceEngine
from server.camera_registry import CameraRegistry
from server.hub_settings import HubSettings
from server.pi_motion import MotionService
from server.capture_storage import CaptureStore
from server.raspberry_pi_server import create_app
from server.collection_dashboard import collection_page
from test import test_bird_inference as fixtures
from test.test_pi_motion import metadata, jpeg


class Engine:
    bird_threshold=.5
    threshold_source='contract'
    species=None
    contract=None
    available=True
    def describe(self):
        return {'ready':True,'species_available':self.available,'species_enabled':self.species is not None}
    def set_species_enabled(self,enabled):
        if enabled and not self.available:raise ValueError('Missing species model')
        self.species=object() if enabled else None


class Service:
    def __init__(self,camera_url,*_,**kwargs):
        self.camera_url=camera_url
        self.expected_camera_id=kwargs.get('expected_camera_id')
        self.camera_name=kwargs.get('camera_name')
        self.options=kwargs
        self.running=False
    def start(self):self.running=True
    def stop(self):self.running=False
    def background_command(self, action, *, confirmed_empty=False):
        if action == 'learn' and not confirmed_empty:
            raise ValueError('Confirm empty scene')
        if action not in ('learn', 'reset'):
            raise ValueError('Invalid action')
        self.background_action = action
        return {'phase': 'learning' if action == 'learn' else 'needs_learning'}
    def describe(self):return {'running':self.running,**{k:v for k,v in self.options.items() if v is None or type(v) in (str,int,float,bool)}}


class HubControlTests(unittest.TestCase):
    def test_background_action_requires_confirmation_and_is_scoped_to_selected_camera(self):
        route = f"/api/cameras/{self.one['id']}/background"
        self.assertEqual(self.client.post(route, json={'action': 'learn'}).status_code, 400)
        response = self.client.post(route, json={'action': 'learn', 'confirmed_empty': True})
        self.assertEqual(response.status_code, 202)
        services = self.app.config['MOTION_SERVICES']
        self.assertEqual(services[self.one['id']].background_action, 'learn')
        self.assertFalse(hasattr(services[self.two['id']], 'background_action'))
        self.assertEqual(self.client.post('/api/cameras/missing/background', json={'action': 'reset'}).status_code, 404)
        self.client.post(f"/api/cameras/{self.one['id']}/pipeline", json={'capture_enabled': False})
        self.assertEqual(self.client.post(route, json={'action': 'learn', 'confirmed_empty': True}).status_code, 409)

    def test_motion_controls_validate_persist_and_reach_workers(self):
        patch = {'motion_triggered': True, 'motion_threshold': 35, 'motion_minimum_area': 200,
                 'empty_sample_seconds': 600}
        self.assertEqual(self.client.post('/api/hub/settings', json=patch).status_code, 200)
        for service in self.app.config['MOTION_SERVICES'].values():
            self.assertEqual(service.options['config'].minimum_threshold, 35)
            self.assertEqual(service.options['config'].minimum_component_pixels, 200)
            self.assertEqual(service.options['empty_sample_seconds'], 600)
        saved = HubSettings(self.root/'captures/hub_settings.json', {}).snapshot()
        for key, value in patch.items():
            self.assertEqual(saved[key], value)
        for invalid in ({'motion_threshold': 0}, {'motion_threshold': 20.5},
                        {'motion_minimum_area': 0}, {'empty_sample_seconds': -1},
                        {'motion_triggered': 'yes'}):
            self.assertEqual(self.client.post('/api/hub/settings', json=invalid).status_code, 400)

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        self.engine=Engine()
        self.registry=CameraRegistry(self.root/'registry.json')
        self.one=self.registry.add('One','http://one.local')
        self.two=self.registry.add('Two','http://two.local')
        self.apps=[]
        self.app=self.make_app()
        self.client=self.app.test_client()

    def make_app(self):
        app=create_app(self.root/'captures',inference_engine=self.engine,camera_registry=self.registry,
            collection_mode=True,manage_motion_services=True,motion_service_factory=Service)
        self.apps.append(app)
        return app

    def tearDown(self):
        for app in self.apps:
            app.config['CLEANUP']()
            app.config['HUB_STATUS_LOGGER'].stop()
        self.temp.cleanup()

    def test_pause_one_camera_keeps_other_running_and_preserves_original(self):
        path=self.root/'captures/capture_original.jpg'
        path.write_bytes(b'original retained')
        other=self.app.config['MOTION_SERVICES'][self.two['id']]
        result=self.client.post('/api/cameras/'+self.one['id']+'/pipeline',json={'capture_enabled':False})
        self.assertEqual(result.status_code,200)
        self.assertNotIn(self.one['id'],self.app.config['MOTION_SERVICES'])
        self.assertIs(self.app.config['MOTION_SERVICES'][self.two['id']],other)
        self.assertTrue(other.running)
        self.assertEqual(path.read_bytes(),b'original retained')
        self.client.post('/api/cameras/'+self.one['id']+'/pipeline',json={'capture_enabled':True})
        self.assertIn(self.one['id'],self.app.config['MOTION_SERVICES'])

    def test_settings_are_independent_persistent_and_control_worker_options(self):
        result=self.client.post('/api/hub/settings',json={'species_enabled':True,'poll_seconds':3,'detector_threshold':.7})
        self.assertEqual(result.status_code,200)
        self.assertTrue(result.json['settings']['collection_mode'])
        self.assertIsNotNone(self.engine.species)
        self.assertEqual(self.engine.bird_threshold,.7)
        self.client.post('/api/hub/settings',json={'motion_localization_enabled':False})
        for service in self.app.config['MOTION_SERVICES'].values():
            self.assertFalse(service.options['motion_localization_enabled'])
            self.assertEqual(service.options['poll_seconds'],3)
        self.client.post('/api/hub/settings',json={'capture_enabled':False})
        self.assertEqual(self.app.config['MOTION_SERVICES'],{})
        reopened=HubSettings(self.root/'captures/hub_settings.json',{})
        self.assertFalse(reopened.value['capture_enabled'])
        self.assertTrue(reopened.value['species_enabled'])
        self.assertTrue(reopened.value['collection_mode'])
        self.assertEqual(self.client.get('/health').status_code,200)

    def test_failed_settings_write_rolls_back_models_and_restarts_previous_capture(self):
        before=self.client.get('/api/hub/settings').json['settings']
        with patch.object(self.app.config['HUB_SETTINGS'],'save',side_effect=OSError('Disk write failed')):
            result=self.client.post('/api/hub/settings',json={'species_enabled':True,'poll_seconds':4})
        self.assertEqual(result.status_code,400)
        self.assertIsNone(self.engine.species)
        self.assertEqual(self.client.get('/api/hub/settings').json['settings'],before)
        self.assertEqual(len(self.app.config['MOTION_SERVICES']),2)
        self.assertTrue(all(s.running for s in self.app.config['MOTION_SERVICES'].values()))

    def test_capture_recovers_after_a_slow_operation_prevents_settings_change(self):
        release, recovered = threading.Event(), threading.Event()
        instances=[]
        class SlowService(MotionService):
            def __init__(self,*args,**kwargs):
                super().__init__(*args,**kwargs)
                instances.append(self)
                self.initial=len(instances)<=2
            def start(self):
                self.running=True
                if self.initial:
                    self._thread=threading.Thread(target=release.wait,daemon=True)
                    self._thread.start()
                elif len(instances)>=4:
                    recovered.set()
            def stop(self,timeout=5):
                self._stop.set()
                self.running=False
        app=create_app(self.root/'slow',inference_engine=Engine(),camera_registry=self.registry,
            collection_mode=True,manage_motion_services=True,motion_service_factory=SlowService)
        self.apps.append(app)
        try:
            response=app.test_client().post('/api/hub/settings',json={'poll_seconds':5})
            self.assertEqual(response.status_code,409)
            release.set()
            self.assertTrue(recovered.wait(3))
            self.assertEqual(app.config['HUB_SETTINGS'].value['poll_seconds'],.25)
            self.assertTrue(all(not s.initial and s.running for s in app.config['MOTION_SERVICES'].values()))
        finally:
            release.set()
            for instance in instances:
                if instance._thread:instance._thread.join(3)

    def test_missing_species_and_invalid_combinations_leave_state_unchanged(self):
        before=self.client.get('/api/hub/settings').json
        self.engine.available=False
        result=self.client.post('/api/hub/settings',json={'species_enabled':True})
        self.assertEqual(result.status_code,400)
        self.assertEqual(result.json['error'],'Species model or labels are missing on the Pi. Detection and collection remain available.')
        self.assertEqual(self.client.get('/api/hub/settings').json['settings'],before['settings'])
        self.assertEqual(self.client.post('/api/hub/settings',json={'collection_mode':False,'detector_enabled':False}).status_code,400)
        self.assertEqual(self.client.post('/api/hub/settings',json={'poll_seconds':float('nan')}).status_code,400)
        self.assertEqual(self.client.post('/api/hub/settings',json={'capture_enabled':'false'}).status_code,400)
        self.assertEqual(len(self.app.config['MOTION_SERVICES']),2)

    def test_turning_analysis_off_does_not_call_the_model_or_discard_raw_data(self):
        class NeverInfer(Engine):
            def infer(self,*_,**__):raise AssertionError('Analysis should be disabled')
        frame=jpeg(np.full((30,40,3),50,np.uint8))
        service=MotionService('http://camera',CaptureStore(self.root/'raw'),NeverInfer(),collection_mode=True,
            detector_enabled=False,fetcher=lambda *_a,**_k:(frame,metadata()))
        result=service.process_once()
        self.assertTrue(result['persisted'])
        self.assertEqual(result['inference']['reason'],'analysis_disabled')
        self.assertIsNone(result['inference']['is_bird'])
        self.assertEqual(len(list((self.root/'raw').glob('capture_*.jpg'))),1)

    def test_uploaded_frames_obey_the_same_analysis_off_switch(self):
        self.client.post('/api/hub/settings',json={'detector_enabled':False,'species_enabled':False})
        with patch.object(self.engine,'infer',create=True,side_effect=AssertionError('Model must not run')) as infer:
            response=self.client.post('/upload',data=jpeg(np.full((30,40,3),50,np.uint8)))
            self.assertEqual(response.status_code,200)
            infer.assert_not_called()
        captured=self.root/'captures'/response.json['filename']
        record=json.loads(captured.with_suffix('.inference.json').read_text())
        self.assertEqual(record['reason'],'analysis_disabled')
        self.assertIsNone(record['is_bird'])

    def test_all_pi_bookmarks_share_the_same_workspace(self):
        for route in ('/','/fleet','/legacy'):
            html=self.client.get(route).get_data(as_text=True)
            self.assertIn('<title>Bird hub</title>',html)
            self.assertIn('id="pane-cameras"',html)
            self.assertIn('id="pane-fleet"',html)
            self.assertIn('id="pane-settings"',html)
            self.assertNotIn('legacy dashboard (full config)',html)
        class IDs(HTMLParser):
            def __init__(self):super().__init__();self.ids=[]
            def handle_starttag(self,tag,attrs):self.ids.extend(v for k,v in attrs if k=='id')
        parser=IDs();parser.feed(collection_page(remote=True))
        self.assertEqual(len(parser.ids),len(set(parser.ids)))

    def test_species_can_be_enabled_lazily_without_changing_detector_output(self):
        detector,species,contract=fixtures.DetectorInferenceTests().write_bundle(self.root)
        factory=fixtures.DetectorInferenceTests.fake_interpreter()
        engine=BirdInferenceEngine(None,species,detector_model=detector,detector_contract_file=contract,
                                   interpreter_factory=factory,species_enabled=False)
        self.assertNotIn(species.name,factory.instances)
        image=self.root/'image.jpg'
        image.write_bytes(jpeg(np.full((30,40,3),60,np.uint8)))
        kwargs={'selected_crop':{'left':0,'top':0,'right':40,'bottom':30},'crop_source':'capture_motion_record.selection.final_bounds_px'}
        before=engine.infer(image,**kwargs)
        engine.set_species_enabled(True)
        enabled=engine.infer(image,**kwargs)
        self.assertEqual(before['detector'],enabled['detector'])
        self.assertIsNotNone(enabled['guess'])
        engine.set_species_enabled(False)
        after=engine.infer(image,**kwargs)
        self.assertIsNone(after['guess'])
        self.assertEqual(after['detector'],before['detector'])
