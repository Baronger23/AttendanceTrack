import os

import numpy as np
from django.test import SimpleTestCase, override_settings

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")

from attendance.services.anti_spoofing_service import AntiSpoofingService


class AntiSpoofingServiceTest(SimpleTestCase):
    @override_settings(ANTISPOOF_ONNX_MODEL_PATH='')
    def test_returns_neutral_scores_without_model(self):
        AntiSpoofingService._instance = None

        service = AntiSpoofingService()
        result = service.predict(np.zeros((80, 80, 3), dtype=np.uint8), [])

        self.assertFalse(result['enabled'])
        self.assertEqual(result['live_score'], 1.0)
        self.assertEqual(result['spoof_score'], 0.0)

    @override_settings(ANTISPOOF_ONNX_MODEL_PATH='models/MiniFASNetV2.onnx')
    def test_with_real_model(self):
        AntiSpoofingService._instance = None

        service = AntiSpoofingService()
        self.assertTrue(service.enabled)
        self.assertIsNotNone(service.session)

        dummy_img = np.zeros((80, 80, 3), dtype=np.uint8)
        result = service.predict(dummy_img, [])

        self.assertTrue(result['enabled'])
        self.assertIn('live_score', result)
        self.assertIn('spoof_score', result)
        self.assertAlmostEqual(result['live_score'] + result['spoof_score'], 1.0, places=2)

