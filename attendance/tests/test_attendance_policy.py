import os

from django.test import TestCase

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "myproject.settings")

from attendance.models import User
from attendance.services.attendance_policy import AttendancePolicyService


class AttendancePolicyRiskTest(TestCase):
    def test_low_risk_when_face_and_liveness_are_strong(self):
        risk = AttendancePolicyService.calculate_risk_score(
            {
                'confidence': 82.0,
                'top_candidate_score': 82.0,
                'second_candidate_score': 60.0,
            },
            {
                'liveness_score': 0.94,
                'spoof_score': 0.03,
                'replay_score': 0.05,
            },
        )

        self.assertLess(risk, 0.2)

    def test_high_risk_when_face_margin_and_liveness_are_weak(self):
        risk = AttendancePolicyService.calculate_risk_score(
            {
                'confidence': 55.0,
                'top_candidate_score': 55.0,
                'second_candidate_score': 53.0,
            },
            {
                'liveness_score': 0.55,
                'spoof_score': 0.6,
                'replay_score': 0.8,
            },
        )

        self.assertGreater(risk, 0.45)

    def test_record_checkin_stores_evidence(self):
        user = User.objects.create_user(username='policy-staff', password='pass')

        decision = AttendancePolicyService.record_checkin(
            user=user,
            recognition_result={
                'confidence': 83.0,
                'method': 'insightface_arcface',
                'top_candidate_score': 83.0,
                'second_candidate_score': 61.0,
            },
            liveness_result={
                'liveness_score': 0.92,
                'spoof_score': 0.02,
                'replay_score': 0.04,
            },
        )

        self.assertEqual(decision.checkin_status, 'success')
        self.assertIsNotNone(decision.log)
        self.assertEqual(decision.log.recognition_method, 'insightface_arcface')
        self.assertEqual(decision.log.evidence['recognition']['confidence'], 83.0)
