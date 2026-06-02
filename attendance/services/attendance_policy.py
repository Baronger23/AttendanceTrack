"""
Attendance policy and decision recording.

This keeps check-in rules in one place so Celery, WebSocket fallback, and
legacy AJAX flows make the same attendance decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime as dt, timedelta
from typing import Any

from django.utils import timezone

from attendance.models import AttendanceLog


def _json_safe(value):
    """Return a JSONField-safe copy of AI result dictionaries."""
    try:
        import numpy as np
    except Exception:
        np = None

    if np is not None:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()

    if isinstance(value, dict):
        return {
            key: _json_safe(val)
            for key, val in value.items()
            if key not in {"embedding", "all_embeddings"}
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass
class AttendanceDecision:
    success: bool
    checkin_status: str
    message: str
    log: AttendanceLog | None = None
    status: str | None = None
    risk_score: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)


class AttendancePolicyService:
    EARLY_MINUTES = 60
    LATE_AFTER_END_MINUTES = 30
    REVIEW_RISK_THRESHOLD = 0.58
    SUSPICIOUS_RISK_THRESHOLD = 0.78

    @classmethod
    def record_checkin(
        cls,
        user,
        recognition_result: dict | None = None,
        liveness_result: dict | None = None,
        snapshot=None,
        kiosk_device: str = "",
    ) -> AttendanceDecision:
        recognition_result = _json_safe(recognition_result or {})
        liveness_result = _json_safe(liveness_result or {})
        now = timezone.localtime(timezone.now())
        today = now.date()

        shift_error = cls._shift_error(user, now)
        evidence = cls._build_evidence(recognition_result, liveness_result, kiosk_device)
        risk_score = cls.calculate_risk_score(recognition_result, liveness_result)

        if shift_error:
            return AttendanceDecision(
                success=True,
                checkin_status='shift_error',
                message=shift_error,
                risk_score=risk_score,
                evidence=evidence,
            )

        existing_log = AttendanceLog.objects.filter(user=user, timestamp__date=today).first()
        if existing_log:
            return AttendanceDecision(
                success=True,
                checkin_status='already_checked',
                message=f'{user.get_full_name()} đã chấm công hôm nay lúc {existing_log.timestamp.strftime("%H:%M")}!',
                log=existing_log,
                status=existing_log.status,
                risk_score=risk_score,
                evidence=evidence,
            )

        status = AttendanceLog.Status.ON_TIME
        reason = 'Matched identity and attendance policy passed.'

        if risk_score >= cls.SUSPICIOUS_RISK_THRESHOLD:
            status = AttendanceLog.Status.SUSPICIOUS
            reason = 'High risk score; stored for admin review.'
        elif risk_score >= cls.REVIEW_RISK_THRESHOLD:
            status = AttendanceLog.Status.MANUAL_REVIEW
            reason = 'Moderate risk score; stored for admin review.'

        log = AttendanceLog(
            user=user,
            status=status,
            snapshot=snapshot,
            face_confidence=recognition_result.get('confidence'),
            liveness_score=liveness_result.get('liveness_score'),
            spoof_score=liveness_result.get('spoof_score'),
            replay_score=liveness_result.get('replay_score'),
            risk_score=risk_score,
            recognition_method=recognition_result.get('method', ''),
            top_candidate_score=recognition_result.get('top_candidate_score', recognition_result.get('confidence')),
            second_candidate_score=recognition_result.get('second_candidate_score'),
            decision_reason=reason,
            evidence=evidence,
        )
        if status in {AttendanceLog.Status.ON_TIME, AttendanceLog.Status.LATE}:
            log.status = log.calculate_status(now)
        log.save()

        status_display = log.get_status_display()
        return AttendanceDecision(
            success=True,
            checkin_status='success' if log.status not in {AttendanceLog.Status.MANUAL_REVIEW, AttendanceLog.Status.SUSPICIOUS} else 'review',
            message=f'Chấm công thành công! Xin chào {user.get_full_name()} - {status_display}',
            log=log,
            status=log.status,
            risk_score=risk_score,
            evidence=evidence,
        )

    @classmethod
    def calculate_risk_score(cls, recognition_result: dict, liveness_result: dict) -> float:
        face_conf = float(recognition_result.get('confidence') or 0.0)
        top_score = float(recognition_result.get('top_candidate_score') or face_conf)
        second_score = recognition_result.get('second_candidate_score')
        second_score = float(second_score) if second_score is not None else 0.0
        margin = max(0.0, top_score - second_score)

        liveness_score = float(liveness_result.get('liveness_score') or 1.0)
        spoof_score = float(liveness_result.get('spoof_score') or 0.0)
        replay_score = float(liveness_result.get('replay_score') or 0.0)

        low_face_risk = max(0.0, (65.0 - face_conf) / 65.0)
        margin_risk = max(0.0, (8.0 - margin) / 8.0) if second_score else 0.0
        liveness_risk = max(0.0, 1.0 - liveness_score)

        risk = (
            0.36 * low_face_risk
            + 0.22 * margin_risk
            + 0.18 * liveness_risk
            + 0.14 * spoof_score
            + 0.10 * replay_score
        )
        return round(float(min(max(risk, 0.0), 1.0)), 3)

    @classmethod
    def _shift_error(cls, user, now) -> str | None:
        if not user.work_shift:
            return None

        shift = user.work_shift
        now_time = now.time()
        today = now.date()
        shift_start_dt = dt.combine(today, shift.start_time)
        shift_end_dt = dt.combine(today, shift.end_time)

        allowed_start = (shift_start_dt - timedelta(minutes=cls.EARLY_MINUTES)).time()
        allowed_end = (shift_end_dt + timedelta(minutes=cls.LATE_AFTER_END_MINUTES)).time()

        if shift.start_time > shift.end_time:
            in_shift = now_time >= allowed_start or now_time <= allowed_end
        else:
            in_shift = allowed_start <= now_time <= allowed_end

        if in_shift:
            return None

        return (
            f'Chưa đến ca của bạn! Ca "{shift.name}" bắt đầu lúc '
            f'{shift.start_time.strftime("%H:%M")} - {shift.end_time.strftime("%H:%M")}. '
            f'Bạn có thể chấm công từ {allowed_start.strftime("%H:%M")}.'
        )

    @staticmethod
    def _build_evidence(recognition_result: dict, liveness_result: dict, kiosk_device: str) -> dict:
        return {
            'recognition': {
                'method': recognition_result.get('method', ''),
                'confidence': recognition_result.get('confidence'),
                'top_candidate_score': recognition_result.get('top_candidate_score'),
                'second_candidate_score': recognition_result.get('second_candidate_score'),
                'match_margin': recognition_result.get('match_margin'),
                'top_candidates': recognition_result.get('top_candidates', []),
            },
            'liveness': {
                'liveness_score': liveness_result.get('liveness_score'),
                'spoof_score': liveness_result.get('spoof_score'),
                'replay_score': liveness_result.get('replay_score'),
                'anti_spoof_model': liveness_result.get('anti_spoof_model', ''),
            },
            'kiosk_device': kiosk_device,
        }
