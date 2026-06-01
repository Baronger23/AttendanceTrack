from datetime import datetime, time
from zoneinfo import ZoneInfo

from django.test import TestCase

from attendance.models import AttendanceLog, User, WorkShift


class AttendanceLogStatusTest(TestCase):
    def test_checkin_within_grace_period_is_on_time(self):
        shift = WorkShift.objects.create(
            name="Morning",
            start_time=time(8, 0),
            end_time=time(17, 0),
            late_grace_period=15,
        )
        user = User.objects.create_user(username="staff1", password="pass", work_shift=shift)
        log = AttendanceLog(user=user)

        status = log.calculate_status(datetime(2026, 6, 1, 8, 15, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh")))

        self.assertEqual(status, AttendanceLog.Status.ON_TIME)

    def test_checkin_after_grace_period_is_late(self):
        shift = WorkShift.objects.create(
            name="Morning",
            start_time=time(8, 0),
            end_time=time(17, 0),
            late_grace_period=15,
        )
        user = User.objects.create_user(username="staff2", password="pass", work_shift=shift)
        log = AttendanceLog(user=user)

        status = log.calculate_status(datetime(2026, 6, 1, 8, 16, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh")))

        self.assertEqual(status, AttendanceLog.Status.LATE)

    def test_overnight_shift_uses_previous_day_start_after_midnight(self):
        shift = WorkShift.objects.create(
            name="Night",
            start_time=time(22, 0),
            end_time=time(6, 0),
            late_grace_period=10,
        )
        user = User.objects.create_user(username="staff3", password="pass", work_shift=shift)
        log = AttendanceLog(user=user)

        status = log.calculate_status(datetime(2026, 6, 2, 1, 0, tzinfo=ZoneInfo("Asia/Ho_Chi_Minh")))

        self.assertEqual(status, AttendanceLog.Status.LATE)
