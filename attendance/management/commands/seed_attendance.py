from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import datetime, timedelta
import random
from attendance.models import User, AttendanceLog


class Command(BaseCommand):
    help = 'Seed attendance logs for all staff members for the last 30 days (excluding Sundays)'

    def add_arguments(self, parser):
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing attendance logs before seeding',
        )

    def handle(self, *args, **options):
        # Clear existing logs if requested
        if options['clear']:
            self.stdout.write('Clearing existing attendance logs...')
            AttendanceLog.objects.all().delete()
            self.stdout.write(self.style.SUCCESS('✅ Cleared all attendance logs'))

        # Get all staff users
        staff_users = User.objects.filter(role=User.Role.STAFF)
        
        if not staff_users.exists():
            self.stdout.write(self.style.ERROR('❌ No staff users found. Please create staff users first.'))
            return

        self.stdout.write(f'Found {staff_users.count()} staff members')

        # Calculate date range (last 30 days)
        today = timezone.now().date()
        start_date = today - timedelta(days=30)

        total_logs_created = 0

        # Iterate through each day in the range
        current_date = start_date
        while current_date <= today:
            # Skip Sundays (weekday() returns 6 for Sunday)
            if current_date.weekday() == 6:
                self.stdout.write(f'  Skipping {current_date.strftime("%Y-%m-%d")} (Sunday)')
                current_date += timedelta(days=1)
                continue

            # For each staff member
            for user in staff_users:
                # Weighted random status selection
                # 85% ON_TIME, 10% LATE, 5% ABSENT
                rand = random.random()
                
                if rand < 0.85:  # 85% ON_TIME
                    status = AttendanceLog.Status.ON_TIME
                    # Random time between 07:30 and 08:00
                    hour = 7
                    minute = random.randint(30, 59)
                    second = random.randint(0, 59)
                    
                elif rand < 0.95:  # 10% LATE (85% + 10%)
                    status = AttendanceLog.Status.LATE
                    # Random time between 08:01 and 08:45
                    hour = 8
                    minute = random.randint(1, 45)
                    second = random.randint(0, 59)
                    
                else:  # 5% ABSENT
                    status = AttendanceLog.Status.ABSENT
                    # Set a default time for absent records (08:00)
                    hour = 8
                    minute = 0
                    second = 0

                # Create timestamp
                timestamp = timezone.make_aware(
                    datetime.combine(current_date, datetime.min.time()).replace(
                        hour=hour, minute=minute, second=second
                    )
                )

                # Create attendance log
                AttendanceLog.objects.create(
                    user=user,
                    timestamp=timestamp,
                    status=status,
                    snapshot=None  # No snapshot for seeded data
                )
                
                total_logs_created += 1

            current_date += timedelta(days=1)

        self.stdout.write(self.style.SUCCESS(f'✅ Successfully created {total_logs_created} attendance logs'))
        self.stdout.write(self.style.SUCCESS(f'   Date range: {start_date} to {today}'))
        self.stdout.write(self.style.SUCCESS(f'   Staff members: {staff_users.count()}'))
        self.stdout.write(self.style.SUCCESS(f'   Distribution: ~85% On-time, ~10% Late, ~5% Absent'))
