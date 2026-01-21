"""
Django management command to seed the database with sample data
Usage: python manage.py seed_data
"""
from django.core.management.base import BaseCommand
from django.utils import timezone
from datetime import datetime, time, timedelta
import random

from attendance.models import User, WorkShift, AttendanceLog


class Command(BaseCommand):
    help = 'Seed database with sample employees, work shifts, and attendance logs'

    def add_arguments(self, parser):
        parser.add_argument(
            '--employees',
            type=int,
            default=20,
            help='Number of employees to create (default: 20)'
        )
        parser.add_argument(
            '--days',
            type=int,
            default=30,
            help='Number of days of attendance logs (default: 30)'
        )
        parser.add_argument(
            '--clear',
            action='store_true',
            help='Clear existing data before seeding'
        )

    def handle(self, *args, **options):
        num_employees = options['employees']
        num_days = options['days']
        clear_data = options['clear']

        if clear_data:
            self.stdout.write('Clearing existing data...')
            from django.db import connection
            
            # Xóa admin log trước với raw SQL
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM django_admin_log")
            self.stdout.write('   Cleared admin logs')
            
            # Xóa attendance logs
            AttendanceLog.objects.all().delete()
            self.stdout.write('   Cleared attendance logs')
            
            # Chỉ xóa nhân viên STAFF (không xóa ADMIN)
            staff_to_delete = User.objects.filter(role=User.Role.STAFF)
            count = staff_to_delete.count()
            staff_to_delete.delete()
            self.stdout.write(f'   Cleared {count} staff members')
            
            # Xóa ca làm việc
            WorkShift.objects.all().delete()
            self.stdout.write('   Cleared work shifts')
            
            self.stdout.write(self.style.SUCCESS('✅ Cleared existing data'))

        # Create work shifts
        self.stdout.write('Creating work shifts...')
        shifts = self.create_shifts()
        
        # Create employees
        self.stdout.write(f'Creating {num_employees} employees...')
        employees = self.create_employees(num_employees, shifts)
        
        # Create attendance logs
        self.stdout.write(f'Creating attendance logs for {num_days} days...')
        self.create_attendance_logs(employees, num_days)

        self.stdout.write(self.style.SUCCESS(f'''
✅ Seed completed!
   - Work shifts: {len(shifts)}
   - Employees: {len(employees)}
   - Days of attendance: {num_days}
        '''))

    def create_shifts(self):
        """Create sample work shifts"""
        shifts_data = [
            {'name': 'Ca sáng', 'start_time': time(6, 0), 'end_time': time(14, 0), 'late_grace_period': 15},
            {'name': 'Ca chiều', 'start_time': time(14, 0), 'end_time': time(22, 0), 'late_grace_period': 15},
            {'name': 'Ca hành chính', 'start_time': time(8, 0), 'end_time': time(17, 0), 'late_grace_period': 15},
            {'name': 'Ca đêm', 'start_time': time(22, 0), 'end_time': time(6, 0), 'late_grace_period': 10},
        ]
        
        shifts = []
        for data in shifts_data:
            shift, created = WorkShift.objects.get_or_create(
                name=data['name'],
                defaults={
                    'start_time': data['start_time'],
                    'end_time': data['end_time'],
                    'late_grace_period': data['late_grace_period']
                }
            )
            shifts.append(shift)
            if created:
                self.stdout.write(f'   Created shift: {shift.name}')
        
        return shifts

    def create_employees(self, count, shifts):
        """Create sample employees"""
        first_names = [
            'An', 'Bình', 'Cường', 'Dũng', 'Em', 'Phúc', 'Giang', 'Hùng', 'Inh', 'Khoa',
            'Linh', 'Minh', 'Nam', 'Oanh', 'Phong', 'Quân', 'Rồng', 'Sơn', 'Tùng', 'Uyên',
            'Việt', 'Xuân', 'Yến', 'Zin', 'Anh', 'Bảo', 'Chi', 'Dương', 'Hiền', 'Hương'
        ]
        
        last_names = [
            'Nguyễn', 'Trần', 'Lê', 'Phạm', 'Hoàng', 'Huỳnh', 'Phan', 'Vũ', 'Võ', 'Đặng',
            'Bùi', 'Đỗ', 'Hồ', 'Ngô', 'Dương', 'Lý', 'Trương', 'Đinh', 'Lưu', 'Cao'
        ]
        
        employees = []
        existing_count = User.objects.filter(role=User.Role.STAFF).count()
        
        for i in range(count):
            username = f'nhanvien{existing_count + i + 1}'
            
            # Skip if already exists
            if User.objects.filter(username=username).exists():
                continue
            
            first_name = random.choice(first_names)
            last_name = random.choice(last_names)
            gender = random.choice(['M', 'F'])
            shift = random.choice(shifts)
            
            user = User.objects.create_user(
                username=username,
                password='123456',
                first_name=first_name,
                last_name=last_name,
                email=f'{username}@company.com',
                phone=f'09{random.randint(10000000, 99999999)}',
                gender=gender,
                role=User.Role.STAFF,
                work_shift=shift
            )
            employees.append(user)
            
            if (i + 1) % 10 == 0:
                self.stdout.write(f'   Created {i + 1} employees...')
        
        return employees

    def create_attendance_logs(self, employees, num_days):
        """Create sample attendance logs"""
        today = timezone.now().date()
        logs_created = 0
        
        for day_offset in range(num_days):
            date = today - timedelta(days=day_offset)
            
            # Skip weekends (optional)
            if date.weekday() >= 5:
                continue
            
            for employee in employees:
                # Random chance to be absent (5%)
                if random.random() < 0.05:
                    continue
                
                # Skip if already has log for this day
                if AttendanceLog.objects.filter(user=employee, timestamp__date=date).exists():
                    continue
                
                # Determine check-in time
                if employee.work_shift:
                    shift_start = employee.work_shift.start_time
                    grace = employee.work_shift.late_grace_period
                    
                    # Random arrival time: -30 min early to +45 min late
                    minutes_offset = random.randint(-30, 45)
                    
                    check_in_hour = shift_start.hour
                    check_in_minute = shift_start.minute + minutes_offset
                    
                    # Normalize time
                    while check_in_minute >= 60:
                        check_in_hour += 1
                        check_in_minute -= 60
                    while check_in_minute < 0:
                        check_in_hour -= 1
                        check_in_minute += 60
                    
                    check_in_time = time(check_in_hour % 24, check_in_minute)
                    
                    # Determine status
                    if minutes_offset <= grace:
                        status = AttendanceLog.Status.ON_TIME
                    else:
                        status = AttendanceLog.Status.LATE
                else:
                    check_in_time = time(8, random.randint(0, 30))
                    status = AttendanceLog.Status.ON_TIME
                
                # Create timestamp
                timestamp = timezone.make_aware(
                    datetime.combine(date, check_in_time)
                )
                
                AttendanceLog.objects.create(
                    user=employee,
                    timestamp=timestamp,
                    status=status
                )
                logs_created += 1
        
        self.stdout.write(f'   Created {logs_created} attendance logs')
