"""
Management command to start Django + Celery worker in a single command.

Usage:
    python manage.py runall
"""

import os
import sys
import subprocess
import signal
import atexit
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = 'Start Django dev server + Celery worker in one command'

    def add_arguments(self, parser):
        parser.add_argument('--port', type=int, default=8000, help='Django port (default: 8000)')
        parser.add_argument('--no-celery', action='store_true', help='Skip Celery worker')

    def handle(self, *args, **options):
        port = options['port']
        skip_celery = options['no_celery']
        
        python_exe = sys.executable
        from django.conf import settings
        project_dir = str(settings.BASE_DIR)
        
        processes = []

        def cleanup():
            for p in processes:
                try:
                    p.terminate()
                    p.wait(timeout=5)
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass

        atexit.register(cleanup)

        # Start Celery worker
        if not skip_celery:
            self.stdout.write(self.style.WARNING('Starting Celery worker...'))
            celery_cmd = [
                python_exe, '-m', 'celery',
                '-A', 'myproject', 'worker',
                '-l', 'info',
                '--pool=solo',
                '-E',
            ]
            celery_proc = subprocess.Popen(
                celery_cmd,
                cwd=project_dir,
            )
            processes.append(celery_proc)
            self.stdout.write(self.style.SUCCESS(f'  Celery worker started (PID: {celery_proc.pid})'))

        # Start Django dev server
        self.stdout.write(self.style.WARNING(f'Starting Django server on port {port}...'))
        django_cmd = [python_exe, 'manage.py', 'runserver', str(port)]
        django_proc = subprocess.Popen(
            django_cmd,
            cwd=project_dir,
        )
        processes.append(django_proc)
        self.stdout.write(self.style.SUCCESS(f'  Django server started (PID: {django_proc.pid})'))

        self.stdout.write('')
        self.stdout.write(self.style.SUCCESS('=' * 50))
        self.stdout.write(self.style.SUCCESS(f'  All services running!'))
        self.stdout.write(self.style.SUCCESS(f'  Django:  http://127.0.0.1:{port}/'))
        if not skip_celery:
            self.stdout.write(self.style.SUCCESS(f'  Celery:  worker ready'))
        self.stdout.write(self.style.SUCCESS('=' * 50))
        self.stdout.write(self.style.WARNING('  Press Ctrl+C to stop all services'))
        self.stdout.write('')

        # Wait for any process to exit
        try:
            while True:
                for p in processes:
                    ret = p.poll()
                    if ret is not None:
                        self.stdout.write(self.style.ERROR(f'Process {p.pid} exited with code {ret}'))
                        cleanup()
                        return
                import time
                time.sleep(1)
        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING('\nShutting down...'))
            cleanup()
            self.stdout.write(self.style.SUCCESS('All services stopped.'))
