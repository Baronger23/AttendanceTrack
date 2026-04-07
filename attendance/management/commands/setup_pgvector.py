"""
Management command to enable pgvector extension on PostgreSQL.

Usage:
    python manage.py setup_pgvector
"""

from django.core.management.base import BaseCommand
from django.db import connection


class Command(BaseCommand):
    help = 'Enable pgvector extension on PostgreSQL database'

    def handle(self, *args, **options):
        with connection.cursor() as cursor:
            self.stdout.write("Enabling pgvector extension...")
            try:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                self.stdout.write(self.style.SUCCESS("✅ pgvector extension enabled successfully!"))
            except Exception as e:
                self.stdout.write(self.style.ERROR(f"❌ Failed to enable pgvector: {e}"))
                self.stdout.write(self.style.WARNING(
                    "Note: If using Supabase, enable the extension from the Supabase Dashboard:\n"
                    "  Database → Extensions → Search 'vector' → Enable"
                ))
            
            # Verify
            try:
                cursor.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
                row = cursor.fetchone()
                if row:
                    self.stdout.write(self.style.SUCCESS(f"pgvector version: {row[0]}"))
                else:
                    self.stdout.write(self.style.WARNING("pgvector extension not found"))
            except Exception:
                pass
