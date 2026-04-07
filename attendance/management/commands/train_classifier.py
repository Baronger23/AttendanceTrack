"""
Management command to train the SVM face classifier.

Collects all face embeddings from the database and trains
an SVM classifier for fast face identification.

Usage:
    python manage.py train_classifier
"""

from django.core.management.base import BaseCommand
import numpy as np


class Command(BaseCommand):
    help = 'Train the SVM face classifier using stored face embeddings'

    def handle(self, *args, **options):
        from attendance.services.face_service import FaceService
        from attendance.models import User, FaceEmbedding

        self.stdout.write("=" * 50)
        self.stdout.write("Training SVM Face Classifier")
        self.stdout.write("=" * 50)

        # Count staff with encodings
        staff_with_face = User.objects.filter(
            role=User.Role.STAFF
        ).exclude(face_encoding_text__isnull=True).exclude(face_encoding_text='').count()

        self.stdout.write(f"\nStaff with face encodings: {staff_with_face}")
        
        # Count FaceEmbedding records
        try:
            emb_count = FaceEmbedding.objects.count()
            self.stdout.write(f"FaceEmbedding records: {emb_count}")
        except Exception:
            emb_count = 0
            self.stdout.write("FaceEmbedding table not available")

        if staff_with_face < 2 and emb_count < 2:
            self.stdout.write(self.style.WARNING(
                "\n⚠️ Need at least 2 different staff members with face encodings to train classifier."
            ))
            return

        # Train classifier
        face_service = FaceService()
        success = face_service.retrain_classifier()

        if success:
            self.stdout.write(self.style.SUCCESS("\n✅ Classifier trained successfully!"))
            self.stdout.write(f"Model saved to: {face_service.classifier.model_path}")
        else:
            self.stdout.write(self.style.ERROR("\n❌ Classifier training failed!"))
            self.stdout.write("Ensure at least 2 staff members have registered their faces.")
