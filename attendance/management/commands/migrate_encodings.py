"""
Management command to migrate existing 128D face encodings to 512D embeddings.

Re-processes face images through the new CNN pipeline:
  MTCNN Detection → Alignment → CLAHE → InceptionResnetV1 (512D)

Usage:
    python manage.py migrate_encodings
    python manage.py migrate_encodings --from-images   # Re-process from stored images
"""

from django.core.management.base import BaseCommand
import numpy as np


class Command(BaseCommand):
    help = 'Migrate existing face encodings to the new 512D embedding format'

    def add_arguments(self, parser):
        parser.add_argument(
            '--from-images',
            action='store_true',
            help='Re-process original images from Supabase Storage (recommended)',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would be done without making changes',
        )

    def handle(self, *args, **options):
        from attendance.models import User, FaceEmbedding
        from attendance.services.face_service import FaceService

        dry_run = options['dry_run']
        from_images = options['from_images']

        self.stdout.write("=" * 60)
        self.stdout.write("Migrating Face Encodings (128D → 512D CNN)")
        self.stdout.write("=" * 60)

        # Get all staff with old encodings
        staff_with_old = User.objects.filter(
            role=User.Role.STAFF
        ).exclude(face_encoding_text__isnull=True).exclude(face_encoding_text='')

        self.stdout.write(f"\nFound {staff_with_old.count()} staff with old face encodings")

        if from_images:
            self._migrate_from_images(staff_with_old, dry_run)
        else:
            self.stdout.write(self.style.WARNING(
                "\n⚠️ Cannot convert 128D dlib vectors directly to 512D CNN vectors."
                "\n   The representations are fundamentally different."
                "\n   Staff will need to re-register their faces."
                "\n\n   Options:"
                "\n   1. Use --from-images to re-process from stored photos"
                "\n   2. Have staff re-register at /face-register/"
            ))
            
            # List staff who need re-registration
            for staff in staff_with_old:
                old_enc = staff.get_encoding()
                dim = len(old_enc) if old_enc is not None else 0
                self.stdout.write(f"  - {staff.get_full_name() or staff.username} ({dim}D encoding)")

    def _migrate_from_images(self, staff_queryset, dry_run):
        """Re-process face images from Supabase Storage."""
        import cv2
        import requests
        from django.conf import settings
        from attendance.models import FaceEmbedding
        from attendance.services.face_service import FaceService

        face_service = FaceService()
        success_count = 0
        fail_count = 0

        for staff in staff_queryset:
            self.stdout.write(f"\nProcessing: {staff.get_full_name() or staff.username}...")

            if not staff.avatar:
                self.stdout.write(self.style.WARNING("  → No avatar image path, skipping"))
                fail_count += 1
                continue

            # Download image from Supabase
            img_url = staff.get_avatar_url()
            if not img_url:
                self.stdout.write(self.style.WARNING("  → Cannot get avatar URL"))
                fail_count += 1
                continue

            try:
                response = requests.get(img_url, timeout=30)
                if response.status_code != 200:
                    self.stdout.write(self.style.WARNING(f"  → Download failed: HTTP {response.status_code}"))
                    fail_count += 1
                    continue

                # Decode image
                img_array = np.frombuffer(response.content, np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img is None:
                    self.stdout.write(self.style.WARNING("  → Cannot decode image"))
                    fail_count += 1
                    continue

                if dry_run:
                    self.stdout.write(self.style.SUCCESS("  → [DRY RUN] Would re-process"))
                    success_count += 1
                    continue

                # Run through new pipeline
                result = face_service.register_face(img)

                if result['success']:
                    # Save new 512D encoding to User
                    staff.set_encoding(result['embedding'])
                    staff.save()
                    embedding_method = result.get('method') or 'facenet_inceptionresnet'

                    # Save to FaceEmbedding table
                    # Clear old embeddings first
                    FaceEmbedding.objects.filter(user=staff).delete()

                    # Save all embeddings (original + augmented)
                    for i, emb in enumerate(result['all_embeddings']):
                        fe = FaceEmbedding(
                            user=staff,
                            source='original' if i == 0 else 'augmented',
                            recognition_method=embedding_method,
                        )
                        fe.set_embedding(np.array(emb))
                        fe.save()

                    self.stdout.write(self.style.SUCCESS(
                        f"  → ✅ Migrated! {len(result['all_embeddings'])} embeddings saved"
                    ))
                    success_count += 1
                else:
                    self.stdout.write(self.style.ERROR(f"  → ❌ {result['error']}"))
                    fail_count += 1

            except Exception as e:
                self.stdout.write(self.style.ERROR(f"  → ❌ Error: {e}"))
                fail_count += 1

        self.stdout.write(f"\n{'=' * 60}")
        self.stdout.write(f"Results: {success_count} migrated, {fail_count} failed")
        
        if success_count >= 2 and not dry_run:
            self.stdout.write("\nAuto-training classifier...")
            if face_service.retrain_classifier():
                self.stdout.write(self.style.SUCCESS("✅ Classifier trained!"))
            else:
                self.stdout.write(self.style.WARNING("⚠️ Classifier training failed"))
