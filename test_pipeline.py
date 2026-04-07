"""End-to-end test of the new CNN face recognition pipeline."""
import os, sys
os.environ['DJANGO_SETTINGS_MODULE'] = 'myproject.settings'
sys.stdout.reconfigure(encoding='utf-8')
import django
django.setup()

import cv2
import numpy as np

print("=" * 60)
print("Testing Full CNN Face Recognition Pipeline")
print("=" * 60)

# Load test image
img = cv2.imread('test_face.jpg')
print(f"\n1. Image loaded: {img.shape}")
print(f"   Mean brightness: {np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)):.1f}")

# Test FaceService registration
from attendance.services.face_service import FaceService

fs = FaceService()
print("\n2. FaceService initialized (MTCNN + InceptionResnetV1)")

print("\n3. Testing registration pipeline (detect > align > CLAHE > augment > embed)...")
result = fs.register_face(img)
print(f"   Success: {result['success']}")

if result['success']:
    emb = result['embedding']
    print(f"   Embedding dimension: {len(emb)}")
    print(f"   Embedding L2 norm: {float(np.linalg.norm(emb)):.4f}")
    print(f"   Detection confidence: {result['confidence']:.4f}")
    print(f"   Total embeddings (original + augmented): {len(result['all_embeddings'])}")
    print(f"\n   PASS - Pipeline working correctly!")
else:
    print(f"   Error: {result.get('error', 'Unknown')}")
    print(f"\n   FAIL - Pipeline error")

print("\n" + "=" * 60)
print("Test completed!")
print("=" * 60)
