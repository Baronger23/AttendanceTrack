import numpy as np

from attendance.services.replay_attack_detector import ReplayAttackDetector


def make_landmarks_for_box(x1=220, y1=140, x2=420, y2=360):
    landmarks = [(320.0, 240.0)] * 468
    coords = {
        1: (320.0, 230.0),
        10: (320.0, y1),
        33: (270.0, 220.0),
        61: (285.0, 310.0),
        152: (320.0, y2),
        199: (320.0, 330.0),
        234: (x1, 250.0),
        263: (370.0, 220.0),
        291: (355.0, 310.0),
        454: (x2, 250.0),
    }
    for idx, point in coords.items():
        landmarks[idx] = point
    return landmarks


def test_phone_frame_score_detects_rectangle_around_face():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[80:420, 170:470] = 40
    image[88:412, 178:462] = 210

    detector = ReplayAttackDetector()
    bbox = detector._face_bbox(image.shape, make_landmarks_for_box())

    score = detector.phone_frame_score(image, bbox)

    assert score >= 0.6


def test_planar_motion_score_detects_flat_translation():
    detector = ReplayAttackDetector()
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    for offset in (0, 20, 40, 60, 80):
        landmarks = [(x + offset, y + offset * 0.5) for x, y in make_landmarks_for_box()]
        detector.update(image, landmarks)

    assert detector.planar_motion_score() >= 0.45


def test_low_risk_on_empty_frame():
    detector = ReplayAttackDetector()
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    risk = detector.update(image, [])

    assert risk.score == 0.0
    assert not risk.suspicious
