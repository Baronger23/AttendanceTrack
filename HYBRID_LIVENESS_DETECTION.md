# Hybrid Liveness Detection — Tài liệu kỹ thuật

## Bối cảnh

Hệ thống chấm công dùng nhận diện khuôn mặt qua Kiosk (WebSocket + Django Channels). Module liveness detection cũ có hai vấn đề đối lập:

- **Quá strict**: Ngưỡng EAR và góc xoay đầu quá nhạy → người thật bình thường không chấm công được.
- **Dễ bypass**: Chỉ cần lắc điện thoại có ảnh hoặc phát video replay là qua được.

Giải pháp: nâng cấp thành hệ thống **3-layer hybrid** — vừa chống spoofing hiệu quả, vừa đảm bảo người thật pass trong ≤ 8 giây.

---

## Kiến trúc tổng quan

```
Browser (5 FPS)
    │  WebSocket frame
    ▼
KioskConsumer
    │  update(image)
    ▼
LivenessService
    ├── Layer 1: Blink Detection (EAR Amplitude)
    ├── Layer 2: Head Movement (PnP)
    ├── Layer 3: Temporal Consistency
    └── Anti-Spoofing: EAR-Yaw Correlation
    │
    ▼  status: pending / passed / failed
KioskConsumer
    ├── pending  → liveness_feedback message → Browser
    ├── failed   → result {success: false}   → Browser (retry)
    └── passed   → identify_face_task.delay() → Celery
```

Mỗi WebSocket connection tạo một `LivenessService` instance riêng — không dùng singleton, đảm bảo isolation giữa các kiosk đồng thời.

---

## Ba lớp kiểm tra

### Layer 1 — Blink Detection (EAR Amplitude)

**Vấn đề cũ**: Dùng ngưỡng EAR tuyệt đối → quá nhạy với ánh sáng và góc camera.

**Giải pháp**: Đo *biên độ* EAR thay vì giá trị tuyệt đối.

```
EAR = (A + B) / (2 × C)

  p1 ─── p2
 /         \
p0           p3   ← horizontal (C)
 \         /
  p5 ─── p4
```

- **A** = khoảng cách dọc p1↔p5
- **B** = khoảng cách dọc p2↔p4  
- **C** = khoảng cách ngang p0↔p3

Sử dụng 6 landmark MediaPipe Face Mesh (468 điểm) cho mỗi mắt:
- Mắt trái: `[33, 160, 158, 133, 153, 144]`
- Mắt phải: `[362, 385, 387, 263, 373, 380]`

**Điều kiện pass**: Sau ≥ 5 frame, `max(EAR) - min(EAR) > 0.08` **VÀ** `min(EAR) < 0.22`

Điều này phát hiện nháy mắt thật (biên độ đủ lớn, có lúc mắt nhắm) mà không bị ảnh hưởng bởi EAR baseline của từng người.

---

### Layer 2 — Head Movement (Perspective-n-Point)

**Vấn đề cũ**: Dùng MSE pixel để đo chuyển động → bị lừa bởi lắc điện thoại.

**Giải pháp**: Ước lượng góc xoay đầu 3D thật sự bằng `cv2.solvePnP`.

6 điểm landmark 2D từ MediaPipe được map sang 6 điểm 3D của mô hình khuôn mặt chuẩn (mm):

| Điểm | MediaPipe index | 3D (mm) |
|------|----------------|---------|
| Nose tip | 1 | (0, 0, 0) |
| Chin | 152 | (0, −330, −65) |
| Left eye corner | 33 | (−225, 170, −135) |
| Right eye corner | 263 | (225, 170, −135) |
| Left mouth | 61 | (−150, −150, −125) |
| Right mouth | 291 | (150, −150, −125) |

Camera matrix được xây dựng với `focal_length = image_width`, `center = (w/2, h/2)` — không cần calibration thực tế.

Pipeline: `solvePnP` → `Rodrigues` (rotation vector → matrix) → `RQDecomp3x3` (Euler angles).

**Điều kiện pass**: Sau ≥ 5 frame, `yaw_range > 8°` **HOẶC** `pitch_range > 6°`

---

### Layer 3 — Temporal Consistency

**Mục đích**: Phân biệt người thật với ảnh tĩnh và video replay.

Người thật luôn có micro-variation tự nhiên (thở, rung nhẹ) → EAR dao động nhỏ nhưng không bằng 0.

```
std(EAR) < 0.005  → ảnh tĩnh hoặc video quá mượt  → FAIL
std(EAR) > 0.08   → lắc điện thoại giả tạo         → FAIL
0.005 ≤ std ≤ 0.08 → micro-variation tự nhiên       → PASS
```

Chỉ đánh giá khi buffer ≥ 10 frame.

---

### Anti-Spoofing — EAR-Yaw Correlation

**Mục đích**: Phát hiện video replay chất lượng cao (người thật trong video).

Trong video replay, EAR và yaw thường thay đổi đồng bộ (cùng nguồn). Người thật có chuyển động mắt và đầu độc lập nhau.

```python
corr = np.corrcoef(ears, yaws)[0, 1]
if abs(corr) > 0.95:  # tương quan quá cao → bất thường
    → FAIL + WARNING log
```

Kiểm tra khi buffer ≥ 15 frame. Edge case: nếu std của một trong hai chuỗi = 0 (correlation undefined) → bỏ qua, không fail.

---

## Frame Buffer

Mỗi frame được lưu dưới dạng dict số — **không lưu ảnh gốc**:

```python
{
    "timestamp": float,  # time.time()
    "ear":       float,  # EAR trung bình (trái + phải) / 2
    "pitch":     float,  # độ (degrees)
    "yaw":       float,  # độ (degrees)
}
```

Buffer giới hạn tối đa 30 frame (sliding window), tự động pop frame cũ nhất khi đầy.

---

## Luồng quyết định trong `update()`

```
Nhận frame
    │
    ├─ Timeout > 8s? ──────────────────────────────→ FAILED (reset)
    │
    ├─ MediaPipe: không có khuôn mặt? ─────────────→ PENDING
    │
    ├─ Tính EAR (trái + phải) / 2
    ├─ Tính pitch, yaw qua PnP
    ├─ Append vào frame_buffer (max 30)
    │
    ├─ buffer ≥ 5: Layer 1 blink check
    ├─ buffer ≥ 5: Layer 2 movement check
    │
    ├─ buffer ≥ 15: Spoofing correlation check ─────→ FAILED nếu |corr| > 0.95
    │
    ├─ buffer ≥ 10: Temporal consistency check ─────→ FAILED nếu std ngoài [0.005, 0.08]
    │
    ├─ blink AND movement AND consistency? ─────────→ PASSED
    │
    └─ Còn thiếu bước nào → PENDING + dynamic feedback
```

**Dynamic feedback** chỉ gợi ý đúng bước còn thiếu:
- `"chớp mắt"` nếu blink chưa pass
- `"quay đầu nhẹ"` nếu movement chưa pass
- `"giữ nguyên tư thế"` nếu consistency chưa pass (chỉ khi buffer ≥ 10)

---

## Cấu hình

Tất cả ngưỡng đọc từ Django settings, có giá trị mặc định an toàn:

```python
# myproject/settings.py
LIVENESS_EAR_THRESHOLD     = 0.22   # EAR tối thiểu khi mắt nhắm
LIVENESS_EAR_AMPLITUDE_MIN = 0.08   # Biên độ EAR tối thiểu để xác nhận nháy mắt
LIVENESS_YAW_RANGE_MIN     = 8.0    # Góc yaw tối thiểu (độ)
LIVENESS_PITCH_RANGE_MIN   = 6.0    # Góc pitch tối thiểu (độ)
LIVENESS_TEMPORAL_STD_MIN  = 0.005  # Std EAR tối thiểu (chống ảnh tĩnh)
LIVENESS_TEMPORAL_STD_MAX  = 0.08   # Std EAR tối đa (chống lắc điện thoại)
LIVENESS_TIMEOUT_SECONDS   = 8.0    # Timeout một phiên
```

Đọc một lần khi khởi tạo instance, không đọc lại trong mỗi frame.

---

## Tích hợp với hệ thống hiện có

Interface công khai giữ nguyên để `KioskConsumer` không cần refactor:

```python
service = LivenessService()          # mỗi WebSocket connection
result  = service.update(image)      # mỗi frame
# result = {"status": "pending"|"passed"|"failed", "feedback_ui": str}
service.reset()                      # sau mỗi phiên
```

`mediapipe` được import lazy bên trong `__init__` — Django startup không bị chậm nếu thư viện chưa cài. Nếu thiếu, raise `ImportError` với hướng dẫn rõ ràng.

---

## Test suite

**37 tests, 100% pass.**

### Unit tests (`test_liveness_service.py`) — 8 tests

| Test | Mục đích |
|------|----------|
| `test_timeout_triggers_failed_and_reset` | Timeout > 8s → failed + reset |
| `test_no_face_returns_pending` | Không có khuôn mặt → pending |
| `test_pnp_failure_skips_frame` | solvePnP fail → không crash, frame vẫn được buffer |
| `test_consistency_failed_triggers_reset` | EAR std = 0 (ảnh tĩnh) → failed + reset |
| `test_mediapipe_import_error` | Thiếu mediapipe → ImportError rõ ràng |
| `test_landmark_indices_correct` | LEFT_EYE, RIGHT_EYE indices chính xác |
| `test_default_thresholds_without_settings` | Giá trị mặc định đúng khi không có settings |
| `test_spoofing_logs_warning` | EAR-yaw correlation cao → WARNING log |

### Property-based tests (`test_liveness_service.py`) — 12 properties × 100 examples

Dùng **Hypothesis** để kiểm tra tính đúng đắn phổ quát trên không gian đầu vào rộng:

| Property | Phát biểu |
|----------|-----------|
| P1 | `__init__` và `reset()` cho trạng thái giống hệt nhau |
| P2 | `len(frame_buffer) ≤ history_size` với mọi N frame |
| P3 | `blink_detected = True` ↔ `ear_range > 0.08 AND min(ear) < 0.22` |
| P4 | Detection flags là idempotent — không thể đảo ngược về False |
| P5 | `movement_detected = True` ↔ `yaw_range > 8° OR pitch_range > 6°` |
| P6 | `consistency_passed` ↔ `0.005 ≤ std(EAR) ≤ 0.08` |
| P7 | `status = "passed"` ↔ cả 3 flag đều True |
| P8 | `update()` luôn trả về dict hợp lệ, không raise exception |
| P9 | Frame buffer chỉ chứa dữ liệu số, không chứa ảnh |
| P10 | Các instance độc lập nhau |
| P11 | `\|corr\| > 0.95` → failed + reset |
| P12 | Settings values được áp dụng đúng |

### Compute tests (`test_liveness_compute_ear_and_head_pose.py`) — 13 tests

Kiểm tra `compute_ear()` và `estimate_head_pose()` với geometry đã biết trước.

### Integration tests (`test_liveness_integration.py`) — 4 tests

Kiểm tra luồng `KioskConsumer` ↔ `LivenessService` end-to-end (mock Celery):

| Test | Mục đích |
|------|----------|
| `test_consumer_dispatches_celery_after_liveness_pass` | Celery task được dispatch đúng 1 lần khi passed |
| `test_consumer_sends_liveness_feedback_on_pending` | `liveness_feedback` message được gửi khi pending |
| `test_consumer_resets_after_successful_dispatch` | `reset()` được gọi sau dispatch thành công |
| `test_consumer_sends_result_on_liveness_failed` | `result {success: false}` được gửi khi failed |

### Chạy tests

```bash
python -m pytest attendance/tests/test_liveness_service.py \
                  attendance/tests/test_liveness_compute_ear_and_head_pose.py \
                  attendance/tests/test_liveness_integration.py \
                  -v --tb=short
# 37 passed
```

---

## Dependencies

```
mediapipe==0.10.21   # Face Mesh 468 landmarks
opencv-python-headless
numpy
hypothesis           # Property-based testing
pytest-asyncio       # Async integration tests
```

---

## Các file thay đổi

| File | Thay đổi |
|------|----------|
| `attendance/services/liveness_service.py` | Viết lại hoàn toàn |
| `attendance/consumers.py` | Fix `failed` handler: gọi `reset()`, bỏ `error` field thừa |
| `attendance/templates/attendance/kiosk.html` | Thêm `#liveness-hint` div, color-coded feedback, retry on failed |
| `myproject/settings.py` | Thêm 7 `LIVENESS_*` constants |
| `requirements.txt` | Đã có sẵn `mediapipe==0.10.21`, `hypothesis` |
| `attendance/tests/test_liveness_service.py` | Mới — 20 tests (unit + PBT) |
| `attendance/tests/test_liveness_compute_ear_and_head_pose.py` | Mới — 13 tests |
| `attendance/tests/test_liveness_integration.py` | Mới — 4 integration tests |
