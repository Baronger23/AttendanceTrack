# Implementation Plan: Hybrid Liveness Detection

## Overview

Nâng cấp `LivenessService` thành hệ thống **3-layer hybrid** (Blink Detection + Head Movement PnP + Temporal Consistency) để vừa chống spoofing hiệu quả vừa đảm bảo người thật pass trong ≤ 8 giây. Các thay đổi tập trung vào `liveness_service.py` (viết lại hoàn toàn), `settings.py` (thêm threshold config), `kiosk.html` (cải thiện liveness feedback UI), và bộ test mới với Hypothesis PBT.

## Tasks

- [x] 1. Cài đặt dependency và cập nhật cấu hình
  - Thêm `mediapipe` vào `requirements.txt` với phiên bản pinned (ví dụ: `mediapipe==0.10.21`)
  - Thêm `hypothesis` vào `requirements.txt` cho property-based testing
  - Thêm block `# Hybrid Liveness Detection Thresholds` vào `myproject/settings.py` với 7 hằng số:
    `LIVENESS_EAR_THRESHOLD = 0.22`, `LIVENESS_EAR_AMPLITUDE_MIN = 0.08`,
    `LIVENESS_YAW_RANGE_MIN = 8.0`, `LIVENESS_PITCH_RANGE_MIN = 6.0`,
    `LIVENESS_TEMPORAL_STD_MIN = 0.005`, `LIVENESS_TEMPORAL_STD_MAX = 0.08`,
    `LIVENESS_TIMEOUT_SECONDS = 8.0`
  - _Requirements: 10.1, 10.2, 8.4_

- [x] 2. Viết lại `LivenessService` — khung lớp và khởi tạo
  - Viết lại toàn bộ `attendance/services/liveness_service.py` từ đầu
  - Định nghĩa hằng số landmark: `LEFT_EYE`, `RIGHT_EYE` (6 indices mỗi mắt), `PNP_LANDMARKS` dict, `PNP_3D_POINTS` array
  - Implement `__init__(self)`: import `mediapipe` lazy bên trong method, khởi tạo `mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)`, đọc tất cả 7 threshold từ `django.conf.settings` với `getattr(settings, 'LIVENESS_*', <default>)`, khởi tạo `frame_buffer = []`, `history_size = 30`, `blink_detected = False`, `movement_detected = False`, `consistency_passed = False`, `session_start = time.time()`
  - Raise `ImportError("mediapipe is required. Run: pip install mediapipe")` nếu import thất bại
  - Implement `reset(self)`: xóa `frame_buffer`, đặt lại 3 flag về `False`, cập nhật `session_start`
  - _Requirements: 1.1, 1.2, 7.2, 7.4, 8.1, 8.2, 8.4, 8.5, 10.1, 10.2, 10.3_

- [x] 3. Implement `compute_ear()` và `estimate_head_pose()`
  - Implement `compute_ear(self, landmarks: list, eye_indices: list) -> float`: tính EAR từ 6 điểm landmark theo công thức `(A + B) / (2 * C)` với A, B là khoảng cách dọc, C là khoảng cách ngang; trả về `0.0` nếu C == 0
  - Implement `estimate_head_pose(self, image: np.ndarray, face_landmarks: list) -> tuple`: dùng `cv2.solvePnP` với `PNP_3D_POINTS` và 6 điểm 2D tương ứng, xây dựng camera matrix với `focal_length = image_width`, `center = (w/2, h/2)`, convert rotation vector sang Euler angles bằng `cv2.Rodrigues` + `cv2.RQDecomp3x3`; trả về `(0.0, 0.0, 0.0)` nếu `solvePnP` thất bại
  - _Requirements: 2.3, 2.4, 3.1, 3.2, 3.3, 3.5_

- [x] 4. Implement Layer 1 (Blink) và Layer 2 (Head Movement) trong `update()`
  - Implement phần đầu của `update(self, image: np.ndarray) -> dict`:
    - Kiểm tra timeout: nếu `time.time() - session_start > timeout_seconds` → gọi `reset()`, trả về `{"status": "failed", "feedback_ui": "Hết thời gian, vui lòng thử lại."}`
    - Chạy MediaPipe FaceMesh trên frame RGB; nếu không có khuôn mặt → trả về `{"status": "pending", "feedback_ui": "Không tìm thấy khuôn mặt rõ ràng."}`
    - Convert landmarks sang pixel coordinates, tính `left_ear`, `right_ear`, `avg_ear`
    - Gọi `estimate_head_pose()` để lấy `pitch`, `yaw`
    - Append entry `{"timestamp": ..., "ear": avg_ear, "pitch": pitch, "yaw": yaw}` vào `frame_buffer`; pop entry cũ nhất nếu `len > history_size`
    - Khi `len(frame_buffer) >= 5` và `not self.blink_detected`: tính `ear_range = max(ears) - min(ears)`; nếu `ear_range > EAR_AMPLITUDE_MIN` và `min(ears) < EAR_THRESHOLD` → đặt `blink_detected = True`
    - Khi `len(frame_buffer) >= 5` và `not self.movement_detected`: tính `yaw_range`, `pitch_range`; nếu `yaw_range > YAW_RANGE_MIN` hoặc `pitch_range > PITCH_RANGE_MIN` → đặt `movement_detected = True`
  - Wrap toàn bộ logic trong try/except; exception không mong đợi → log error, trả về `{"status": "pending", "feedback_ui": "Lỗi xử lý frame, vui lòng thử lại."}`
  - _Requirements: 1.3, 1.4, 2.1, 2.2, 2.3, 2.5, 2.6, 3.1, 3.4, 3.5, 3.6, 7.3, 7.5, 8.1_

- [x] 5. Implement Layer 3 (Temporal Consistency) và Anti-Spoofing Correlation
  - Implement `_check_temporal_consistency(self) -> bool`: tính `std = np.std([f["ear"] for f in frame_buffer])`; trả về `True` nếu `TEMPORAL_STD_MIN <= std <= TEMPORAL_STD_MAX`, ngược lại `False`
  - Implement `_check_spoofing_correlation(self) -> bool`: tính `np.corrcoef(ears, yaws)[0, 1]`; trả về `True` (spoofing detected) nếu `abs(corr) > 0.95`; xử lý edge case khi std của một trong hai chuỗi bằng 0 (correlation undefined → trả về `False`)
  - Tích hợp vào `update()`:
    - Khi `len(frame_buffer) >= 15`: gọi `_check_spoofing_correlation()`; nếu spoofing → log `logger.warning(f"Spoofing detected: corr={corr:.3f}, session_duration={...}s")`, gọi `reset()`, trả về `{"status": "failed", "feedback_ui": "Phát hiện hành vi bất thường, vui lòng thử lại tự nhiên."}`
    - Khi `len(frame_buffer) >= 10`: gọi `_check_temporal_consistency()` để cập nhật `consistency_passed`; nếu `consistency_passed = False` sau khi đã có đủ 10 frame → trả về `{"status": "failed", "feedback_ui": "Phát hiện hành vi bất thường, vui lòng thử lại tự nhiên."}`, gọi `reset()`
    - Khi `len(frame_buffer) < 10`: giữ `consistency_passed = False`
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 9.1, 9.2, 9.3, 9.4_

- [x] 6. Implement Hybrid Decision và Dynamic Feedback trong `update()`
  - Sau khi đã chạy cả 3 layer, thêm phần quyết định cuối vào `update()`:
    - Nếu `blink_detected and movement_detected and consistency_passed` → trả về `{"status": "passed", "feedback_ui": "Xác thực thành công!"}`
    - Ngược lại, xây dựng `feedback` list động: thêm `"chớp mắt"` nếu `not blink_detected`, thêm `"quay đầu nhẹ"` nếu `not movement_detected`, thêm `"giữ nguyên tư thế"` nếu `not consistency_passed` (chỉ khi buffer >= 10)
    - Trả về `{"status": "pending", "feedback_ui": "Vui lòng " + " và ".join(feedback)}`
  - _Requirements: 5.1, 5.3, 5.5_

- [x] 7. Checkpoint — Kiểm tra LivenessService hoạt động đúng
  - Ensure all tests pass, ask the user if questions arise.

- [x] 8. Cập nhật `KioskConsumer` để xử lý đúng trạng thái `failed`
  - Trong `attendance/consumers.py`, tìm block xử lý `liveness_result["status"] == "failed"` trong `handle_checkin()`
  - Đảm bảo khi `status == "failed"`, consumer gửi đúng message `{"type": "result", "result": {"success": False, "feedback_ui": liveness_result["feedback_ui"]}}` và **không** dispatch Celery task
  - Đảm bảo sau khi gửi `failed`, `liveness_passed` được reset về `False` và `liveness_service.reset()` được gọi để cho phép thử lại
  - Xác nhận rằng khi `status == "passed"`, consumer gửi `{"type": "liveness_feedback", "message": "Xác thực thành công!"}` trước khi dispatch Celery task, sau đó gọi `liveness_service.reset()` và đặt `liveness_passed = False`
  - _Requirements: 5.2, 6.1, 6.3, 6.4, 6.5_

- [x] 9. Cập nhật `kiosk.html` — Liveness Feedback UI
  - Trong `attendance/templates/attendance/kiosk.html`, cập nhật block xử lý `data.type === 'liveness_feedback'` trong `ws.onmessage`:
    - Thêm element `<div id="liveness-hint">` bên dưới camera wrap (hoặc bên trong `processingPanel`) để hiển thị feedback text
    - Khi nhận `liveness_feedback`, cập nhật nội dung `#liveness-hint` với `data.message`; áp dụng style phân biệt: màu xanh lá cho "Xác thực thành công!", màu vàng/cam cho các hướng dẫn còn thiếu
    - Khi nhận `result` với `success: false` và có `feedback_ui`, hiển thị `feedback_ui` trong `#liveness-hint` với màu đỏ, sau đó reset UI về trạng thái ban đầu (hiện lại nút "Chấm Công", ẩn `processingPanel`)
    - Đảm bảo `#liveness-hint` bị ẩn khi bắt đầu phiên mới (click "Chấm Công")
    - Cập nhật `step1` label từ "Phát hiện khuôn mặt" thành "Xác thực liveness" để phản ánh đúng luồng mới
  - _Requirements: 6.1, 6.3, 6.4_

- [x] 10. Viết unit tests cho `LivenessService`
  - Tạo file `attendance/tests/test_liveness_service.py`
  - [x] 10.1 Viết unit tests cho các edge cases cụ thể:
    - `test_timeout_triggers_failed_and_reset`: mock `time.time()` để giả lập timeout > 8s, kiểm tra trả về `status="failed"` và `frame_buffer` rỗng sau đó
    - `test_no_face_returns_pending`: mock `face_mesh.process()` trả về `None`, kiểm tra trả về `{"status": "pending", "feedback_ui": "Không tìm thấy khuôn mặt rõ ràng."}`
    - `test_pnp_failure_skips_frame`: mock `cv2.solvePnP` trả về `(False, ...)`, kiểm tra không crash và frame vẫn được xử lý
    - `test_consistency_failed_triggers_reset`: inject 10+ frame với EAR std < 0.005 (ảnh tĩnh), kiểm tra trả về `status="failed"` và `reset()` được gọi
    - `test_mediapipe_import_error`: mock `import mediapipe` raise `ImportError`, kiểm tra `LivenessService()` raise `ImportError` với message đúng
    - `test_landmark_indices_correct`: kiểm tra `LEFT_EYE == [33, 160, 158, 133, 153, 144]` và `RIGHT_EYE == [362, 385, 387, 263, 373, 380]`
    - `test_default_thresholds_without_settings`: khởi tạo `LivenessService` mà không có `LIVENESS_*` trong settings, kiểm tra các giá trị mặc định đúng
    - `test_spoofing_logs_warning`: inject 15+ frame với EAR và yaw có tương quan cao, kiểm tra `logger.warning` được gọi
    - _Requirements: 1.4, 2.4, 2.5, 3.5, 4.3, 8.5, 9.3, 10.2_

  - [x] 10.2 Viết property test — Property 1: init và reset tương đương
    - **Property 1: Trạng thái khởi tạo và reset là tương đương**
    - Với bất kỳ số frame N nào được thêm vào, sau khi gọi `reset()`, trạng thái phải giống hệt trạng thái sau `__init__`
    - **Validates: Requirements 1.1, 1.2**

  - [x] 10.3 Viết property test — Property 2: Frame Buffer bị giới hạn
    - **Property 2: Frame Buffer bị giới hạn bởi history_size**
    - Với bất kỳ N > 30 frame được thêm vào, `len(frame_buffer) <= 30` luôn đúng
    - **Validates: Requirements 1.3, 7.3**

  - [x] 10.4 Viết property test — Property 3: Blink detection kích hoạt đúng điều kiện
    - **Property 3: Blink detection kích hoạt đúng điều kiện**
    - Với chuỗi EAR bất kỳ (>= 5 giá trị), `blink_detected = True` khi và chỉ khi `ear_range > EAR_AMPLITUDE_MIN` VÀ `min(ears) < EAR_THRESHOLD`
    - Mock `face_mesh.process()` để inject EAR values trực tiếp vào buffer
    - **Validates: Requirements 2.1, 2.2**

  - [x] 10.5 Viết property test — Property 4: Detection flags là idempotent
    - **Property 4: Detection flags là idempotent (không thể đảo ngược)**
    - Khi `blink_detected = True` hoặc `movement_detected = True`, thêm bất kỳ frame nào tiếp theo không làm flag đó về `False`
    - **Validates: Requirements 2.6, 3.6**

  - [x] 10.6 Viết property test — Property 5: Movement detection kích hoạt đúng điều kiện
    - **Property 5: Movement detection kích hoạt đúng điều kiện**
    - Với chuỗi yaw/pitch bất kỳ (>= 5 giá trị), `movement_detected = True` khi và chỉ khi `yaw_range > YAW_RANGE_MIN` HOẶC `pitch_range > PITCH_RANGE_MIN`
    - **Validates: Requirements 3.4**

  - [x] 10.7 Viết property test — Property 6: Temporal consistency phụ thuộc đúng vào std EAR
    - **Property 6: Temporal consistency phụ thuộc đúng vào độ lệch chuẩn EAR**
    - Với chuỗi EAR >= 10 giá trị, `consistency_passed = True` khi và chỉ khi `TEMPORAL_STD_MIN <= std(ears) <= TEMPORAL_STD_MAX`; với buffer < 10 frame, `consistency_passed = False`
    - **Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**

  - [x] 10.8 Viết property test — Property 7: Quyết định tổng hợp và feedback động
    - **Property 7: Quyết định tổng hợp và feedback động**
    - `status = "passed"` khi và chỉ khi cả 3 flag đều `True`; khi `"pending"`, `feedback_ui` chứa đúng các gợi ý cho bước còn thiếu
    - Inject trực tiếp các giá trị flag vào instance để test logic quyết định
    - **Validates: Requirements 5.1, 5.3, 5.5**

  - [x] 10.9 Viết property test — Property 8: Interface output luôn hợp lệ
    - **Property 8: Interface output luôn hợp lệ**
    - Với bất kỳ ảnh đầu vào (ảnh trống, ảnh nhiễu, ảnh không có khuôn mặt), `update()` luôn trả về dict với `"status"` ∈ `{"pending", "passed", "failed"}` và `"feedback_ui"` là chuỗi không rỗng, không raise exception
    - **Validates: Requirements 8.1**

  - [x] 10.10 Viết property test — Property 9: Frame Buffer chỉ chứa dữ liệu số
    - **Property 9: Frame Buffer chỉ chứa dữ liệu số, không chứa ảnh**
    - Với bất kỳ frame nào được thêm vào, mỗi entry trong `frame_buffer` chỉ có keys `timestamp`, `ear`, `pitch`, `yaw` với giá trị kiểu số
    - **Validates: Requirements 7.5**

  - [x] 10.11 Viết property test — Property 10: Instance isolation
    - **Property 10: Instance isolation — các instance độc lập nhau**
    - Thay đổi trạng thái của instance A không ảnh hưởng đến instance B
    - **Validates: Requirements 7.4**

  - [x] 10.12 Viết property test — Property 11: Spoofing detection kích hoạt reset
    - **Property 11: Spoofing detection kích hoạt reset và trả về failed**
    - Với chuỗi EAR và yaw có `|correlation| > 0.95` (>= 15 frame), `update()` trả về `status="failed"` và `frame_buffer` rỗng sau đó
    - **Validates: Requirements 9.1, 9.2, 9.4**

  - [x] 10.13 Viết property test — Property 12: Cấu hình từ settings được áp dụng đúng
    - **Property 12: Cấu hình từ settings được áp dụng đúng**
    - Với các giá trị `LIVENESS_*` tùy ý hợp lệ trong settings, `LivenessService` sử dụng đúng các giá trị đó; khi không có settings, dùng đúng giá trị mặc định
    - Dùng `django.test.override_settings` để inject giá trị tùy ý
    - **Validates: Requirements 10.1, 10.2**

- [x] 11. Checkpoint — Chạy toàn bộ test suite
  - Ensure all tests pass, ask the user if questions arise.

- [x] 12. Viết integration tests cho `KioskConsumer` + `LivenessService`
  - Tạo file `attendance/tests/test_liveness_integration.py`
  - [x] 12.1 Viết integration test: consumer dispatch Celery sau khi liveness pass
    - Mock `LivenessService.update()` trả về `{"status": "passed", ...}`, kiểm tra `identify_face_task.delay()` được gọi đúng 1 lần với đúng arguments
    - _Requirements: 5.2, 8.3_
  - [x] 12.2 Viết integration test: consumer gửi `liveness_feedback` khi pending
    - Mock `LivenessService.update()` trả về `{"status": "pending", "feedback_ui": "Vui lòng chớp mắt"}`, kiểm tra WebSocket message `{"type": "liveness_feedback", "message": "Vui lòng chớp mắt"}` được gửi
    - _Requirements: 6.1_
  - [x] 12.3 Viết integration test: consumer reset sau khi dispatch thành công
    - Sau khi liveness pass và Celery task được dispatch, kiểm tra `liveness_service.reset()` được gọi và `liveness_passed = False`
    - _Requirements: 6.5_
  - [x] 12.4 Viết integration test: consumer gửi `result` khi liveness failed
    - Mock `LivenessService.update()` trả về `{"status": "failed", "feedback_ui": "Hết thời gian..."}`, kiểm tra WebSocket message `{"type": "result", "result": {"success": False, "feedback_ui": "Hết thời gian..."}}` được gửi
    - _Requirements: 6.4_

- [x] 13. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks đánh dấu `*` là optional và có thể bỏ qua để triển khai MVP nhanh hơn
- Mỗi task tham chiếu đến requirements cụ thể để đảm bảo traceability
- Property tests dùng Hypothesis với `@settings(max_examples=100)` tối thiểu; mock `face_mesh.process()` để inject landmark data trực tiếp thay vì dùng MediaPipe thật
- Integration tests dùng `channels.testing.WebsocketCommunicator` của Django Channels
- `mediapipe` chưa có trong `requirements.txt` hiện tại — Task 1 phải được thực hiện trước tất cả các task khác
- `KioskConsumer` hiện tại đã xử lý đúng `pending` và `passed`; Task 8 chỉ cần kiểm tra và fix phần `failed` reset logic
