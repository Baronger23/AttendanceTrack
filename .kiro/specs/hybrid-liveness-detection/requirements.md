# Requirements Document

## Introduction

Hệ thống chấm công hiện tại sử dụng nhận diện khuôn mặt qua Kiosk (WebSocket + Django). Module liveness detection hiện tại (`liveness_service.py`) có hai vấn đề đối lập:

1. **Quá strict**: Ngưỡng EAR và góc xoay đầu quá nhạy, khiến người thật bình thường không chấm công được.
2. **Dễ bypass**: Chỉ cần lắc điện thoại có ảnh hoặc phát video replay là có thể qua được kiểm tra.

Tính năng **Hybrid Liveness Detection** giải quyết cả hai vấn đề bằng cách kết hợp ba lớp kiểm tra độc lập:

- **Layer 1 — Blink Detection**: Phát hiện nháy mắt thật dựa trên biên độ EAR (Eye Aspect Ratio) với MediaPipe Face Mesh 468 điểm.
- **Layer 2 — Head Movement (PnP)**: Đo góc xoay đầu 3D thật sự bằng thuật toán Perspective-n-Point, thay thế MSE đơn giản.
- **Layer 3 — Temporal Consistency**: Kiểm tra tính nhất quán theo thời gian để phân biệt người thật với video replay hoặc ảnh lắc.

Hệ thống phải đạt được cân bằng giữa bảo mật (chống spoofing) và trải nghiệm người dùng (người thật dễ pass trong ≤ 8 giây).

---

## Glossary

- **LivenessService**: Module Python trong `attendance/services/liveness_service.py` chịu trách nhiệm xác thực người thật.
- **KioskConsumer**: Django Channels WebSocket consumer trong `attendance/consumers.py`, nhận frame từ browser và gọi LivenessService.
- **EAR (Eye Aspect Ratio)**: Tỷ lệ hình học của mắt, tính từ 6 điểm landmark. Giá trị giảm mạnh khi mắt nhắm.
- **PnP (Perspective-n-Point)**: Thuật toán `cv2.solvePnP` ước lượng góc xoay đầu 3D (pitch, yaw, roll) từ các điểm 2D trên ảnh.
- **Temporal Consistency**: Tính nhất quán của các đặc trưng sinh trắc học (EAR, góc đầu) theo chuỗi thời gian.
- **Frame Buffer**: Bộ đệm lưu trữ dữ liệu của các frame gần nhất trong một phiên liveness.
- **Liveness Session**: Một phiên xác thực liveness bắt đầu khi người dùng nhấn "Chấm Công" và kết thúc khi pass hoặc timeout.
- **Spoofing Attack**: Hành vi giả mạo danh tính bằng ảnh tĩnh, video replay, hoặc ảnh trên điện thoại.
- **EAR Amplitude**: Hiệu số giữa giá trị EAR cao nhất và thấp nhất trong frame buffer — đặc trưng của nháy mắt thật.
- **Micro-variation**: Dao động nhỏ tự nhiên của EAR và góc đầu do chuyển động cơ thể người thật (thở, rung nhẹ).
- **MediaPipe Face Mesh**: Thư viện Google MediaPipe cung cấp 468 điểm landmark khuôn mặt theo thời gian thực.
- **DLIB/MTCNN**: Các detector khuôn mặt hiện có trong hệ thống, dùng cho face recognition pipeline.

---

## Requirements

### Requirement 1: Khởi tạo và Quản lý Phiên Liveness

**User Story:** Là một nhân viên, tôi muốn mỗi lần nhấn "Chấm Công" bắt đầu một phiên xác thực mới, để trạng thái từ lần trước không ảnh hưởng đến lần này.

#### Acceptance Criteria

1. WHEN người dùng bắt đầu một Liveness Session mới, THE LivenessService SHALL khởi tạo Frame Buffer rỗng, đặt tất cả cờ trạng thái (`blink_detected`, `movement_detected`, `consistency_passed`) về `False`, và ghi lại thời điểm bắt đầu phiên.
2. WHEN `reset()` được gọi, THE LivenessService SHALL xóa toàn bộ Frame Buffer và đặt lại tất cả trạng thái về giá trị khởi tạo ban đầu.
3. THE LivenessService SHALL giới hạn kích thước Frame Buffer tối đa là 30 frame để tránh tiêu tốn bộ nhớ.
4. WHEN một Liveness Session vượt quá 8 giây mà chưa pass, THE LivenessService SHALL trả về `{"status": "failed", "feedback_ui": "Hết thời gian, vui lòng thử lại."}` và tự động gọi `reset()`.

---

### Requirement 2: Layer 1 — Blink Detection bằng EAR Amplitude

**User Story:** Là một nhân viên, tôi muốn hệ thống nhận ra nháy mắt tự nhiên của tôi, để tôi không cần nháy mắt quá mạnh hay quá nhiều lần mới được chấm công.

#### Acceptance Criteria

1. WHEN Frame Buffer chứa ít nhất 5 frame, THE LivenessService SHALL tính EAR Amplitude bằng cách lấy `max(EAR) - min(EAR)` trên toàn bộ Frame Buffer.
2. WHEN EAR Amplitude vượt ngưỡng 0.08 VÀ giá trị `min(EAR)` trong buffer nhỏ hơn 0.22, THE LivenessService SHALL đặt `blink_detected = True`.
3. THE LivenessService SHALL tính EAR riêng cho mắt trái và mắt phải, sau đó lấy trung bình cộng làm giá trị EAR đại diện cho frame đó.
4. THE LivenessService SHALL sử dụng đúng 6 chỉ số landmark MediaPipe cho mỗi mắt: mắt trái `[33, 160, 158, 133, 153, 144]`, mắt phải `[362, 385, 387, 263, 373, 380]`.
5. IF MediaPipe không phát hiện được khuôn mặt trong frame, THEN THE LivenessService SHALL bỏ qua frame đó và trả về `{"status": "pending", "feedback_ui": "Không tìm thấy khuôn mặt rõ ràng."}`.
6. WHEN `blink_detected` đã là `True`, THE LivenessService SHALL không tính lại EAR Amplitude cho Layer 1 trong cùng phiên đó.

---

### Requirement 3: Layer 2 — Head Movement Detection bằng PnP

**User Story:** Là một nhân viên, tôi muốn hệ thống nhận ra chuyển động đầu tự nhiên của tôi (gật đầu nhẹ, quay nhẹ), để tôi không cần thực hiện động tác quá lớn hay gượng gạo.

#### Acceptance Criteria

1. WHEN Frame Buffer chứa ít nhất 5 frame, THE LivenessService SHALL ước lượng góc xoay đầu (pitch, yaw, roll) cho mỗi frame bằng thuật toán `cv2.solvePnP` với 6 điểm landmark 3D chuẩn.
2. THE LivenessService SHALL sử dụng đúng 6 điểm landmark MediaPipe sau để làm điểm 2D đầu vào cho PnP: nose tip (index 1), chin (index 152), left eye corner (index 33), right eye corner (index 263), left mouth corner (index 61), right mouth corner (index 291).
3. THE LivenessService SHALL xây dựng camera matrix giả định với `focal_length = image_width` và `center = (image_width/2, image_height/2)`.
4. WHEN yaw range (`max(yaw) - min(yaw)`) trong Frame Buffer vượt quá 8 độ HOẶC pitch range vượt quá 6 độ, THE LivenessService SHALL đặt `movement_detected = True`.
5. IF `cv2.solvePnP` trả về `success = False`, THEN THE LivenessService SHALL bỏ qua frame đó và không cập nhật dữ liệu góc đầu.
6. WHEN `movement_detected` đã là `True`, THE LivenessService SHALL không tính lại PnP cho Layer 2 trong cùng phiên đó.

---

### Requirement 4: Layer 3 — Temporal Consistency (Chống Video Replay và Ảnh Lắc)

**User Story:** Là quản trị viên hệ thống, tôi muốn hệ thống phân biệt được người thật với kẻ tấn công dùng video replay hoặc lắc điện thoại có ảnh, để đảm bảo tính toàn vẹn của dữ liệu chấm công.

#### Acceptance Criteria

1. WHEN Frame Buffer chứa ít nhất 10 frame, THE LivenessService SHALL tính độ lệch chuẩn (standard deviation) của chuỗi EAR trong buffer.
2. WHEN độ lệch chuẩn EAR nằm trong khoảng `[0.005, 0.08]`, THE LivenessService SHALL đánh giá temporal consistency là hợp lệ (`consistency_passed = True`).
3. IF độ lệch chuẩn EAR nhỏ hơn 0.005, THEN THE LivenessService SHALL đánh giá đây là ảnh tĩnh hoặc video quá mượt và giữ `consistency_passed = False`.
4. IF độ lệch chuẩn EAR lớn hơn 0.08, THEN THE LivenessService SHALL đánh giá đây là chuyển động giả tạo (lắc điện thoại) và giữ `consistency_passed = False`.
5. WHILE Frame Buffer chứa ít hơn 10 frame, THE LivenessService SHALL mặc định `consistency_passed = False` và tiếp tục thu thập frame.
6. THE LivenessService SHALL tính temporal consistency độc lập với Layer 1 và Layer 2, không phụ thuộc vào kết quả của hai layer kia.

---

### Requirement 5: Quyết định Liveness Tổng Hợp (Hybrid Decision)

**User Story:** Là một nhân viên, tôi muốn hệ thống đưa ra quyết định chấm công nhanh và chính xác, để tôi không phải chờ đợi lâu hay bị từ chối oan.

#### Acceptance Criteria

1. WHEN `blink_detected = True` VÀ `movement_detected = True` VÀ `consistency_passed = True`, THE LivenessService SHALL trả về `{"status": "passed", "feedback_ui": "Xác thực thành công!"}`.
2. WHEN trạng thái liveness là `"passed"`, THE KioskConsumer SHALL đặt `liveness_passed = True` và ngay lập tức dispatch Celery task nhận diện khuôn mặt.
3. WHEN trạng thái liveness là `"pending"`, THE LivenessService SHALL trả về feedback động mô tả chính xác bước nào còn thiếu trong số: "chớp mắt", "quay đầu nhẹ", "giữ nguyên tư thế".
4. IF `consistency_passed = False` sau khi đã có đủ 10 frame, THEN THE LivenessService SHALL trả về `{"status": "failed", "feedback_ui": "Phát hiện hành vi bất thường, vui lòng thử lại tự nhiên."}` và gọi `reset()`.
5. THE LivenessService SHALL đánh giá kết quả tổng hợp sau mỗi frame mới được thêm vào Frame Buffer.

---

### Requirement 6: Phản Hồi Giao Diện Người Dùng (UX Feedback)

**User Story:** Là một nhân viên, tôi muốn nhận được hướng dẫn rõ ràng trong thời gian thực khi chấm công, để tôi biết mình cần làm gì tiếp theo.

#### Acceptance Criteria

1. WHEN LivenessService trả về `{"status": "pending"}`, THE KioskConsumer SHALL gửi WebSocket message `{"type": "liveness_feedback", "message": <feedback_ui>}` đến browser trong vòng 200ms kể từ khi nhận frame.
2. THE KioskConsumer SHALL nhận frame từ browser với tần suất 5 FPS (một frame mỗi 200ms) trong suốt Liveness Session.
3. WHEN liveness_passed chuyển sang `True`, THE KioskConsumer SHALL gửi WebSocket message `{"type": "liveness_feedback", "message": "Xác thực thành công!"}` trước khi dispatch Celery task.
4. WHEN LivenessService trả về `{"status": "failed"}`, THE KioskConsumer SHALL gửi WebSocket message `{"type": "result", "result": {"success": false, "feedback_ui": <message>}}` và reset `liveness_passed = False` để cho phép thử lại.
5. THE KioskConsumer SHALL reset `liveness_service` và `liveness_passed` về trạng thái ban đầu sau mỗi lần dispatch Celery task thành công.

---

### Requirement 7: Hiệu Năng và Tài Nguyên

**User Story:** Là quản trị viên hệ thống, tôi muốn module liveness không làm nghẽn CPU của server, để hệ thống có thể phục vụ nhiều kiosk đồng thời.

#### Acceptance Criteria

1. THE LivenessService SHALL xử lý mỗi frame (MediaPipe + EAR + PnP + Temporal) trong vòng 100ms trên CPU thông thường (không cần GPU).
2. THE LivenessService SHALL sử dụng MediaPipe FaceMesh với cấu hình `static_image_mode=False`, `max_num_faces=1`, `refine_landmarks=True` để tối ưu hiệu năng streaming.
3. THE LivenessService SHALL giới hạn Frame Buffer tối đa 30 frame, tự động loại bỏ frame cũ nhất khi buffer đầy.
4. WHERE hệ thống chạy nhiều KioskConsumer đồng thời, THE LivenessService SHALL được khởi tạo riêng biệt cho mỗi WebSocket connection (không dùng singleton).
5. THE LivenessService SHALL không lưu trữ dữ liệu ảnh gốc trong Frame Buffer — chỉ lưu các giá trị số đã tính toán (EAR, pitch, yaw, timestamp).

---

### Requirement 8: Tích Hợp với Pipeline Nhận Diện Khuôn Mặt Hiện Có

**User Story:** Là developer, tôi muốn module liveness mới tích hợp liền mạch với `KioskConsumer` và `identify_face_task` hiện có, để không cần refactor toàn bộ hệ thống.

#### Acceptance Criteria

1. THE LivenessService SHALL duy trì interface `update(image: np.ndarray) -> dict` với output schema `{"status": str, "feedback_ui": str}` để tương thích ngược với `KioskConsumer`.
2. THE LivenessService SHALL duy trì method `reset()` không có tham số để `KioskConsumer` có thể gọi sau mỗi phiên.
3. WHEN liveness pass, THE KioskConsumer SHALL truyền đúng frame ảnh cuối cùng (base64) sang `identify_face_task.delay()` như hiện tại, không thay đổi signature của Celery task.
4. THE LivenessService SHALL import `mediapipe` theo cách lazy (chỉ import khi khởi tạo instance) để không làm chậm startup của Django nếu thư viện chưa được cài.
5. IF thư viện `mediapipe` chưa được cài đặt, THEN THE LivenessService SHALL raise `ImportError` với message rõ ràng hướng dẫn cài đặt: `"mediapipe is required. Run: pip install mediapipe"`.

---

### Requirement 9: Chống Spoofing — Phân Tích Micro-variation

**User Story:** Là quản trị viên bảo mật, tôi muốn hệ thống phát hiện kẻ tấn công dùng video replay chất lượng cao, để ngăn chặn bypass bằng video của người thật.

#### Acceptance Criteria

1. WHEN Frame Buffer chứa ít nhất 15 frame, THE LivenessService SHALL tính hệ số tương quan (correlation) giữa chuỗi EAR và chuỗi yaw angle trong buffer.
2. IF hệ số tương quan tuyệt đối giữa EAR và yaw vượt quá 0.95 trong 15 frame liên tiếp, THEN THE LivenessService SHALL đánh giá đây là chuyển động đồng bộ bất thường (dấu hiệu video/ảnh lắc) và trả về `{"status": "failed", "feedback_ui": "Phát hiện hành vi bất thường, vui lòng thử lại tự nhiên."}`.
3. THE LivenessService SHALL ghi log cảnh báo (WARNING level) khi phát hiện spoofing attempt, bao gồm: timestamp, correlation value, và session duration.
4. FOR ALL Liveness Sessions kết thúc với `status = "failed"` do spoofing detection, THE LivenessService SHALL gọi `reset()` để xóa toàn bộ dữ liệu phiên.

---

### Requirement 10: Cấu Hình Ngưỡng (Threshold Configuration)

**User Story:** Là quản trị viên hệ thống, tôi muốn có thể điều chỉnh các ngưỡng liveness mà không cần sửa code, để dễ dàng tinh chỉnh theo điều kiện thực tế của từng kiosk.

#### Acceptance Criteria

1. THE LivenessService SHALL đọc các ngưỡng sau từ Django settings với giá trị mặc định tương ứng nếu không được cấu hình:
   - `LIVENESS_EAR_THRESHOLD` (mặc định: `0.22`) — ngưỡng EAR tối thiểu để xác nhận mắt nhắm
   - `LIVENESS_EAR_AMPLITUDE_MIN` (mặc định: `0.08`) — biên độ EAR tối thiểu để xác nhận nháy mắt
   - `LIVENESS_YAW_RANGE_MIN` (mặc định: `8.0`) — góc yaw tối thiểu (độ) để xác nhận chuyển động đầu
   - `LIVENESS_PITCH_RANGE_MIN` (mặc định: `6.0`) — góc pitch tối thiểu (độ) để xác nhận chuyển động đầu
   - `LIVENESS_TEMPORAL_STD_MIN` (mặc định: `0.005`) — độ lệch chuẩn EAR tối thiểu
   - `LIVENESS_TEMPORAL_STD_MAX` (mặc định: `0.08`) — độ lệch chuẩn EAR tối đa
   - `LIVENESS_TIMEOUT_SECONDS` (mặc định: `8.0`) — thời gian tối đa cho một phiên
2. WHEN Django settings không định nghĩa một ngưỡng, THE LivenessService SHALL sử dụng giá trị mặc định mà không raise exception.
3. THE LivenessService SHALL đọc cấu hình một lần khi khởi tạo instance, không đọc lại trong mỗi lần gọi `update()`.
