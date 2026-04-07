from django.db import models
from django.contrib.auth.models import AbstractUser
import json
import numpy as np
from datetime import datetime, time

try:
    from pgvector.django import VectorField, HnswIndex
    PGVECTOR_AVAILABLE = True
except ImportError:
    PGVECTOR_AVAILABLE = False

# 1. BẢNG CA LÀM VIỆC (Đưa lên trên để User tham chiếu tới)
class WorkShift(models.Model):
    name = models.CharField(max_length=100, default="Ca hành chính")
    start_time = models.TimeField(default=time(8, 0))  # 08:00
    end_time = models.TimeField(default=time(17, 0))   # 17:00
    late_grace_period = models.IntegerField(default=15) # Phút cho phép trễ

    class Meta:
        unique_together = ['name', 'start_time', 'end_time']
        verbose_name = 'Ca làm việc'
        verbose_name_plural = 'Ca làm việc'

    def __str__(self):
        return f"{self.name} ({self.start_time} - {self.end_time})"

# 2. BẢNG USER (Mở rộng)
class User(AbstractUser):
    # Các trường có sẵn của AbstractUser: username, password, first_name, last_name, email...

    class Role(models.TextChoices):
        ADMIN = "ADMIN", "Quản trị viên"
        STAFF = "STAFF", "Nhân viên"
    
    class Gender(models.TextChoices):
        MALE = "M", "Nam"
        FEMALE = "F", "Nữ"
        OTHER = "O", "Khác"

    # --- Thông tin bổ sung ---
    role = models.CharField(max_length=50, choices=Role.choices, default=Role.STAFF)
    gender = models.CharField(max_length=1, choices=Gender.choices, default=Gender.MALE)
    dob = models.DateField(null=True, blank=True, verbose_name="Ngày sinh")
    phone = models.CharField(max_length=15, null=True, blank=True, verbose_name="Số điện thoại")
    
    # Liên kết ca làm việc (Mỗi nhân viên thuộc 1 ca)
    work_shift = models.ForeignKey(WorkShift, on_delete=models.SET_NULL, null=True, blank=True)

    # Lưu path trong Supabase Storage, VD: avatars/username (1).jpg
    avatar = models.CharField(max_length=500, null=True, blank=True, verbose_name='Ảnh đại diện')
    
    # --- Xử lý Vector khuôn mặt ---
    # Backward compatible: lưu trữ encoding dạng JSON text (128D hoặc 512D)
    face_encoding_text = models.TextField(null=True, blank=True)

    def get_avatar_url(self):
        """Lấy public URL của avatar từ Supabase"""
        if self.avatar:
            from django.conf import settings
            return f"{settings.SUPABASE_URL}/storage/v1/object/public/{settings.SUPABASE_BUCKET_NAME}/{self.avatar}"
        return None

    def set_encoding(self, encoding_array):
        """Lưu face encoding (hỗ trợ cả 128D và 512D)"""
        if encoding_array is not None:
            self.face_encoding_text = json.dumps(encoding_array.tolist())

    def get_encoding(self):
        """Lấy face encoding dạng numpy array"""
        if self.face_encoding_text:
            try:
                data = json.loads(self.face_encoding_text)
                return np.array(data)
            except:
                return None
        return None

    def __str__(self):
        # Hiển thị tên đầy đủ nếu có, không thì hiện username
        full_name = f"{self.last_name} {self.first_name}".strip()
        return full_name if full_name else self.username

# 3. BẢNG NHẬT KÝ CHẤM CÔNG
class AttendanceLog(models.Model):
    class Status(models.TextChoices):
        ON_TIME = "ON_TIME", "Đúng giờ"
        LATE = "LATE", "Đi muộn"
        ABSENT = "ABSENT", "Vắng mặt" # Cái này thường dùng khi chạy cronjob cuối ngày

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='logs')
    timestamp = models.DateTimeField(auto_now_add=True)
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.ON_TIME)
    snapshot = models.ImageField(upload_to='attendance_snaps/', null=True, blank=True)

    # Logic tự động tính toán Đi Muộn hay Đúng Giờ ngay khi lưu
    def save(self, *args, **kwargs):
        # Nếu chưa có status (lần tạo đầu tiên) và User có ca làm việc
        if not self.pk and self.user.work_shift:
            shift = self.user.work_shift
            check_in_time = datetime.now().time()
            
            # Logic so sánh giờ (cơ bản)
            # Chuyển đổi grace_period thành phút để cộng (Logic này cần xử lý kỹ hơn chút với datetime)
            # Ở đây mình demo logic so sánh thô:
            if check_in_time > shift.start_time:
                # Cần logic cộng phút grace_period phức tạp hơn ở view, 
                # nhưng đây là chỗ để bạn hình dung logic.
                self.status = self.Status.LATE 
            else:
                self.status = self.Status.ON_TIME
                
        super().save(*args, **kwargs)

    def __str__(self):
        return f"Check-in: {self.user.username} - {self.status}"


# 4. BẢNG THÔNG BÁO CHO ADMIN
class Notification(models.Model):
    class Type(models.TextChoices):
        WRONG_RECOGNITION = "WRONG_RECOGNITION", "Nhận diện sai"
        NEW_STAFF = "NEW_STAFF", "Nhân viên mới"
        SYSTEM = "SYSTEM", "Hệ thống"

    type = models.CharField(max_length=50, choices=Type.choices, default=Type.SYSTEM)
    title = models.CharField(max_length=200)
    message = models.TextField()
    related_user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='notifications')
    confidence = models.FloatField(null=True, blank=True)
    is_read = models.BooleanField(default=False)
    is_resolved = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    resolved_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Thông báo'
        verbose_name_plural = 'Thông báo'

    def __str__(self):
        return f"{self.get_type_display()}: {self.title}"


# 5. BẢNG LƯU TRỮ FACE EMBEDDING (pgvector)
class FaceEmbedding(models.Model):
    """
    Lưu trữ face embeddings dạng vector 512D sử dụng pgvector.
    Mỗi nhân viên có thể có nhiều embeddings (gốc + augmented).
    Sử dụng HNSW index cho tìm kiếm nhanh (miligiây thay vì giây).
    """
    class Source(models.TextChoices):
        ORIGINAL = "original", "Ảnh gốc"
        AUGMENTED = "augmented", "Ảnh tăng cường"
        REGISTRATION = "registration", "Đăng ký webcam"

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='face_embeddings')
    
    # Vector 512D — sử dụng pgvector nếu có, fallback JSON text nếu không
    if PGVECTOR_AVAILABLE:
        embedding = VectorField(dimensions=512)
    else:
        embedding_json = models.TextField(default='[]')
    
    source = models.CharField(
        max_length=20, 
        choices=Source.choices, 
        default=Source.ORIGINAL
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = 'Face Embedding'
        verbose_name_plural = 'Face Embeddings'
        if PGVECTOR_AVAILABLE:
            indexes = [
                HnswIndex(
                    name='face_emb_hnsw_idx',
                    fields=['embedding'],
                    m=16,
                    ef_construction=64,
                    opclasses=['vector_cosine_ops'],
                ),
            ]

    def set_embedding(self, embedding_array):
        """Lưu embedding từ numpy array"""
        if PGVECTOR_AVAILABLE:
            self.embedding = embedding_array.tolist()
        else:
            self.embedding_json = json.dumps(embedding_array.tolist())

    def get_embedding(self):
        """Lấy embedding dạng numpy array"""
        if PGVECTOR_AVAILABLE:
            return np.array(self.embedding)
        else:
            try:
                return np.array(json.loads(self.embedding_json))
            except:
                return None

    def __str__(self):
        return f"Embedding: {self.user.username} ({self.get_source_display()})"
