# Hướng dẫn Setup Dự án Chấm Công Django

## 1. Cài đặt packages
```bash
pip install -r requirements.txt
```

## 2. Cấu hình Neon Database

### Bước 1: Tạo database trên Neon
1. Truy cập https://neon.tech/
2. Đăng ký/Đăng nhập tài khoản
3. Tạo project mới
4. Tạo database mới
5. Copy **Connection String** (dạng: `postgresql://user:password@ep-xxxxx.region.aws.neon.tech/dbname?sslmode=require`)

### Bước 2: Cấu hình trong file .env
Mở file `.env` và điền thông tin:
```env
DATABASE_URL=postgresql://user:password@ep-xxxxx.region.aws.neon.tech/dbname?sslmode=require
```

## 3. Cấu hình Supabase Storage

### Bước 1: Tạo project và bucket trên Supabase
1. Truy cập https://supabase.com/
2. Đăng ký/Đăng nhập
3. Tạo project mới
4. Vào **Storage** -> Create new bucket:
   - Tên bucket: `attendance-storage`
   - Public bucket: Bật (để có thể truy cập ảnh)
5. Lấy credentials:
   - Vào **Settings** -> **API**
   - Copy **Project URL** (SUPABASE_URL)
   - Copy **anon/public key** (SUPABASE_KEY)

### Bước 2: Cấu hình trong file .env
```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key
SUPABASE_BUCKET_NAME=attendance-storage
```

## 4. Chạy migrations
```bash
python manage.py makemigrations
python manage.py migrate
```

## 5. Tạo superuser
```bash
python manage.py createsuperuser
```

## 6. Chạy server
```bash
python manage.py runserver
```

## 7. Truy cập Admin
- URL: http://127.0.0.1:8000/admin
- Đăng nhập bằng tài khoản superuser vừa tạo

## Các Model trong hệ thống

### 1. WorkShift (Ca làm việc)
- name: Tên ca
- start_time: Giờ bắt đầu
- end_time: Giờ kết thúc
- late_grace_period: Số phút cho phép đi muộn

### 2. User (Người dùng)
- Kế thừa từ AbstractUser
- Thêm: role, gender, dob, phone, work_shift, avatar
- face_encoding_text: Lưu vector khuôn mặt

### 3. AttendanceLog (Nhật ký chấm công)
- user: Người chấm công
- timestamp: Thời gian
- status: ON_TIME/LATE/ABSENT
- snapshot: Ảnh chụp khi chấm công

## Ghi chú
- File ảnh sẽ được tự động upload lên Supabase khi lưu
- Nếu không cấu hình Supabase, ảnh sẽ lưu local trong thư mục `media/`
- Timezone đã được set thành `Asia/Ho_Chi_Minh`
