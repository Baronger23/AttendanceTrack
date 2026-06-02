# 📸 K2N3 Attendance Tracking System

Hệ thống chấm công thông minh sử dụng công nghệ nhận diện khuôn mặt (Face Recognition) được xây dựng bằng Django.

## 🌟 Tính năng chính

### 👨‍💼 Dành cho Admin
- **Dashboard tổng quan**: Thống kê real-time về nhân viên, chấm công hôm nay
- **Quản lý nhân viên**: Thêm, sửa, xóa nhân viên với face encoding
- **Quản lý ca làm việc**: Cấu hình giờ làm, thời gian cho phép đi muộn
- **Lịch sử chấm công**: Xem toàn bộ log chấm công với bộ lọc
- **Biểu đồ thống kê**: Chart.js visualization cho dữ liệu 7 ngày

### 👤 Dành cho Nhân viên
- **Dashboard cá nhân**: Xem thống kê chấm công của bản thân
- **Profile card**: Thông tin cá nhân, ca làm việc
- **Lịch sử chi tiết**: Danh sách ngày đi muộn, ngày vắng
- **Biểu đồ tỉ lệ**: Doughnut chart cho attendance ratio

### 🎯 Kiosk Check-in
- **Face Recognition**: Nhận diện khuôn mặt tự động
- **Camera tự động**: Bật camera ngay khi vào trang
- **UI hiện đại**: Glassmorphism design với neon effects
- **Chấm công 1 click**: Chụp và submit trong 1 nút bấm

## 🛠️ Công nghệ sử dụng

### Backend
- **Django 5.2.10**: Web framework
- **PostgreSQL**: Database (Supabase)
- **Python 3.11+**: Programming language

### AI & Computer Vision
- **dlib**: Face detection library
- **face_recognition**: Face encoding & matching
- **OpenCV**: Image processing
- **NumPy**: Numerical computing

### Frontend
- **Bootstrap 5**: Responsive UI framework
- **Chart.js**: Data visualization
- **FontAwesome 6**: Icons
- **Vanilla JavaScript**: Interactive features

### Cloud Services
- **Supabase**: PostgreSQL database + Storage
- **Supabase Storage**: Lưu trữ ảnh nhân viên & snapshots

## 📦 Cài đặt

### 1. Clone repository
```bash
git clone https://github.com/Baronger23/AttendanceTrack.git
cd AttendanceTrack
```

### 2. Tạo môi trường ảo
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 4. Cấu hình môi trường

Tạo file `.env` từ `.env.example`:
```bash
cp .env.example .env
```

Điền thông tin vào `.env`:
```env
# Django
SECRET_KEY=your-secret-key-here
DEBUG=True

# Supabase Database
DATABASE_URL=postgresql://user:password@host.supabase.co:5432/postgres

# Supabase Storage
SUPABASE_URL=https://your-project.supabase.co/
SUPABASE_KEY=your-supabase-anon-key
SUPABASE_SERVICE_ROLE_KEY=your-supabase-service-role-key
SUPABASE_BUCKET_NAME=Timekeeping
```

> 📘 **Hướng dẫn chi tiết**: Xem [SUPABASE_SETUP.md](SUPABASE_SETUP.md) để biết cách tạo Supabase project và lấy credentials.

> Lưu ý: `SUPABASE_SERVICE_ROLE_KEY` nên dùng cho upload/xóa ảnh từ Django backend; `SUPABASE_KEY` chỉ nên giữ cho các luồng public/read-only.

### 5. Chạy migrations
```bash
python manage.py migrate
```

### 6. Tạo superuser (Admin)
```bash
python manage.py createsuperuser
```

### 7. (Tùy chọn) Seed dữ liệu mẫu

**Tạo nhân viên và ca làm việc:**
```bash
python manage.py seed_data
```

**Tạo dữ liệu chấm công (30 ngày gần nhất):**
```bash
python manage.py seed_attendance --clear
```

### 8. Chạy development server
```bash
python manage.py runserver
```

Truy cập: http://127.0.0.1:8000

## 🚀 Sử dụng

### Đăng nhập
- **Admin**: http://127.0.0.1:8000/login
- **Kiosk**: http://127.0.0.1:8000/kiosk

### Tài khoản mẫu (sau khi seed)
- **Admin**: `admin` / `admin123`
- **Nhân viên**: `nhanvien1` / `123456`

### Quy trình chấm công
1. Nhân viên vào trang Kiosk
2. Camera tự động bật
3. Bấm nút "CHẤM CÔNG"
4. Hệ thống nhận diện khuôn mặt
5. Tự động ghi log với trạng thái (Đúng giờ/Đi muộn)

## 📁 Cấu trúc dự án

```
AttendanceTrack/
├── attendance/                 # Main app
│   ├── management/
│   │   └── commands/
│   │       ├── seed_data.py          # Seed users & shifts
│   │       └── seed_attendance.py    # Seed attendance logs
│   ├── templates/
│   │   └── attendance/
│   │       ├── admin_base.html       # Admin layout
│   │       ├── admin_dashboard.html  # Admin dashboard
│   │       ├── staff_dashboard.html  # Staff dashboard
│   │       ├── kiosk.html           # Check-in kiosk
│   │       ├── staff_list.html      # Staff management
│   │       └── ...
│   ├── models.py              # User, WorkShift, AttendanceLog
│   ├── views.py               # Business logic
│   └── urls.py                # URL routing
├── myproject/                 # Project settings
├── requirements.txt           # Dependencies
├── .env.example              # Environment template
└── README.md                 # This file
```

## 🎨 Screenshots

### Admin Dashboard
- Sidebar navigation
- Key metrics cards (Total Staff, Present, Late, Absent)
- 7-day bar chart
- Today's doughnut chart
- Recent activity table

### Staff Dashboard
- Profile card (Avatar, Name, Shift)
- Stats cards (Days Worked, Late Count, Absent Count)
- Attendance history table
- Attendance ratio chart
- Late days list
- Absent days list

### Kiosk Check-in
- Glassmorphism design
- Neon camera frame with glow effect
- Auto-camera activation
- Modern gradient button
- Real-time face recognition

## 🔧 Management Commands

### `seed_data`
Tạo dữ liệu mẫu (users, shifts):
```bash
python manage.py seed_data [--clear]
```

### `seed_attendance`
Tạo attendance logs (30 ngày, bỏ qua Chủ Nhật):
```bash
python manage.py seed_attendance [--clear]
```

## 📊 Database Models

### User
- Extends Django's AbstractUser
- Fields: `role`, `gender`, `dob`, `phone`, `work_shift`, `avatar`
- `face_encoding_text`: JSON-encoded face vector

### WorkShift
- Fields: `name`, `start_time`, `end_time`, `late_grace_period`

### AttendanceLog
- Fields: `user`, `timestamp`, `status`, `snapshot`
- Status: `ON_TIME`, `LATE`, `ABSENT`

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**K2N3 Team**
- GitHub: [@Baronger23](https://github.com/Baronger23)

## 🙏 Acknowledgments

- [Django](https://www.djangoproject.com/)
- [face_recognition](https://github.com/ageitgey/face_recognition)
- [Supabase](https://supabase.com/)
- [Bootstrap](https://getbootstrap.com/)
- [Chart.js](https://www.chartjs.org/)
