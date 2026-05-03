from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.utils import timezone
from datetime import timedelta
from .models import User, WorkShift, AttendanceLog, Notification


# ==================== AUTHENTICATION ====================

def login_view(request):
    """Trang đăng nhập"""
    if request.user.is_authenticated:
        return redirect('dashboard')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request, user)
            return redirect('dashboard')
        else:
            messages.error(request, 'Tên đăng nhập hoặc mật khẩu không đúng!')
    
    return render(request, 'attendance/login.html')


def logout_view(request):
    """Đăng xuất"""
    logout(request)
    messages.success(request, 'Đã đăng xuất thành công!')
    return redirect('login')


@login_required
def open_kiosk(request):
    """Đăng xuất admin và chuyển sang trang Kiosk"""
    logout(request)
    return redirect('kiosk')


# ==================== DASHBOARD ====================

@login_required
def dashboard(request):
    """Trang chủ sau khi đăng nhập"""
    if request.user.role == User.Role.ADMIN:
        return redirect('admin_dashboard')
    else:
        return redirect('staff_dashboard')


@login_required
def admin_dashboard(request):
    """Dashboard cho Admin"""
    if request.user.role != User.Role.ADMIN:
        messages.error(request, 'Bạn không có quyền truy cập trang quản trị!')
        return redirect('staff_dashboard')
    
    today = timezone.now().date()
    
    # --- Key Metrics ---
    total_staff = User.objects.filter(role=User.Role.STAFF).count()
    
    # Số người đi làm hôm nay (Unique users)
    present_today = AttendanceLog.objects.filter(
        timestamp__date=today
    ).values('user').distinct().count()
    
    # Số người đi muộn hôm nay
    late_today = AttendanceLog.objects.filter(
        timestamp__date=today,
        status=AttendanceLog.Status.LATE
    ).count()
    
    # Số người vắng mặt (Giả sử đi làm hết nếu không có log)
    absent_today = total_staff - present_today
    if absent_today < 0: absent_today = 0

    # --- Charts Data ---
    
    # 1. Biểu đồ 7 ngày gần nhất
    labels_7_days = []
    data_7_days = []
    
    for i in range(6, -1, -1):
        date = today - timedelta(days=i)
        day_str = date.strftime('%d/%m')
        count = AttendanceLog.objects.filter(timestamp__date=date).values('user').distinct().count()
        labels_7_days.append(day_str)
        data_7_days.append(count)
        
    # 2. Biểu đồ tròn hôm nay (Doughnut)
    on_time_today = AttendanceLog.objects.filter(
        timestamp__date=today, 
        status=AttendanceLog.Status.ON_TIME
    ).count()
    
    pie_data = [on_time_today, late_today, absent_today]
    
    # --- Recent Activity ---
    recent_logs = AttendanceLog.objects.select_related('user').order_by('-timestamp')[:10]
    
    context = {
        'total_staff': total_staff,
        'present_today': present_today,
        'late_today': late_today,
        'absent_today': absent_today,
        'labels_7_days': labels_7_days,
        'data_7_days': data_7_days,
        'pie_data': pie_data,
        'recent_logs': recent_logs,
    }
    return render(request, 'attendance/admin_dashboard.html', context)


@login_required
def staff_dashboard(request):
    """Dashboard cho Nhân viên"""
    user = request.user
    
    # Lấy logs gần nhất
    my_logs = AttendanceLog.objects.filter(user=user).order_by('-timestamp')[:20]
    
    # Tính toán thống kê
    # 1. Tổng số ngày đã đi làm (unique dates)
    days_worked = AttendanceLog.objects.filter(user=user).values('timestamp__date').distinct().count()
    
    # 2. Số lần đi muộn
    late_count = AttendanceLog.objects.filter(user=user, status=AttendanceLog.Status.LATE).count()
    
    # 3. Số lần đúng giờ
    on_time_count = AttendanceLog.objects.filter(user=user, status=AttendanceLog.Status.ON_TIME).count()
    
    # 4. Số ngày vắng (chỉ đếm record ABSENT thực tế trong DB)
    absent_count = AttendanceLog.objects.filter(user=user, status=AttendanceLog.Status.ABSENT).count()
    
    # Dữ liệu cho biểu đồ tròn
    chart_data = [on_time_count, late_count, absent_count]
    
    # Lấy danh sách ngày đi muộn
    late_logs = AttendanceLog.objects.filter(user=user, status=AttendanceLog.Status.LATE).order_by('-timestamp')
    
    # Lấy danh sách ngày vắng
    absent_logs = AttendanceLog.objects.filter(user=user, status=AttendanceLog.Status.ABSENT).order_by('-timestamp')
    
    context = {
        'my_logs': my_logs,
        'days_worked': days_worked,
        'late_count': late_count,
        'on_time_count': on_time_count,
        'absent_count': absent_count,
        'chart_data': chart_data,
        'late_logs': late_logs,
        'absent_logs': absent_logs,
    }
    return render(request, 'attendance/staff_dashboard.html', context)


# ==================== QUẢN LÝ NHÂN VIÊN (Admin only) ====================

@login_required
def staff_list(request):
    """Danh sách nhân viên với tìm kiếm đa năng"""
    if request.user.role != User.Role.ADMIN:
        messages.error(request, 'Bạn không có quyền truy cập!')
        return redirect('staff_dashboard')
    
    # Loại trừ chính mình khỏi danh sách để tránh tự xóa
    staff_members = User.objects.filter(role=User.Role.STAFF).exclude(id=request.user.id).select_related('work_shift')
    
    # Xử lý tìm kiếm
    search_query = request.GET.get('search', '').strip()
    if search_query:
        from django.db.models import Q
        
        # Map từ tiếng Việt sang mã giới tính
        gender_map = {
            'nam': 'M',
            'nữ': 'F',
            'nu': 'F',
            'khác': 'O',
            'khac': 'O',
        }
        
        # Tìm kiếm theo nhiều trường
        query = Q(username__icontains=search_query) | \
                Q(first_name__icontains=search_query) | \
                Q(last_name__icontains=search_query) | \
                Q(email__icontains=search_query) | \
                Q(phone__icontains=search_query)
        
        # Tìm kiếm theo giới tính (tiếng Việt)
        search_lower = search_query.lower()
        if search_lower in gender_map:
            query |= Q(gender=gender_map[search_lower])
        
        staff_members = staff_members.filter(query)
    
    context = {
        'staff_members': staff_members,
        'search_query': search_query
    }
    return render(request, 'attendance/staff_list.html', context)


@login_required
def staff_create(request):
    """Tạo nhân viên mới"""
    if request.user.role != User.Role.ADMIN:
        messages.error(request, 'Bạn không có quyền truy cập!')
        return redirect('staff_dashboard')
    
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        gender = request.POST.get('gender', 'M')
        work_shift_id = request.POST.get('work_shift')
        
        if User.objects.filter(username=username).exists():
            messages.error(request, 'Tên đăng nhập đã tồn tại!')
        else:
            user = User.objects.create_user(
                username=username,
                password=password,
                first_name=first_name,
                last_name=last_name,
                email=email,
                phone=phone,
                gender=gender,
                role=User.Role.STAFF
            )
            if work_shift_id:
                user.work_shift_id = work_shift_id
            
            # Xử lý upload nhiều ảnh lên Supabase Storage
            face_images = request.FILES.getlist('face_images')
            if face_images:
                try:
                    from supabase import create_client
                    import os
                    from django.conf import settings
                    import cv2
                    import numpy as np
                    from attendance.services.face_service import FaceService
                    from attendance.models import FaceEmbedding
                    
                    # Đảm bảo SUPABASE_URL có trailing slash
                    supabase_url = settings.SUPABASE_URL
                    if not supabase_url.endswith('/'):
                        supabase_url += '/'
                    
                    # Khởi tạo Supabase client
                    supabase = create_client(
                        supabase_url,
                        settings.SUPABASE_KEY
                    )
                    
                    # Upload ảnh lên Supabase Storage
                    cv_images = []
                    for idx, image in enumerate(face_images, start=1):
                        ext = os.path.splitext(image.name)[1] or '.jpg'
                        file_name = f"{username} ({idx}){ext}"
                        file_path = f"avatars/{file_name}"
                        
                        image.seek(0)
                        file_content = image.read()
                        
                        supabase.storage.from_(settings.SUPABASE_BUCKET_NAME).upload(
                            path=file_path,
                            file=file_content,
                            file_options={
                                "content-type": image.content_type,
                                "upsert": "true"
                            }
                        )
                        
                        # Decode image for face processing
                        image.seek(0)
                        img_array = np.frombuffer(image.read(), np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        if img is not None:
                            cv_images.append(img)
                    
                    # Lưu path ảnh đầu tiên vào avatar field
                    user.avatar = f"avatars/{username} (1){os.path.splitext(face_images[0].name)[1] or '.jpg'}"
                    
                    # === CNN Pipeline: Detect → Align → Augment → Embed ===
                    if cv_images:
                        face_service = FaceService()
                        result = face_service.register_multiple_images(cv_images)
                        
                        if result['success']:
                            # Lưu embedding trung bình vào User (backward compatible)
                            user.set_encoding(result['embedding'])
                            
                            # Lưu tất cả embeddings vào FaceEmbedding table (pgvector)
                            for i, emb in enumerate(result['all_embeddings']):
                                fe = FaceEmbedding(
                                    user=user,
                                    source='original' if i == 0 else 'augmented',
                                )
                                fe.set_embedding(np.array(emb))
                                fe.save()
                            
                            messages.success(request, f'✅ Đã upload {len(face_images)} ảnh và tạo {len(result["all_embeddings"])} face embeddings (CNN 512D)!')
                        else:
                            messages.warning(request, f'⚠️ {result["error"]}')
                    
                    user.save()
                    
                except Exception as e:
                    messages.warning(request, f'Upload ảnh thất bại: {str(e)}')
                    user.save()
            else:
                user.save()
            
            messages.success(request, f'Đã tạo nhân viên {username} thành công!')
            return redirect('staff_list')
    
    work_shifts = WorkShift.objects.all()
    context = {'work_shifts': work_shifts}
    return render(request, 'attendance/staff_form.html', context)


@login_required
def staff_edit(request, pk):
    """Chỉnh sửa thông tin nhân viên"""
    if request.user.role != User.Role.ADMIN:
        messages.error(request, 'Bạn không có quyền truy cập!')
        return redirect('staff_dashboard')
    
    staff = get_object_or_404(User, pk=pk)
    
    if request.method == 'POST':
        staff.first_name = request.POST.get('first_name')
        staff.last_name = request.POST.get('last_name')
        staff.email = request.POST.get('email')
        staff.phone = request.POST.get('phone')
        staff.gender = request.POST.get('gender', 'M')
        work_shift_id = request.POST.get('work_shift')
        
        if work_shift_id:
            staff.work_shift_id = work_shift_id
        
        # Xử lý upload lại ảnh mới (nếu có)
        face_images = request.FILES.getlist('face_images')
        if face_images:
            try:
                from supabase import create_client
                import os
                from django.conf import settings
                import cv2
                import numpy as np
                
                # Đảm bảo SUPABASE_URL có trailing slash
                supabase_url = settings.SUPABASE_URL
                if not supabase_url.endswith('/'):
                    supabase_url += '/'
                
                # Khởi tạo Supabase client
                supabase = create_client(
                    supabase_url,
                    settings.SUPABASE_KEY
                )
                
                # Xóa ảnh cũ trước (tìm tất cả ảnh có pattern username (1), username (2), ...)
                username = staff.username
                try:
                    # List tất cả files trong folder avatars
                    files = supabase.storage.from_(settings.SUPABASE_BUCKET_NAME).list('avatars')
                    
                    # Xóa các file có tên bắt đầu bằng username
                    files_to_delete = []
                    for file in files:
                        if file['name'].startswith(f"{username} ("):
                            files_to_delete.append(f"avatars/{file['name']}")
                    
                    if files_to_delete:
                        supabase.storage.from_(settings.SUPABASE_BUCKET_NAME).remove(files_to_delete)
                        messages.info(request, f'Đã xóa {len(files_to_delete)} ảnh cũ của {username}')
                except Exception as e:
                    messages.warning(request, f'Không thể xóa ảnh cũ: {str(e)}')
                
                # Upload ảnh mới và decode face encoding
                cv_images = []
                
                for idx, image in enumerate(face_images, start=1):
                    ext = os.path.splitext(image.name)[1] or '.jpg'
                    file_name = f"{username} ({idx}){ext}"
                    file_path = f"avatars/{file_name}"
                    
                    # Upload lên Supabase
                    image.seek(0)
                    file_content = image.read()
                    
                    supabase.storage.from_(settings.SUPABASE_BUCKET_NAME).upload(
                        path=file_path,
                        file=file_content,
                        file_options={
                            "content-type": image.content_type,
                            "upsert": "true"
                        }
                    )
                    
                    # Decode image for face processing
                    image.seek(0)
                    img_array = np.frombuffer(image.read(), np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img is not None:
                        cv_images.append(img)
                
                # Cập nhật avatar path
                staff.avatar = f"avatars/{username} (1){os.path.splitext(face_images[0].name)[1] or '.jpg'}"
                
                # === CNN Pipeline: Detect → Align → Augment → Embed ===
                if cv_images:
                    from attendance.services.face_service import FaceService
                    from attendance.models import FaceEmbedding
                    import numpy as np
                    
                    face_service = FaceService()
                    result = face_service.register_multiple_images(cv_images)
                    
                    if result['success']:
                        # Lưu embedding trung bình vào User (backward compatible)
                        staff.set_encoding(result['embedding'])
                        
                        # Xóa embeddings cũ và lưu mới vào FaceEmbedding table
                        FaceEmbedding.objects.filter(user=staff).delete()
                        for i, emb in enumerate(result['all_embeddings']):
                            fe = FaceEmbedding(
                                user=staff,
                                source='original' if i == 0 else 'augmented',
                            )
                            fe.set_embedding(np.array(emb))
                            fe.save()
                        
                        messages.success(request, f'✅ Đã upload {len(face_images)} ảnh và cập nhật {len(result["all_embeddings"])} face embeddings (CNN 512D)!')
                    else:
                        messages.warning(request, f'⚠️ {result["error"]}')
                    
            except Exception as e:
                messages.warning(request, f'Upload ảnh thất bại: {str(e)}')
        
        staff.save()
        messages.success(request, 'Đã cập nhật thông tin nhân viên!')
        return redirect('staff_list')
    
    work_shifts = WorkShift.objects.all()
    context = {'staff': staff, 'work_shifts': work_shifts}
    return render(request, 'attendance/staff_form.html', context)


@login_required
def staff_delete(request, pk):
    """Xóa nhân viên"""
    if request.user.role != User.Role.ADMIN:
        messages.error(request, 'Bạn không có quyền truy cập!')
        return redirect('staff_dashboard')
    
    staff = get_object_or_404(User, pk=pk)
    if request.method == 'POST':
        # Xóa ảnh trên Supabase Storage trước khi xóa user
        try:
            from supabase import create_client
            from django.conf import settings
            
            supabase = create_client(
                settings.SUPABASE_URL,
                settings.SUPABASE_KEY
            )
            
            # Tìm và xóa tất cả ảnh của nhân viên này
            username = staff.username
            try:
                files = supabase.storage.from_(settings.SUPABASE_BUCKET_NAME).list('avatars')
                files_to_delete = []
                for file in files:
                    if file['name'].startswith(f"{username} ("):
                        files_to_delete.append(f"avatars/{file['name']}")
                
                if files_to_delete:
                    supabase.storage.from_(settings.SUPABASE_BUCKET_NAME).remove(files_to_delete)
                    messages.info(request, f'Đã xóa {len(files_to_delete)} ảnh từ Supabase Storage')
            except Exception as e:
                messages.warning(request, f'Không thể xóa ảnh trên Supabase: {str(e)}')
        except Exception as e:
            messages.warning(request, f'Lỗi kết nối Supabase: {str(e)}')
        
        staff.delete()
        messages.success(request, 'Đã xóa nhân viên!')
        return redirect('staff_list')
    
    context = {'staff': staff}
    return render(request, 'attendance/staff_confirm_delete.html', context)


# ==================== QUẢN LÝ CA LÀM VIỆC (Admin only) ====================

@login_required
def shift_list(request):
    """Danh sách ca làm việc"""
    if request.user.role != User.Role.ADMIN:
        messages.error(request, 'Bạn không có quyền truy cập!')
        return redirect('staff_dashboard')
    
    shifts = WorkShift.objects.all()
    context = {'shifts': shifts}
    return render(request, 'attendance/shift_list.html', context)


@login_required
def shift_create(request):
    """Tạo ca làm việc mới"""
    if request.user.role != User.Role.ADMIN:
        messages.error(request, 'Bạn không có quyền truy cập!')
        return redirect('staff_dashboard')
    
    if request.method == 'POST':
        name = request.POST.get('name')
        start_time = request.POST.get('start_time')
        end_time = request.POST.get('end_time')
        late_grace_period = request.POST.get('late_grace_period', 15)
        
        # Kiểm tra ca làm việc đã tồn tại chưa
        if WorkShift.objects.filter(name=name, start_time=start_time, end_time=end_time).exists():
            messages.error(request, f'Ca làm việc "{name}" ({start_time} - {end_time}) đã tồn tại!')
            return render(request, 'attendance/shift_form.html')
        
        WorkShift.objects.create(
            name=name,
            start_time=start_time,
            end_time=end_time,
            late_grace_period=late_grace_period
        )
        messages.success(request, 'Đã tạo ca làm việc mới!')
        return redirect('shift_list')
    
    return render(request, 'attendance/shift_form.html')


@login_required
def shift_edit(request, pk):
    """Chỉnh sửa ca làm việc"""
    if request.user.role != User.Role.ADMIN:
        messages.error(request, 'Bạn không có quyền truy cập!')
        return redirect('staff_dashboard')
    
    shift = get_object_or_404(WorkShift, pk=pk)
    
    if request.method == 'POST':
        name = request.POST.get('name')
        start_time = request.POST.get('start_time')
        end_time = request.POST.get('end_time')
        late_grace_period = request.POST.get('late_grace_period', 15)
        
        # Kiểm tra ca làm việc trùng lặp (trừ chính nó)
        if WorkShift.objects.filter(
            name=name, 
            start_time=start_time, 
            end_time=end_time
        ).exclude(pk=pk).exists():
            messages.error(request, f'Ca làm việc "{name}" ({start_time} - {end_time}) đã tồn tại!')
            context = {'shift': shift}
            return render(request, 'attendance/shift_form.html', context)
        
        shift.name = name
        shift.start_time = start_time
        shift.end_time = end_time
        shift.late_grace_period = late_grace_period
        shift.save()
        messages.success(request, 'Đã cập nhật ca làm việc!')
        return redirect('shift_list')
    
    context = {'shift': shift}
    return render(request, 'attendance/shift_form.html', context)


@login_required
def shift_delete(request, pk):
    """Xóa ca làm việc"""
    if request.user.role != User.Role.ADMIN:
        messages.error(request, 'Bạn không có quyền truy cập!')
        return redirect('staff_dashboard')
    
    shift = get_object_or_404(WorkShift, pk=pk)
    if request.method == 'POST':
        shift.delete()
        messages.success(request, 'Đã xóa ca làm việc!')
        return redirect('shift_list')
    
    context = {'shift': shift}
    return render(request, 'attendance/shift_confirm_delete.html', context)


# ==================== KIOSK CHẤM CÔNG (Public - Không cần login) ====================

def kiosk_checkin(request):
    """Màn hình kiosk chấm công công cộng"""
    if request.method == 'POST':
        uploaded_image = request.FILES.get('face_image')
        
        if not uploaded_image:
            messages.error(request, '⚠️ Vui lòng chụp ảnh!')
            return render(request, 'attendance/kiosk.html')
        
        try:
            import cv2
            import numpy as np
            from attendance.services.face_service import FaceService
            
            # Đọc ảnh upload
            img_array = np.frombuffer(uploaded_image.read(), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                messages.error(request, '❌ Không thể đọc ảnh! Vui lòng thử lại.')
                return render(request, 'attendance/kiosk.html')
            
            # === CNN Pipeline: Detect (MTCNN) → Align → CLAHE → Embed (512D) → SVM/pgvector ===
            face_service = FaceService()
            result = face_service.identify_face(img)
            
            if not result['success']:
                messages.error(request, f'❌ {result["error"]}')
                return render(request, 'attendance/kiosk.html')
            
            # Lấy thông tin nhân viên từ kết quả nhận diện
            best_match = User.objects.get(id=result['user_id'])
            confidence = result['confidence']
            
            # Tìm thấy nhân viên - Kiểm tra ca làm việc
            today = timezone.now().date()
            now_time = timezone.localtime(timezone.now()).time()
            
            # Kiểm tra nhân viên có đang trong ca làm việc không
            if best_match.work_shift:
                shift = best_match.work_shift
                from datetime import timedelta, datetime as dt
                
                # Cho phép chấm công sớm 60 phút trước ca và trễ 30 phút sau ca
                EARLY_MINUTES = 60
                LATE_AFTER_END_MINUTES = 30
                
                # Chuyển time thành datetime để tính toán
                today_dt = timezone.localtime(timezone.now()).date()
                shift_start_dt = dt.combine(today_dt, shift.start_time)
                shift_end_dt = dt.combine(today_dt, shift.end_time)
                
                # Tính khoảng cho phép
                allowed_start = (shift_start_dt - timedelta(minutes=EARLY_MINUTES)).time()
                allowed_end = (shift_end_dt + timedelta(minutes=LATE_AFTER_END_MINUTES)).time()
                
                # Xử lý ca đêm (ví dụ: 22:00 - 06:00)
                if shift.start_time > shift.end_time:
                    # Ca đêm: cho phép từ (start - 60min) đến hết ngày HOẶC từ đầu ngày đến (end + 30min)
                    in_shift = now_time >= allowed_start or now_time <= allowed_end
                else:
                    # Ca bình thường: chấm công trong khoảng [start - 60min, end + 30min]
                    in_shift = allowed_start <= now_time <= allowed_end
                
                if not in_shift:
                    messages.error(
                        request, 
                        f'❌ Chưa đến ca của bạn! Ca "{shift.name}" bắt đầu lúc {shift.start_time.strftime("%H:%M")} - {shift.end_time.strftime("%H:%M")}. '
                        f'Bạn có thể chấm công từ {allowed_start.strftime("%H:%M")}.'
                    )
                    # Vẫn trả về success view để hiện thông tin nhân viên + thông báo lỗi
                    context = {
                        'success': True,
                        'user': best_match,
                        'confidence': confidence
                    }
                    return render(request, 'attendance/kiosk.html', context)
            
            # Kiểm tra đã chấm công hôm nay chưa
            existing_log = AttendanceLog.objects.filter(
                user=best_match,
                timestamp__date=today
            ).first()
            
            if existing_log:
                messages.warning(request, f'⚠️ {best_match.get_full_name()} đã chấm công hôm nay lúc {existing_log.timestamp.strftime("%H:%M")}!')
            else:
                # Tạo log mới
                log = AttendanceLog.objects.create(user=best_match)
                
                messages.success(request, f'✅ Chấm công thành công! Xin chào {best_match.get_full_name()} - {log.get_status_display()}')
            
            context = {
                'success': True,
                'user': best_match,
                'confidence': confidence
            }
            return render(request, 'attendance/kiosk.html', context)
                
        except User.DoesNotExist:
            messages.error(request, '❌ Không tìm thấy nhân viên trong hệ thống!')
            return render(request, 'attendance/kiosk.html')
        except Exception as e:
            messages.error(request, f'❌ Lỗi: {str(e)}')
            return render(request, 'attendance/kiosk.html')
    
    return render(request, 'attendance/kiosk.html')


# ==================== ASYNC KIOSK API ====================

from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import json
import base64


@csrf_exempt
def kiosk_checkin_async(request):
    """
    AJAX endpoint for async face check-in.
    Dispatches the CNN pipeline to Celery and returns task_id instantly.
    Falls back to synchronous processing if Celery/Redis unavailable.
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'Method not allowed'}, status=405)
    
    # Get base64 image from AJAX request
    try:
        body = json.loads(request.body)
        image_data = body.get('image')
        previous_image_data = body.get('previous_image')
    except (json.JSONDecodeError, AttributeError):
        image_data = None
        previous_image_data = None
    
    if not image_data:
        return JsonResponse({'success': False, 'error': 'Không có dữ liệu ảnh!'})
    
    # Strip data URL prefix if present (data:image/jpeg;base64,...)
    if ',' in image_data:
        image_data = image_data.split(',', 1)[1]
        
    if previous_image_data and ',' in previous_image_data:
        previous_image_data = previous_image_data.split(',', 1)[1]
    
    # Try async (Celery) first, fall back to sync
    try:
        from attendance.tasks import identify_face_task
        task = identify_face_task.delay(image_data, previous_image_data)
        return JsonResponse({
            'success': True,
            'task_id': task.id,
            'mode': 'async',
        })
    except Exception as e:
        # Celery/Redis unavailable — run synchronously
        import logging
        logging.getLogger(__name__).warning(f"Celery unavailable, running sync: {e}")
        
        try:
            import cv2
            import numpy as np
            from attendance.services.face_service import FaceService
            
            img_bytes = base64.b64decode(image_data)
            img_array = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                return JsonResponse({'success': False, 'error': 'Không thể đọc ảnh!'})
            
            face_service = FaceService()
            
            # [ANTI-SPOOFING] Liveness Check if previous frame is provided
            if previous_image_data:
                prev_img_bytes = base64.b64decode(previous_image_data)
                prev_img_array = np.frombuffer(prev_img_bytes, np.uint8)
                prev_img = cv2.imdecode(prev_img_array, cv2.IMREAD_COLOR)
                
                if prev_img is not None:
                    is_live = face_service.verify_liveness(img, prev_img)
                    if not is_live:
                        return JsonResponse({
                            'success': False, 
                            'error': 'Phát hiện ảnh tĩnh/giả mạo!',
                            'feedback_ui': 'Cảnh báo: Phát hiện khuôn mặt không có vi biểu cảm thật. Vui lòng thử lại.',
                            'mode': 'sync'
                        })
            
            result = face_service.identify_face(img)
            
            if not result['success']:
                return JsonResponse({
                    'success': False,
                    'error': result.get('error', 'Lỗi không xác định'),
                    'feedback_ui': result.get('feedback_ui', 'Lỗi hệ thống, vui lòng thử lại.'),
                    'mode': 'sync',
                })
            
            user = User.objects.get(id=result['user_id'])
            today = timezone.now().date()
            
            # Check existing log
            existing_log = AttendanceLog.objects.filter(
                user=user, timestamp__date=today
            ).first()
            
            if existing_log:
                checkin_status = 'already_checked'
                msg = f'{user.get_full_name()} đã chấm công hôm nay lúc {existing_log.timestamp.strftime("%H:%M")}!'
            else:
                log = AttendanceLog.objects.create(user=user)
                checkin_status = 'success'
                msg = f'Chấm công thành công! Xin chào {user.get_full_name()} - {log.get_status_display()}'
            
            return JsonResponse({
                'success': True,
                'mode': 'sync',
                'result': {
                    'success': True,
                    'user_id': user.id,
                    'user_full_name': str(user),
                    'first_initial': (user.first_name[:1].upper() if user.first_name else '?'),
                    'confidence': result['confidence'],
                    'avatar_url': user.get_avatar_url() or '',
                    'shift_name': user.work_shift.name if user.work_shift else '',
                    'checkin_status': checkin_status,
                    'message': msg,
                }
            })
        except Exception as sync_e:
            return JsonResponse({'success': False, 'error': str(sync_e), 'mode': 'sync'})


def check_result(request, task_id):
    """Poll Celery task result by task_id."""
    try:
        from celery.result import AsyncResult
        result = AsyncResult(task_id)
        
        if result.ready():
            task_result = result.get(timeout=1)
            return JsonResponse({
                'status': 'done',
                'result': task_result,
            })
        elif result.state == 'STARTED':
            return JsonResponse({'status': 'processing'})
        elif result.state == 'PENDING':
            return JsonResponse({'status': 'pending'})
        else:
            return JsonResponse({'status': 'unknown', 'state': result.state})
    except Exception as e:
        return JsonResponse({'status': 'error', 'error': str(e)})


# ==================== BÁO CÁO NHẬN DIỆN SAI ====================

def report_error(request):
    """Xử lý báo cáo khi nhận diện sai người"""
    if request.method == 'POST':
        wrong_user_id = request.POST.get('wrong_user_id')
        confidence = request.POST.get('confidence')
        # Xử lý dấu phẩy thập phân (định dạng Việt Nam: 73,7 → 73.7)
        if confidence:
            confidence = confidence.replace(',', '.')
        description = request.POST.get('description', '')
        
        try:
            wrong_user = User.objects.get(id=wrong_user_id)
            today = timezone.now().date()
            
            # Xóa log chấm công sai (nếu có)
            deleted_count, _ = AttendanceLog.objects.filter(
                user=wrong_user,
                timestamp__date=today
            ).delete()
            
            # Tạo thông báo cho admin
            Notification.objects.create(
                type=Notification.Type.WRONG_RECOGNITION,
                title=f"Báo cáo nhận diện sai - {wrong_user.get_full_name()}",
                message=f"""Có báo cáo nhận diện sai người.
- Người bị nhận nhầm: {wrong_user.get_full_name()} ({wrong_user.username})
- Độ chính xác: {confidence}%
- Mô tả từ người dùng: {description if description else 'Không có mô tả'}
- Đã tự động xóa {deleted_count} log chấm công sai""",
                related_user=wrong_user,
                confidence=float(confidence) if confidence else None,
            )
            
            messages.success(request, f'✅ Đã gửi báo cáo thành công! Log chấm công sai đã được xóa. Quản trị viên sẽ xem xét.')
            
        except User.DoesNotExist:
            messages.error(request, '❌ Không tìm thấy thông tin người dùng.')
        except Exception as e:
            messages.error(request, f'❌ Lỗi: {str(e)}')
    
    return redirect('kiosk')


# ==================== QUẢN LÝ THÔNG BÁO ====================

@login_required
def notification_list(request):
    """Danh sách tất cả thông báo"""
    if request.user.role != User.Role.ADMIN:
        return redirect('staff_dashboard')
    
    notifications = Notification.objects.all()
    
    # Đánh dấu tất cả là đã đọc
    if request.GET.get('mark_all_read'):
        Notification.objects.filter(is_read=False).update(is_read=True)
        messages.success(request, '✅ Đã đánh dấu tất cả là đã đọc')
        return redirect('notification_list')
    
    context = {
        'page_title': 'Thông báo',
        'all_notifications': notifications,
    }
    return render(request, 'attendance/notification_list.html', context)


@login_required
def notification_detail(request, pk):
    """Xem chi tiết và xử lý thông báo"""
    if request.user.role != User.Role.ADMIN:
        return redirect('staff_dashboard')
    
    notification = get_object_or_404(Notification, pk=pk)
    
    # Đánh dấu là đã đọc
    if not notification.is_read:
        notification.is_read = True
        notification.save()
    
    # Xử lý resolve
    if request.method == 'POST':
        if 'resolve' in request.POST:
            notification.is_resolved = True
            notification.resolved_at = timezone.now()
            notification.save()
            messages.success(request, '✅ Đã xử lý thông báo!')
            return redirect('notification_list')
        elif 'delete' in request.POST:
            notification.delete()
            messages.success(request, '✅ Đã xóa thông báo!')
            return redirect('notification_list')
    
    context = {
        'page_title': f'Thông báo: {notification.title}',
        'notification': notification,
    }
    return render(request, 'attendance/notification_detail.html', context)


# ==================== ĐĂNG KÝ KHUÔN MẶT NHANH ====================

@login_required
def face_register(request):
    """Đăng ký khuôn mặt cho nhân viên đang đăng nhập"""
    if request.method == 'POST':
        uploaded_image = request.FILES.get('face_image')
        
        if not uploaded_image:
            messages.error(request, '⚠️ Vui lòng chụp ảnh!')
            return render(request, 'attendance/face_register.html')
        
        try:
            import cv2
            import numpy as np
            from attendance.services.face_service import FaceService
            from attendance.models import FaceEmbedding
            
            # Đọc ảnh upload
            img_array = np.frombuffer(uploaded_image.read(), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                messages.error(request, '❌ Không thể đọc ảnh!')
                return render(request, 'attendance/face_register.html')
            
            # === CNN Pipeline: Detect (MTCNN) → Align → CLAHE → Augment → Embed (512D) ===
            face_service = FaceService()
            result = face_service.register_face(img)
            
            if not result['success']:
                messages.error(request, f'❌ {result["error"]}')
                return render(request, 'attendance/face_register.html')
            
            # Lưu encoding vào user hiện tại (backward compatible)
            request.user.set_encoding(result['embedding'])
            request.user.save()
            
            # Lưu vào FaceEmbedding table (pgvector)
            FaceEmbedding.objects.filter(user=request.user).delete()
            for i, emb in enumerate(result['all_embeddings']):
                fe = FaceEmbedding(
                    user=request.user,
                    source='registration' if i == 0 else 'augmented',
                )
                fe.set_embedding(np.array(emb))
                fe.save()
            
            messages.success(request, f'✅ Đã đăng ký khuôn mặt thành công cho {request.user.get_full_name() or request.user.username}! ({len(result["all_embeddings"])} embeddings CNN 512D)')
            return render(request, 'attendance/face_register.html', {'success': True})
            
        except Exception as e:
            messages.error(request, f'❌ Lỗi: {str(e)}')
            return render(request, 'attendance/face_register.html')
    
    return render(request, 'attendance/face_register.html')


# ==================== LỊCH SỬ CHẤM CÔNG ====================

@login_required
def attendance_history(request):
    """Lịch sử chấm công với bộ lọc"""
    if request.user.role != User.Role.ADMIN:
        return redirect('staff_dashboard')
    
    from django.core.paginator import Paginator
    from django.db.models import Q
    
    logs = AttendanceLog.objects.select_related('user', 'user__work_shift').order_by('-timestamp')
    
    # Bộ lọc theo nhân viên
    staff_id = request.GET.get('staff_id')
    if staff_id and staff_id.isdigit():
        logs = logs.filter(user_id=staff_id)
    else:
        staff_id = ''
    
    # Bộ lọc theo ngày
    date_from = request.GET.get('date_from')
    date_to = request.GET.get('date_to')
    if date_from:
        logs = logs.filter(timestamp__date__gte=date_from)
    if date_to:
        logs = logs.filter(timestamp__date__lte=date_to)
    
    # Bộ lọc theo trạng thái
    status = request.GET.get('status')
    if status:
        logs = logs.filter(status=status)
    
    # Phân trang
    paginator = Paginator(logs, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    # Danh sách nhân viên cho dropdown
    staff_members = User.objects.filter(role=User.Role.STAFF).order_by('last_name', 'first_name')
    
    context = {
        'page_title': 'Lịch sử chấm công',
        'page_obj': page_obj,
        'staff_members': staff_members,
        'selected_staff': staff_id,
        'date_from': date_from or '',
        'date_to': date_to or '',
        'selected_status': status or '',
    }
    return render(request, 'attendance/attendance_history.html', context)


# ==================== BÁO CÁO THÁNG ====================

@login_required
def monthly_report(request):
    """Báo cáo chấm công theo tháng"""
    if request.user.role != User.Role.ADMIN:
        return redirect('staff_dashboard')
    
    from datetime import datetime, date
    import calendar
    from django.db.models import Count, Q
    
    # Lấy tháng/năm từ request (mặc định: tháng hiện tại)
    now = timezone.localtime(timezone.now())
    month = int(request.GET.get('month', now.month))
    year = int(request.GET.get('year', now.year))
    
    # Số ngày làm việc trong tháng (trừ thứ 7, CN)
    _, days_in_month = calendar.monthrange(year, month)
    workdays = 0
    for day in range(1, days_in_month + 1):
        d = date(year, month, day)
        if d.weekday() < 5:  # Thứ 2-6
            workdays += 1
    
    # Thống kê bằng 1 query duy nhất (thay vì N queries cho N nhân viên)
    staff_members = User.objects.filter(
        role=User.Role.STAFF
    ).select_related('work_shift').annotate(
        on_time=Count('logs', filter=Q(
            logs__timestamp__year=year,
            logs__timestamp__month=month,
            logs__status=AttendanceLog.Status.ON_TIME
        )),
        late=Count('logs', filter=Q(
            logs__timestamp__year=year,
            logs__timestamp__month=month,
            logs__status=AttendanceLog.Status.LATE
        )),
        absent=Count('logs', filter=Q(
            logs__timestamp__year=year,
            logs__timestamp__month=month,
            logs__status=AttendanceLog.Status.ABSENT
        )),
        days_present=Count('logs__timestamp__date', filter=Q(
            logs__timestamp__year=year,
            logs__timestamp__month=month,
        ), distinct=True),
    ).order_by('last_name', 'first_name')
    
    report_data = []
    total_on_time = 0
    total_late = 0
    total_absent = 0
    chart_labels = []
    chart_rates = []
    
    for staff in staff_members:
        rate = round((staff.days_present / workdays) * 100, 1) if workdays > 0 else 0
        
        report_data.append({
            'staff': staff,
            'days_present': staff.days_present,
            'on_time': staff.on_time,
            'late': staff.late,
            'absent': staff.absent,
            'rate': rate,
        })
        
        total_on_time += staff.on_time
        total_late += staff.late
        total_absent += staff.absent
        
        name = staff.get_full_name() or staff.username
        chart_labels.append(name)
        chart_rates.append(rate)
    
    # Tạo danh sách năm cho dropdown (3 năm trước → năm hiện tại)
    year_choices = list(range(now.year - 2, now.year + 1))
    
    context = {
        'page_title': f'Báo cáo tháng {month}/{year}',
        'report_data': report_data,
        'month': month,
        'year': year,
        'workdays': workdays,
        'total_staff': staff_members.count(),
        'total_on_time': total_on_time,
        'total_late': total_late,
        'total_absent': total_absent,
        'year_choices': year_choices,
        'chart_labels': chart_labels,
        'chart_rates': chart_rates,
    }
    return render(request, 'attendance/monthly_report.html', context)
