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
                    import face_recognition
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
                    
                    # Lưu face encodings từ nhiều ảnh
                    all_encodings = []
                    uploaded_urls = []
                    
                    for idx, image in enumerate(face_images, start=1):
                        # Lấy extension từ tên file gốc
                        ext = os.path.splitext(image.name)[1] or '.jpg'
                        # Tạo tên file: username (1).jpg, username (2).jpg
                        file_name = f"{username} ({idx}){ext}"
                        file_path = f"avatars/{file_name}"
                        
                        # Upload lên Supabase Storage
                        image.seek(0)
                        file_content = image.read()
                        
                        response = supabase.storage.from_(settings.SUPABASE_BUCKET_NAME).upload(
                            path=file_path,
                            file=file_content,
                            file_options={
                                "content-type": image.content_type,
                                "upsert": "true"
                            }
                        )
                        
                        # Lấy public URL
                        public_url = supabase.storage.from_(settings.SUPABASE_BUCKET_NAME).get_public_url(file_path)
                        uploaded_urls.append(public_url)
                        
                        # Xử lý face encoding
                        image.seek(0)
                        img_array = np.frombuffer(image.read(), np.uint8)
                        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                        if img is not None:
                            # Resize ảnh lớn hơn để giữ chi tiết
                            max_dimension = 1000
                            height, width = img.shape[:2]
                            if max(height, width) > max_dimension:
                                scale = max_dimension / max(height, width)
                                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                            
                            # Histogram equalization để chuẩn hóa ánh sáng
                            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                            lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
                            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                            
                            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            
                            # Phát hiện khuôn mặt với HOG (nhanh hơn CNN)
                            face_locations = face_recognition.face_locations(rgb_img, number_of_times_to_upsample=1, model="hog")
                            
                            # Tìm và mã hóa khuôn mặt với num_jitters=5 cho độ chính xác cao
                            if face_locations:
                                face_encodings = face_recognition.face_encodings(rgb_img, face_locations, num_jitters=5)
                                if face_encodings:
                                    all_encodings.append(face_encodings[0].tolist())
                    
                    # Lưu path ảnh đầu tiên vào avatar field
                    user.avatar = f"avatars/{username} (1){os.path.splitext(face_images[0].name)[1] or '.jpg'}"
                    
                    # Lưu face encodings (trung bình của tất cả ảnh)
                    if all_encodings:
                        avg_encoding = np.mean(all_encodings, axis=0).tolist()
                        user.set_encoding(np.array(avg_encoding))
                        messages.success(request, f'✅ Đã upload {len(face_images)} ảnh và tạo face encoding!')
                    else:
                        messages.warning(request, f'⚠️ Không phát hiện khuôn mặt trong ảnh. Vui lòng upload lại!')
                    
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
                import face_recognition
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
                uploaded_urls = []
                all_encodings = []
                
                for idx, image in enumerate(face_images, start=1):
                    ext = os.path.splitext(image.name)[1] or '.jpg'
                    file_name = f"{username} ({idx}){ext}"
                    file_path = f"avatars/{file_name}"
                    
                    # Upload lên Supabase
                    image.seek(0)
                    file_content = image.read()
                    
                    response = supabase.storage.from_(settings.SUPABASE_BUCKET_NAME).upload(
                        path=file_path,
                        file=file_content,
                        file_options={
                            "content-type": image.content_type,
                            "upsert": "true"
                        }
                    )
                    
                    public_url = supabase.storage.from_(settings.SUPABASE_BUCKET_NAME).get_public_url(file_path)
                    uploaded_urls.append(public_url)
                    
                    # Decode face encoding
                    image.seek(0)
                    img_array = np.frombuffer(image.read(), np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    if img is not None:
                        # Resize ảnh lớn hơn để giữ chi tiết
                        max_dimension = 1000
                        height, width = img.shape[:2]
                        if max(height, width) > max_dimension:
                            scale = max_dimension / max(height, width)
                            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                        
                        # Histogram equalization để chuẩn hóa ánh sáng
                        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                        lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
                        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                        
                        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Phát hiện khuôn mặt với HOG
                        face_locations = face_recognition.face_locations(rgb_img, number_of_times_to_upsample=1, model="hog")
                        
                        # Mã hóa khuôn mặt với num_jitters=5 cho độ chính xác cao
                        if face_locations:
                            face_encodings = face_recognition.face_encodings(rgb_img, face_locations, num_jitters=5)
                            if face_encodings:
                                all_encodings.append(face_encodings[0].tolist())
                
                # Cập nhật avatar path
                staff.avatar = f"avatars/{username} (1){os.path.splitext(face_images[0].name)[1] or '.jpg'}"
                
                # Lưu face encoding (trung bình nếu có nhiều ảnh)
                if all_encodings:
                    avg_encoding = np.mean(all_encodings, axis=0).tolist()
                    staff.set_encoding(np.array(avg_encoding))
                    messages.success(request, f'✅ Đã upload {len(face_images)} ảnh và cập nhật face encoding!')
                else:
                    messages.warning(request, f'⚠️ Đã upload {len(face_images)} ảnh nhưng không phát hiện khuôn mặt!')
                    
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
            import face_recognition
            import cv2
            import numpy as np
            from datetime import datetime
            
            # Đọc ảnh upload
            img_array = np.frombuffer(uploaded_image.read(), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                messages.error(request, '❌ Không thể đọc ảnh! Vui lòng thử lại.')
                return render(request, 'attendance/kiosk.html')
            
            # Histogram equalization để chuẩn hóa ánh sáng (giống lúc đăng ký)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Tìm khuôn mặt trong ảnh - thử nhiều phương pháp
            face_locations = []
            
            # Phương pháp 1: HOG với upsampling (nhanh)
            face_locations = face_recognition.face_locations(rgb_img, number_of_times_to_upsample=2, model="hog")
            
            # Phương pháp 2: Nếu không tìm thấy, thử CNN (chính xác hơn)
            if not face_locations:
                try:
                    face_locations = face_recognition.face_locations(rgb_img, model="cnn")
                except Exception:
                    pass  # CNN có thể không khả dụng
            
            # Phương pháp 3: Resize ảnh lớn hơn và thử lại
            if not face_locations:
                scale = 2
                large_img = cv2.resize(rgb_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                face_locations = face_recognition.face_locations(large_img, model="hog")
                # Điều chỉnh lại tọa độ
                face_locations = [(int(top/scale), int(right/scale), int(bottom/scale), int(left/scale)) 
                                  for top, right, bottom, left in face_locations]
            
            if not face_locations:
                messages.error(request, '❌ Không phát hiện khuôn mặt! Vui lòng đảm bảo khuôn mặt rõ ràng, đủ sáng và chụp lại.')
                return render(request, 'attendance/kiosk.html')
            
            # Mã hóa khuôn mặt với num_jitters=5 để tăng độ chính xác
            # num_jitters: re-sample và lấy trung bình, cao hơn = chính xác hơn nhưng chậm hơn
            face_encodings = face_recognition.face_encodings(rgb_img, face_locations, num_jitters=5)
            
            if not face_encodings:
                messages.error(request, '❌ Không thể mã hóa khuôn mặt! Vui lòng thử lại.')
                return render(request, 'attendance/kiosk.html')
            
            unknown_encoding = face_encodings[0]
            
            # So sánh với tất cả nhân viên trong database
            staff_members = User.objects.filter(role=User.Role.STAFF).exclude(face_encoding_text__isnull=True)
            
            best_match = None
            best_distance = 0.50  # Ngưỡng cho phép > 60% confidence
            # Khoảng cách càng nhỏ = match càng chính xác
            # 0.40 = ~75%, 0.45 = ~70%, 0.50 = ~60%
            
            for staff in staff_members:
                known_encoding = staff.get_encoding()
                if known_encoding is not None:
                    # Tính khoảng cách
                    distance = face_recognition.face_distance([known_encoding], unknown_encoding)[0]
                    
                    if distance < best_distance:
                        best_distance = distance
                        best_match = staff
            
            # Kiểm tra độ chính xác tối thiểu
            # Nếu best_match is None, nghĩa là không có ai match đủ tốt (distance > 0.38)
            if best_match is None:
                messages.error(request, '❌ Không nhận diện được! Khuôn mặt không khớp với bất kỳ nhân viên nào trong hệ thống.')
                return render(request, 'attendance/kiosk.html')
            
            # Tìm thấy nhân viên - Tạo log chấm công
            today = timezone.now().date()
            
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
            
            # Tính độ chính xác đơn giản và trực quan
            # Dùng công thức gốc: (1 - distance) * 100
            # Với threshold 0.50, confidence tối thiểu là 50%
            confidence = round((1 - best_distance) * 100, 1)
            
            context = {
                'success': True,
                'user': best_match,
                'confidence': confidence
            }
            return render(request, 'attendance/kiosk.html', context)
                
        except Exception as e:
            messages.error(request, f'❌ Lỗi: {str(e)}')
            return render(request, 'attendance/kiosk.html')
    
    return render(request, 'attendance/kiosk.html')


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
            import face_recognition
            import cv2
            import numpy as np
            
            # Đọc ảnh upload
            img_array = np.frombuffer(uploaded_image.read(), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                messages.error(request, '❌ Không thể đọc ảnh!')
                return render(request, 'attendance/face_register.html')
            
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Tìm khuôn mặt - thử nhiều phương pháp
            face_locations = face_recognition.face_locations(rgb_img, number_of_times_to_upsample=2, model="hog")
            
            if not face_locations:
                try:
                    face_locations = face_recognition.face_locations(rgb_img, model="cnn")
                except Exception:
                    pass
            
            if not face_locations:
                messages.error(request, '❌ Không phát hiện khuôn mặt! Vui lòng đảm bảo mặt rõ ràng.')
                return render(request, 'attendance/face_register.html')
            
            # Mã hóa khuôn mặt
            face_encodings = face_recognition.face_encodings(rgb_img, face_locations, num_jitters=2)
            
            if not face_encodings:
                messages.error(request, '❌ Không thể mã hóa khuôn mặt!')
                return render(request, 'attendance/face_register.html')
            
            # Lưu encoding vào user hiện tại
            request.user.set_encoding(face_encodings[0])
            request.user.save()
            
            messages.success(request, f'✅ Đã đăng ký khuôn mặt thành công cho {request.user.get_full_name() or request.user.username}!')
            return render(request, 'attendance/face_register.html', {'success': True})
            
        except Exception as e:
            messages.error(request, f'❌ Lỗi: {str(e)}')
            return render(request, 'attendance/face_register.html')
    
    return render(request, 'attendance/face_register.html')
