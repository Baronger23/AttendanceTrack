from django.urls import path
from . import views

urlpatterns = [
    # Authentication
    path('', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('open-kiosk/', views.open_kiosk, name='open_kiosk'),
    
    # Dashboard
    path('dashboard/', views.dashboard, name='dashboard'),
    path('admin/dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('staff/dashboard/', views.staff_dashboard, name='staff_dashboard'),
    
    # Quản lý nhân viên
    path('admin/staff/', views.staff_list, name='staff_list'),
    path('admin/staff/create/', views.staff_create, name='staff_create'),
    path('admin/staff/<int:pk>/edit/', views.staff_edit, name='staff_edit'),
    path('admin/staff/<int:pk>/delete/', views.staff_delete, name='staff_delete'),
    
    # Quản lý ca làm việc
    path('admin/shifts/', views.shift_list, name='shift_list'),
    path('admin/shifts/create/', views.shift_create, name='shift_create'),
    path('admin/shifts/<int:pk>/edit/', views.shift_edit, name='shift_edit'),
    path('admin/shifts/<int:pk>/delete/', views.shift_delete, name='shift_delete'),
    
    # Kiosk chấm công (Public - Không cần login)
    path('kiosk/', views.kiosk_checkin, name='kiosk_checkin'),
    path('kiosk/', views.kiosk_checkin, name='kiosk'),
    path('report-error/', views.report_error, name='report_error'),
    
    # Thông báo
    path('admin/notifications/', views.notification_list, name='notification_list'),
    path('admin/notifications/<int:pk>/', views.notification_detail, name='notification_detail'),
    
    # Đăng ký khuôn mặt (Cần login)
    path('face-register/', views.face_register, name='face_register'),
    
    # Báo cáo
    path('admin/attendance-history/', views.attendance_history, name='attendance_history'),
    path('admin/monthly-report/', views.monthly_report, name='monthly_report'),
]

