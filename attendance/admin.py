from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import User, WorkShift, AttendanceLog


@admin.register(WorkShift)
class WorkShiftAdmin(admin.ModelAdmin):
    list_display = ['name', 'start_time', 'end_time', 'late_grace_period']
    search_fields = ['name']


@admin.register(User)
class UserAdmin(BaseUserAdmin):
    list_display = ['username', 'email', 'first_name', 'last_name', 'role', 'work_shift', 'is_active']
    list_filter = ['role', 'gender', 'work_shift', 'is_active']
    
    # Gộp thông tin cá nhân và bổ sung vào một khung
    fieldsets = (
        (None, {'fields': ('username', 'password')}),
        ('Thông tin cá nhân', {
            'fields': ('first_name', 'last_name', 'email', 'gender', 'dob', 'phone', 'avatar')
        }),
        ('Thông tin làm việc', {
            'fields': ('role', 'work_shift', 'is_active')
        }),
        ('Ngày tháng quan trọng', {
            'fields': ('last_login', 'date_joined'),
            'classes': ('collapse',)
        }),
    )
    
    add_fieldsets = (
        (None, {
            'classes': ('wide',),
            'fields': ('username', 'password1', 'password2'),
        }),
        ('Thông tin cá nhân', {
            'fields': ('first_name', 'last_name', 'email', 'gender', 'dob', 'phone')
        }),
        ('Thông tin làm việc', {
            'fields': ('role', 'work_shift')
        }),
    )
    
    # Ẩn trường face_encoding_text, groups, user_permissions
    exclude = ['face_encoding_text']
    
    def save_model(self, request, obj, form, change):
        """Tự động set quyền dựa vào role"""
        if obj.role == User.Role.ADMIN:
            obj.is_staff = True
            obj.is_superuser = True
        else:  # STAFF
            obj.is_staff = False
            obj.is_superuser = False
        
        super().save_model(request, obj, form, change)


@admin.register(AttendanceLog)
class AttendanceLogAdmin(admin.ModelAdmin):
    list_display = ['user', 'timestamp', 'status', 'snapshot']
    list_filter = ['status', 'timestamp']
    search_fields = ['user__username', 'user__first_name', 'user__last_name']
    date_hierarchy = 'timestamp'
    readonly_fields = ['timestamp']
