# Hướng Dẫn Cấu Hình Supabase Storage

## Bước 1: Tạo Bucket trên Supabase

1. Truy cập: https://supabase.com/dashboard
2. Chọn project: **wyucizqcpzaglxtmqlmp**
3. Vào **Storage** từ menu bên trái
4. Tạo bucket mới tên: **Timekeeping** (hoặc kiểm tra đã tồn tại)
5. Đặt **Public bucket** = TRUE để có thể truy cập ảnh công khai

## Bước 2: Cấu Hình Storage Policies

Vào **Storage** > **Policies** > Chọn bucket **Timekeeping**, sau đó click **New Policy**.

### Policy 1: Upload Files (INSERT)
Click **"For full customization"** và điền:

- **Policy name:** `Allow authenticated uploads`
- **Allowed operation:** Chọn **INSERT**
- **Target roles:** Chọn **authenticated**
- **WITH CHECK expression:** `bucket_id = 'Timekeeping'`

Hoặc đơn giản hơn: Click **"Allow all"** cho INSERT operation.

---

### Policy 2: Read Files (SELECT)  
Click **New Policy** > **"For full customization"**:

- **Policy name:** `Public read access`
- **Allowed operation:** Chọn **SELECT**
- **Target roles:** Chọn **public**
- **USING expression:** `bucket_id = 'Timekeeping'`

Hoặc: Click **"Allow all"** cho SELECT operation.

---

### Policy 3: Update Files (UPDATE)
Click **New Policy** > **"For full customization"**:

- **Policy name:** `Allow authenticated updates`
- **Allowed operation:** Chọn **UPDATE**
- **Target roles:** Chọ **authenticated**
- **USING expression:** `bucket_id = 'Timekeeping'`

---

### Policy 4: Delete Files (DELETE)
Click **New Policy** > **"For full customization"**:

- **Policy name:** `Allow authenticated deletes`
- **Allowed operation:** Chọn **DELETE**
- **Target roles:** Chọn **authenticated**
- **USING expression:** `bucket_id = 'Timekeeping'`

---

### ⚡ Cách Nhanh Nhất:
1. Vào **Storage** > **Timekeeping** bucket > **Policies**
2. Click **"Add policy"** > Chọn template **"Allow access to a bucket for authenticated users"**
3. Chọn tất cả operations: **INSERT, SELECT, UPDATE, DELETE**
4. Click **Review** > **Save policy**

## Bước 3: Kiểm Tra Kết Nối

Sau khi cấu hình xong, thử tạo nhân viên mới với 5-10 ảnh khuôn mặt.

### Cấu trúc file trên Supabase:
```
Timekeeping/
  └── avatars/
      ├── username (1).jpg
      ├── username (2).jpg
      ├── username (3).jpg
      └── ...
```

### URL ảnh sẽ có dạng:
```
https://wyucizqcpzaglxtmqlmp.supabase.co/storage/v1/object/public/Timekeeping/avatars/username (1).jpg
```

## Xử Lý Lỗi Thường Gặp

### Lỗi: "new row violates row-level security policy"
→ Kiểm tra lại Storage Policies, đảm bảo đã cho phép INSERT

### Lỗi: "Bucket not found"
→ Kiểm tra tên bucket trong .env phải khớp với tên trên Supabase (Timekeeping)

### Lỗi: 404 Not Found
→ Bucket chưa được tạo hoặc chưa public, đặt Public = TRUE

## Test Upload Thủ Công

Có thể test upload file trực tiếp trên Supabase Dashboard:
1. Vào **Storage** > **Timekeeping**
2. Tạo folder **avatars**
3. Upload 1 file test
4. Copy URL và test trên browser
