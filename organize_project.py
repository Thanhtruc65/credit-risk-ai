import os
import shutil

def organize():
    # Định nghĩa cấu trúc thư mục
    folders = ['scripts', 'tests', 'models', 'data', 'static', 'templates']
    
    # Tạo thư mục nếu chưa có
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"✅ Đã tạo thư mục: {folder}")

    # Danh sách các file cần di chuyển vào 'scripts/'
    script_files = [
        'preprocess.py', 
        'setup_cloud_db.py', 
        'setup_data.py', 
        'setup_sql.py', 
        'setup_users_sql.py', 
        'train.py', 
        'tune.py',
        'migrate_data.py'
    ]
    
    # Danh sách các file cần di chuyển vào 'tests/'
    test_files = [
        'test_predict.py'
    ]

    # Di chuyển file vào 'scripts/'
    for f in script_files:
        if os.path.exists(f):
            try:
                shutil.move(f, os.path.join('scripts', f))
                print(f"➡️  Đã dọn dẹp: {f} -> scripts/")
            except Exception as e:
                print(f"❌ Lỗi khi di chuyển {f}: {e}")

    # Di chuyển file vào 'tests/'
    for f in test_files:
        if os.path.exists(f):
            try:
                shutil.move(f, os.path.join('tests', f))
                print(f"➡️  Đã dọn dẹp: {f} -> tests/")
            except Exception as e:
                print(f"❌ Lỗi khi di chuyển {f}: {e}")

    print("\n🎉 HOÀN TẤT! Dự án của bạn bây giờ trông đã rất chuyên nghiệp rồi đó.")
    print("Bạn có thể xóa file organize_project.py này sau khi chạy xong.")

if __name__ == "__main__":
    organize()
