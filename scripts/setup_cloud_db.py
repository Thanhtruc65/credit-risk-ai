"""
Script khởi tạo bảng Users và PredictionHistory trên Cloud Database (PostgreSQL).
Sử dụng cho Render, Supabase, Neon.tech,...
Chạy: python setup_cloud_db.py
"""
import sqlalchemy
import os
from dotenv import load_dotenv

load_dotenv()

# Lấy DATABASE_URL từ môi trường (ví dụ từ Supabase)
# Định dạng: postgresql://user:password@host:port/dbname
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("❌ LỖI: Không tìm thấy biến môi trường DATABASE_URL trong file .env")
    print("Vui lòng thêm DATABASE_URL=postgresql://... vào file .env rồi chạy lại.")
    exit(1)

# Xử lý trường hợp URL bắt đầu bằng postgres:// (Render/Railway thường dùng)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

print("=" * 60)
print("KHỞI TẠO BẢNG TRÊN CLOUD DATABASE (POSTGRESQL)")
print("=" * 60)

try:
    print("Đang kết nối tới database...")
    engine = sqlalchemy.create_engine(DATABASE_URL)

    with engine.connect() as conn:
        print("✅ Kết nối thành công!")

        # ── 1. Tạo bảng Users ──
        print("\n[1/2] Đang tạo bảng Users...")
        conn.execute(sqlalchemy.text("""
            CREATE TABLE IF NOT EXISTS Users (
                id SERIAL PRIMARY KEY,
                username VARCHAR(50) NOT NULL UNIQUE,
                email VARCHAR(100) NOT NULL UNIQUE,
                full_name VARCHAR(100) NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        conn.commit()
        print("   ✅ Bảng Users — OK!")

        # ── 2. Tạo bảng PredictionHistory ──
        print("[2/3] Đang tạo bảng PredictionHistory...")
        conn.execute(sqlalchemy.text("""
            CREATE TABLE IF NOT EXISTS PredictionHistory (
                id SERIAL PRIMARY KEY,
                user_id INT NOT NULL REFERENCES Users(id) ON DELETE CASCADE,
                sk_id_curr INT DEFAULT 0,
                prediction_code INT NOT NULL,
                risk_status VARCHAR(50),
                confidence FLOAT,
                dti FLOAT,
                pti FLOAT,
                form_data TEXT,
                result_data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        conn.commit()
        print("   ✅ Bảng PredictionHistory — OK!")

        # ── 3. Tạo bảng Clients (Để Dashboard không bị lỗi) ──
        print("[3/3] Đang tạo bảng Clients (Dữ liệu mẫu)...")
        conn.execute(sqlalchemy.text("""
            CREATE TABLE IF NOT EXISTS Clients (
                SK_ID_CURR INT PRIMARY KEY,
                TARGET INT,
                NAME_CONTRACT_TYPE VARCHAR(50),
                CODE_GENDER VARCHAR(5),
                FLAG_OWN_CAR VARCHAR(5),
                FLAG_OWN_REALTY VARCHAR(5),
                AMT_INCOME_TOTAL FLOAT,
                AMT_CREDIT FLOAT,
                AMT_ANNUITY FLOAT,
                NAME_EDUCATION_TYPE VARCHAR(100),
                NAME_FAMILY_STATUS VARCHAR(100),
                NAME_HOUSING_TYPE VARCHAR(100),
                DAYS_BIRTH INT,
                DAYS_EMPLOYED INT,
                OCCUPATION_TYPE VARCHAR(100),
                AGE INT
            );
        """))
        conn.commit()
        print("   ✅ Bảng Clients — OK!")

        print("\n🎉 HOÀN TẤT KHỞI TẠO DATABASE TRÊN CLOUD!")
        print("Bây giờ anh đã có thể đẩy code lên GitHub và deploy.")

except Exception as e:
    print(f"\n❌ Lỗi khi khởi tạo database: {e}")
