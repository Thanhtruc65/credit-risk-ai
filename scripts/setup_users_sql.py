"""
Script tạo bảng Users và PredictionHistory trên SQL Server.
Chạy: python setup_users_sql.py
"""
import sqlalchemy
import urllib

print("=" * 50)
print("TẠO BẢNG USERS & PREDICTION HISTORY TRÊN SQL SERVER")
print("=" * 50)

# Cấu hình kết nối SQL Server
server = r'LAPTOP-STR3U83B\SQLEXPRESS01'
database = 'DoAn'
driver = 'ODBC Driver 17 for SQL Server'

try:
    print(f"Đang kết nối tới SQL Server: {server} | DB: {database}...")

    params = urllib.parse.quote_plus(
        f'Driver={{{driver}}};'
        f'Server={server};'
        f'Database={database};'
        f'Trusted_Connection=yes;'
    )
    engine = sqlalchemy.create_engine(f"mssql+pyodbc:///?odbc_connect={params}")

    with engine.connect() as conn:
        print("✅ Đã kết nối thành công tới SQL Server!")

        # ── 1. Tạo bảng Users ──
        print("\n[1/2] Đang tạo bảng Users...")
        conn.execute(sqlalchemy.text("""
            IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'Users')
            BEGIN
                CREATE TABLE Users (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    username NVARCHAR(50) NOT NULL UNIQUE,
                    email NVARCHAR(100) NOT NULL UNIQUE,
                    full_name NVARCHAR(100) NOT NULL,
                    password_hash NVARCHAR(255) NOT NULL,
                    created_at DATETIME DEFAULT GETDATE()
                );
                PRINT 'Bảng Users đã được tạo.';
            END
            ELSE
                PRINT 'Bảng Users đã tồn tại, bỏ qua.';
        """))
        conn.commit()
        print("   ✅ Bảng Users — OK!")

        # ── 2. Tạo bảng PredictionHistory ──
        print("[2/2] Đang tạo bảng PredictionHistory...")
        conn.execute(sqlalchemy.text("""
            IF NOT EXISTS (SELECT * FROM sys.tables WHERE name = 'PredictionHistory')
            BEGIN
                CREATE TABLE PredictionHistory (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    user_id INT NOT NULL,
                    sk_id_curr INT DEFAULT 0,
                    prediction_code INT NOT NULL,
                    risk_status NVARCHAR(50),
                    confidence FLOAT,
                    dti FLOAT,
                    pti FLOAT,
                    form_data NVARCHAR(MAX),
                    result_data NVARCHAR(MAX),
                    created_at DATETIME DEFAULT GETDATE(),
                    CONSTRAINT FK_PredictionHistory_Users
                        FOREIGN KEY (user_id) REFERENCES Users(id)
                        ON DELETE CASCADE
                );
                PRINT 'Bảng PredictionHistory đã được tạo.';
            END
            ELSE
                PRINT 'Bảng PredictionHistory đã tồn tại, bỏ qua.';
        """))
        conn.commit()
        print("   ✅ Bảng PredictionHistory — OK!")

        # ── Verify ──
        result = conn.execute(sqlalchemy.text(
            "SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_NAME IN ('Users', 'PredictionHistory')"
        ))
        tables = [row[0] for row in result]
        print(f"\n✅ HOÀN TẤT! Các bảng hiện có: {tables}")
        print("\nBạn có thể bắt đầu đăng ký/đăng nhập trên giao diện web.")

except sqlalchemy.exc.OperationalError as e:
    print("\n❌ LỖI KẾT NỐI: Không thể truy cập SQL Server.")
    print("Vui lòng đảm bảo:")
    print("1. SQL Server đang chạy")
    print("2. Tên Server và Database chính xác")
    print("Chi tiết lỗi:", str(e))
except Exception as e:
    print(f"\n❌ Lỗi: {e}")
