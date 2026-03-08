import pandas as pd
import sqlalchemy
import urllib
import os

print("="*50)
print("HƯỚNG DẪN IMPORT DỮ LIỆU TỪ CSV VÀO SQL SERVER")
print("="*50)

# Cấu hình kết nối SQL Server (dựa trên ảnh của bạn)
server = r'LAPTOP-STR3U83B\SQLEXPRESS01'
database = 'DoAn'
driver = 'ODBC Driver 17 for SQL Server' # Bạn có thể cần cài đặt ODBC Driver trên áy

# Đường dẫn tới thư mục data
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
APP_TRAIN_PATH = os.path.join(DATA_DIR, "application_train.csv")
BUREAU_PATH = os.path.join(DATA_DIR, "bureau.csv")

try:
    print(f"Đang kết nối tới SQL Server: {server} | DB: {database}...")
    
    # Tạo connection string cho SQLAlchemy
    params = urllib.parse.quote_plus(
        f'Driver={{{driver}}};'
        f'Server={server};'
        f'Database={database};'
        f'Trusted_Connection=yes;'
    )
    engine = sqlalchemy.create_engine(f"mssql+pyodbc:///?odbc_connect={params}")
    
    # Kể nối thử
    with engine.connect() as conn:
        print("✅ Đã kết nối thành công tới SQL Server!")
        
        # 1. ĐỌC DỮ LIỆU
        print("\nĐang đọc dữ liệu từ CSV...")
        if not os.path.exists(APP_TRAIN_PATH):
            print(f"❌ Không tìm thấy file: {APP_TRAIN_PATH}")
            print("Vui lòng tải application_train.csv bỏ vào thư mục data/")
            exit(1)
            
        app_df = pd.read_csv(APP_TRAIN_PATH)
        print(f"-> Đã đọc {len(app_df)} dòng từ bảng chính.")
        
        # 2. GHÉP BẢNG LỊCH SỬ TÍN DỤNG (BUREAU)
        if os.path.exists(BUREAU_PATH):
            print("-> Đã tìm thấy bureau.csv. Đang tổng hợp lịch sử tín dụng...")
            bureau_df = pd.read_csv(BUREAU_PATH)
            
            # Tính số khoản vay đang hoạt động (Active)
            active_loans = bureau_df[bureau_df['CREDIT_ACTIVE'] == 'Active'].groupby('SK_ID_CURR').size().reset_index(name='ACTIVE_LOANS_COUNT')
            
            # Tổng nợ hiện tại
            debt_sum = bureau_df.groupby('SK_ID_CURR')['AMT_CREDIT_SUM_DEBT'].sum().reset_index(name='CURRENT_DEBT_TOTAL')
            
            # Số ngày trễ hạn cao nhất (DPD)
            dpd_max = bureau_df.groupby('SK_ID_CURR')['CREDIT_DAY_OVERDUE'].max().reset_index(name='MAX_DPD')
            
            # Ghép vào bảng chính
            app_df = app_df.merge(active_loans, on='SK_ID_CURR', how='left')
            app_df = app_df.merge(debt_sum, on='SK_ID_CURR', how='left')
            app_df = app_df.merge(dpd_max, on='SK_ID_CURR', how='left')
            
            # Điền 0 cho những người không có hồ sơ nợ
            app_df['ACTIVE_LOANS_COUNT'] = app_df['ACTIVE_LOANS_COUNT'].fillna(0)
            app_df['CURRENT_DEBT_TOTAL'] = app_df['CURRENT_DEBT_TOTAL'].fillna(0)
            app_df['MAX_DPD'] = app_df['MAX_DPD'].fillna(0)
            print("-> Đã ghép xong dữ liệu lịch sử vay vốn/nợ hiện tại!")
        else:
            print("-> Không tìm thấy bureau.csv. Bỏ qua ghép dữ liệu nợ ngoài.")
            
        # 3. TIỀN XỬ LÝ DỮ LIỆU CƠ BẢN TRƯỚC KHI ĐẨY LÊN SQL
        print("-> Đang chuẩn hóa dữ liệu (Tuổi, Ngày làm việc)...")
        # Chuyển DAYS_BIRTH thành AGE
        app_df['AGE'] = app_df['DAYS_BIRTH'].apply(lambda x: abs(x) // 365)
        # Chuẩn hóa DAYS_EMPLOYED (lấy giá trị tuyệt đối, xử lý giá trị lỗi 365243)
        app_df['DAYS_EMPLOYED'] = app_df['DAYS_EMPLOYED'].replace(365243, 0).apply(lambda x: abs(x))
        # Xử lý CNT_FAM_MEMBERS và CNT_CHILDREN
        app_df['CNT_CHILDREN'] = app_df['CNT_CHILDREN'].fillna(0)

        # 4. CHỈ LẤY CÁC CỘT QUAN TRỌNG
        cols_to_keep = [
            'SK_ID_CURR', 'TARGET', 'NAME_CONTRACT_TYPE', 'CODE_GENDER', 
            'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 
            'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 
            'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 
            'AGE', 'DAYS_EMPLOYED', 'OCCUPATION_TYPE'
        ]
        
        if 'ACTIVE_LOANS_COUNT' in app_df.columns:
            cols_to_keep.extend(['ACTIVE_LOANS_COUNT', 'CURRENT_DEBT_TOTAL', 'MAX_DPD'])
            
        # Chỉ giữ lại những cột có trong df
        cols_to_keep = [c for c in cols_to_keep if c in app_df.columns]
        final_df = app_df[cols_to_keep]
        
        # 4. ĐẨY LÊN SQL SERVER (Bảng Clients)
        table_name = "Clients"
        print(f"\nĐang đẩy {len(final_df)} dòng dữ liệu vào bảng [{table_name}] trên SQL Server...")
        print("⚠️ Quá trình này có thể mất vài phút tùy tốc độ máy...")
        
        # Dùng chunksize để tránh quá tải RAM
        final_df.to_sql(name=table_name, con=engine, if_exists='replace', index=False, chunksize=1000)
        
        print(f"\n✅ HOÀN TẤT! Đã tạo bảng [{table_name}] và lưu toàn bộ data thành công.")
        print("\nBây giờ bạn có thể sửa code trong app.py để đọc từ SQL Server:")
        print("```python")
        print("import pyodbc")
        print("conn = pyodbc.connect('Driver={ODBC Driver 17 for SQL Server};Server=LAPTOP-STR3U83B\\SQLEXPRESS01;Database=DoAn;Trusted_Connection=yes;')")
        print("client_df = pd.read_sql(f'SELECT * FROM Clients WHERE SK_ID_CURR = {sk_id}', conn)")
        print("```")

except sqlalchemy.exc.OperationalError as e:
    print("\n❌ LỖI KẾT NỐI: Không thể truy cập SQL Server.")
    print("Vui lòng đảm bảo:")
    print("1. Đã cài đặt SQLAlchemy và pyodbc: `pip install sqlalchemy pyodbc`")
    print("2. Đã tải ODBC Driver 17 for SQL Server từ trang chủ Microsoft.")
    print("3. Tên Server và Database chính xác.")
    print("Chi tiết lỗi:", str(e))
except Exception as e:
    print(f"\n❌ Lỗi: {e}")
