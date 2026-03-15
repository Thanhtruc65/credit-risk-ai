import pandas as pd
import sqlalchemy
import os
import sys
from dotenv import load_dotenv

# Thêm đường dẫn gốc vào sys.path để load_dotenv tìm thấy file .env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

if not DATABASE_URL:
    print("❌ LỖI: Không tìm thấy DATABASE_URL")
    exit(1)

# File dữ liệu nguồn
FILE_PATH = "data/train_processed.csv"

def migrate():
    try:
        if not os.path.exists(FILE_PATH):
            print(f"❌ LỖI: Không tìm thấy file {FILE_PATH}. Vui lòng chạy python scripts/preprocess.py trước.")
            return

        print(f"Reading {FILE_PATH}...")
        df = pd.read_csv(FILE_PATH)
        
        # Chỉ lấy 1000 dòng để test
        df_sample = df.head(1000).copy()
        
        # Đảm bảo các cột cần thiết khớp với Database
        mapping = {
            'SK_ID_CURR': 'SK_ID_CURR',
            'TARGET': 'TARGET',
            'NAME_CONTRACT_TYPE': 'NAME_CONTRACT_TYPE',
            'CODE_GENDER': 'CODE_GENDER',
            'FLAG_OWN_CAR': 'FLAG_OWN_CAR',
            'FLAG_OWN_REALTY': 'FLAG_OWN_REALTY',
            'AMT_INCOME_TOTAL': 'AMT_INCOME_TOTAL',
            'AMT_CREDIT': 'AMT_CREDIT',
            'AMT_ANNUITY': 'AMT_ANNUITY',
            'NAME_EDUCATION_TYPE': 'NAME_EDUCATION_TYPE',
            'NAME_FAMILY_STATUS': 'NAME_FAMILY_STATUS',
            'NAME_HOUSING_TYPE': 'NAME_HOUSING_TYPE',
            'DAYS_BIRTH': 'DAYS_BIRTH',
            'DAYS_EMPLOYED': 'DAYS_EMPLOYED',
            'OCCUPATION_TYPE': 'OCCUPATION_TYPE'
        }
        
        cols_to_keep = [c for c in mapping.keys() if c in df.columns]
        df_final = df_sample[cols_to_keep].copy()
        
        if 'AGE' not in df_final.columns and 'DAYS_BIRTH' in df_final.columns:
            df_final['AGE'] = (df_final['DAYS_BIRTH'].abs() / 365).astype(int)
        elif 'AGE' in df.columns:
            df_final['AGE'] = df['AGE'].head(1000)

        print(f"Connecting to Cloud Database...")
        engine = sqlalchemy.create_engine(DATABASE_URL)
        
        # Xóa dữ liệu cũ nếu có trước khi đẩy mới (Tùy chọn)
        # with engine.connect() as conn:
        #     conn.execute(sqlalchemy.text("TRUNCATE TABLE clients RESTART IDENTITY CASCADE;"))
        #     conn.commit()

        print(f"Pushing {len(df_final)} rows to table 'clients'...")
        df_final.to_sql('clients', engine, if_exists='replace', index=False)
        
        print("✅ Thành công! Dữ liệu đã được cập nhật lại lên Cloud.")

    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    migrate()
