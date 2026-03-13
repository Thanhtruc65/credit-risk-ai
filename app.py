from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os
import json
import bcrypt
import uuid
import random
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
import google.generativeai as genai
from dotenv import load_dotenv
import sqlalchemy
from sqlalchemy import text
import urllib

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "loan_model.pkl")
ENCODERS_PATH = os.path.join(BASE_DIR, "models", "label_encoders.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "models", "feature_columns.pkl")
TRAIN_DATA_PATH = os.path.join(BASE_DIR, "data", "train_processed.csv")

HIGH_RISK_THRESHOLD = float(os.getenv("HIGH_RISK_THRESHOLD", "0.5"))
model = None
encoders = None
feature_columns = None
train_df_sample = None

# --- DATABASE CONFIG ---
DATABASE_URL = os.getenv("DATABASE_URL")
DB_CONN_STR = os.getenv("DB_CONN_STR", 'Driver={ODBC Driver 17 for SQL Server};Server=LAPTOP-STR3U83B\\SQLEXPRESS01;Database=DoAn;Trusted_Connection=yes;')

def get_engine():
    """Tạo engine SQLAlchemy phù hợp với môi trường (Cloud/Local)."""
    if DATABASE_URL:
        url = DATABASE_URL
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return sqlalchemy.create_engine(url, pool_pre_ping=True)
    else:
        # Local SQL Server
        quoted = urllib.parse.quote_plus(DB_CONN_STR)
        return sqlalchemy.create_engine(f"mssql+pyodbc:///?odbc_connect={quoted}")

engine = get_engine()

# --- AUTH CONFIG ---
SESSION_SECRET = os.getenv("SESSION_SECRET", "ai-loan-predictor-secret-key-2026")
active_sessions = {}  # { session_id: user_id }
otp_store = {}  # { email: { "otp": "123456", "expires": datetime } }

# --- SMTP CONFIG ---
SMTP_EMAIL = os.getenv("SMTP_EMAIL", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, encoders, feature_columns, train_df_sample
    print("Loading AI models and encoders...")
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print(f"  Model loaded from {MODEL_PATH}")
    else:
        print(f"  WARNING: Model not found at {MODEL_PATH}")
    if os.path.exists(ENCODERS_PATH):
        encoders = joblib.load(ENCODERS_PATH)
        print(f"  Encoders loaded")
    if os.path.exists(FEATURES_PATH):
        feature_columns = joblib.load(FEATURES_PATH)
        print(f"  Feature columns loaded ({len(feature_columns)} features)")
    if os.path.exists(TRAIN_DATA_PATH):
        train_df_sample = pd.read_csv(TRAIN_DATA_PATH, nrows=1000)
        print(f"  Sample data loaded ({len(train_df_sample)} rows)")
    print("Startup complete!")
    yield

app = FastAPI(title="AI Loan Predictor - Dự Đoán Vay Vốn", lifespan=lifespan)

static_dir = os.path.join(BASE_DIR, "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

templates_dir = os.path.join(BASE_DIR, "templates")
if not os.path.exists(templates_dir):
    os.makedirs(templates_dir)
templates = Jinja2Templates(directory=templates_dir)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class LoanRequest(BaseModel):
    SK_ID_CURR: int = 0
    NAME_CONTRACT_TYPE: str
    CODE_GENDER: str
    FLAG_OWN_CAR: str
    FLAG_OWN_REALTY: str
    CNT_CHILDREN: int
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    NAME_EDUCATION_TYPE: str
    NAME_FAMILY_STATUS: str
    NAME_HOUSING_TYPE: str
    AGE: int
    DAYS_EMPLOYED: float
    OCCUPATION_TYPE: str = ""
    ACTIVE_LOANS_COUNT: int = 0
    CURRENT_DEBT_TOTAL: float = 0
    MAX_DPD: int = 0

class ChatRequest(BaseModel):
    message: str
    context: dict

class RegisterRequest(BaseModel):
    username: str
    email: str
    full_name: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class SendOtpRequest(BaseModel):
    email: str

class CheckOtpRequest(BaseModel):
    email: str
    otp: str

class VerifyOtpRequest(BaseModel):
    email: str
    otp: str
    new_password: str

def get_current_user(request: Request):
    """Lấy user_id từ session cookie. Trả về None nếu chưa login."""
    session_id = request.cookies.get("session_id")
    if not session_id or session_id not in active_sessions:
        return None
    return active_sessions[session_id]

def get_db_conn():
    """Trả về connection từ SQLAlchemy engine."""
    return engine.connect()

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "active_page": "home"})

@app.get("/dashboard", response_class=HTMLResponse)
def read_dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request, "active_page": "dashboard"})

@app.get("/predict-loan", response_class=HTMLResponse)
def read_predict(request: Request):
    return templates.TemplateResponse("predict.html", {"request": request, "active_page": "predict"})

@app.get("/about", response_class=HTMLResponse)
def read_about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request, "active_page": "about"})

@app.get("/history", response_class=HTMLResponse)
def read_history(request: Request):
    return templates.TemplateResponse("history.html", {"request": request, "active_page": "history"})

# ══════════════════════════════════════════
#   AUTH APIs
# ══════════════════════════════════════════

@app.post("/api/register")
def register_user(req: RegisterRequest):
    try:
        with get_db_conn() as conn:
            # Kiểm tra username đã tồn tại
            res = conn.execute(text("SELECT id FROM Users WHERE username = :u"), {"u": req.username})
            if res.fetchone():
                raise HTTPException(status_code=400, detail="Tên đăng nhập đã tồn tại!")
            
            # Kiểm tra email đã tồn tại
            res = conn.execute(text("SELECT id FROM Users WHERE email = :e"), {"e": req.email})
            if res.fetchone():
                raise HTTPException(status_code=400, detail="Email đã được sử dụng!")
            
            # Hash mật khẩu
            password_hash = bcrypt.hashpw(req.password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            # Insert user
            conn.execute(
                text("INSERT INTO Users (username, email, full_name, password_hash) VALUES (:u, :e, :fn, :ph)"),
                {"u": req.username, "e": req.email, "fn": req.full_name, "ph": password_hash}
            )
            conn.commit()
        
        return {"success": True, "message": "Đăng ký thành công! Vui lòng đăng nhập."}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Register error: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi đăng ký: {str(e)}")

@app.post("/api/login")
def login_user(req: LoginRequest):
    try:
        with get_db_conn() as conn:
            res = conn.execute(text("SELECT id, username, full_name, email, password_hash FROM Users WHERE username = :u"), {"u": req.username})
            row = res.fetchone()
        
        if not row:
            raise HTTPException(status_code=401, detail="Tên đăng nhập không tồn tại!")
        
        user_id, username, full_name, email, password_hash = row
        
        # Verify password
        if not bcrypt.checkpw(req.password.encode('utf-8'), password_hash.encode('utf-8')):
            raise HTTPException(status_code=401, detail="Mật khẩu không đúng!")
        
        # Tạo session
        session_id = str(uuid.uuid4())
        active_sessions[session_id] = user_id
        
        response = JSONResponse(content={
            "success": True,
            "message": f"Chào mừng {full_name}!",
            "user": {
                "id": user_id,
                "username": username,
                "full_name": full_name,
                "email": email
            }
        })
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            max_age=86400 * 7,  # 7 ngày
            samesite="lax"
        )
        return response
    except HTTPException:
        raise
    except Exception as e:
        print(f"Login error: {e}")
        raise HTTPException(status_code=500, detail=f"Lỗi đăng nhập: {str(e)}")

@app.post("/api/forgot-password/send-otp")
def send_otp(req: SendOtpRequest):
    try:
        with get_db_conn() as conn:
            res = conn.execute(text("SELECT id FROM Users WHERE email = :e"), {"e": req.email})
            row = res.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Email không tồn tại trong hệ thống!")
        
        # Tạo mã OTP 6 chữ số
        otp_code = str(random.randint(100000, 999999))
        otp_store[req.email] = {
            "otp": otp_code,
            "expires": datetime.now() + timedelta(minutes=10)
        }
        
        # Gửi email qua SMTP
        if not SMTP_EMAIL or not SMTP_PASSWORD:
            print(f"[DEV MODE] OTP cho {req.email}: {otp_code}")
            return {"success": True, "message": f"Mã OTP đã được gửi tới {req.email}. Vui lòng kiểm tra hộp thư."}
        
        try:
            msg = MIMEMultipart()
            msg['From'] = SMTP_EMAIL
            msg['To'] = req.email
            msg['Subject'] = '🔐 AI Loan Predictor - Mã xác nhận đặt lại mật khẩu'
            
            body = f"""
            <html>
            <body style="font-family: 'Segoe UI', Arial, sans-serif; background: #0f0f23; color: #e2e8f0; padding: 40px;">
                <div style="max-width: 480px; margin: 0 auto; background: #1a1a3e; border-radius: 16px; padding: 36px; border: 1px solid rgba(139,92,246,0.3);">
                    <div style="text-align: center; margin-bottom: 24px;">
                        <div style="font-size: 28px; font-weight: 800; color: #8b5cf6;">AI Loan Predictor</div>
                        <p style="color: #94a3b8; font-size: 14px; margin-top: 4px;">Đặt lại mật khẩu</p>
                    </div>
                    <p style="font-size: 15px; color: #cbd5e1;">Mã xác nhận của bạn là:</p>
                    <div style="text-align: center; margin: 24px 0;">
                        <div style="display: inline-block; padding: 16px 40px; background: linear-gradient(135deg, #8b5cf6, #6d28d9); border-radius: 12px; font-size: 32px; font-weight: 800; letter-spacing: 8px; color: white;">
                            {otp_code}
                        </div>
                    </div>
                    <p style="font-size: 13px; color: #64748b; text-align: center;">Mã có hiệu lực trong <strong>10 phút</strong>. Không chia sẻ mã này cho ai.</p>
                </div>
            </body>
            </html>
            """
            msg.attach(MIMEText(body, 'html', 'utf-8'))
            
            server = smtplib.SMTP('smtp.gmail.com', 587)
            server.starttls()
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.sendmail(SMTP_EMAIL, req.email, msg.as_string())
            server.quit()
            
            return {"success": True, "message": f"Mã OTP đã được gửi tới {req.email}. Vui lòng kiểm tra hộp thư."}
        except Exception as mail_err:
            print(f"SMTP Error: {mail_err}")
            print(f"[FALLBACK] OTP cho {req.email}: {otp_code}")
            raise HTTPException(status_code=500, detail=f"Không thể gửi email. Vui lòng kiểm tra cấu hình SMTP.")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Send OTP error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/forgot-password/check-otp")
def check_otp(req: CheckOtpRequest):
    """Xác minh OTP đúng hay không (không reset mật khẩu, không xóa OTP)"""
    try:
        stored = otp_store.get(req.email)
        if not stored:
            raise HTTPException(status_code=400, detail="Chưa yêu cầu gửi mã OTP cho email này!")
        if datetime.now() > stored["expires"]:
            del otp_store[req.email]
            raise HTTPException(status_code=400, detail="Mã OTP đã hết hạn! Vui lòng gửi lại.")
        if stored["otp"] != req.otp:
            raise HTTPException(status_code=400, detail="Mã OTP không đúng!")
        return {"success": True, "message": "Mã OTP hợp lệ!"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/forgot-password/verify-otp")
def verify_otp(req: VerifyOtpRequest):
    try:
        # Kiểm tra OTP
        stored = otp_store.get(req.email)
        if not stored:
            raise HTTPException(status_code=400, detail="Chưa yêu cầu gửi mã OTP cho email này!")
        
        if datetime.now() > stored["expires"]:
            del otp_store[req.email]
            raise HTTPException(status_code=400, detail="Mã OTP đã hết hạn! Vui lòng gửi lại.")
        
        if stored["otp"] != req.otp:
            raise HTTPException(status_code=400, detail="Mã OTP không đúng!")
        
        # OTP hợp lệ — đặt lại mật khẩu
        with get_db_conn() as conn:
            new_hash = bcrypt.hashpw(req.new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            conn.execute(text("UPDATE Users SET password_hash = :ph WHERE email = :e"), {"ph": new_hash, "e": req.email})
            conn.commit()
        
        # Xóa OTP đã dùng
        del otp_store[req.email]
        
        return {"success": True, "message": "Đặt lại mật khẩu thành công! Vui lòng đăng nhập lại."}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Verify OTP error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/me")
def get_me(request: Request):
    user_id = get_current_user(request)
    if not user_id:
        return {"logged_in": False}
    try:
        with get_db_conn() as conn:
            res = conn.execute(text("SELECT id, username, full_name, email FROM Users WHERE id = :uid"), {"uid": user_id})
            row = res.fetchone()
        if not row:
            return {"logged_in": False}
        return {
            "logged_in": True,
            "user": {
                "id": row[0],
                "username": row[1],
                "full_name": row[2],
                "email": row[3]
            }
        }
    except Exception as e:
        print(f"Get me error: {e}")
        return {"logged_in": False}

@app.post("/api/logout")
def logout_user(request: Request):
    session_id = request.cookies.get("session_id")
    if session_id and session_id in active_sessions:
        del active_sessions[session_id]
    response = JSONResponse(content={"success": True, "message": "Đã đăng xuất."})
    response.delete_cookie("session_id")
    return response

# ══════════════════════════════════════════
#   PREDICTION HISTORY APIs
# ══════════════════════════════════════════

@app.post("/api/history/save")
async def save_history(request: Request):
    user_id = get_current_user(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Chưa đăng nhập")
    try:
        data = await request.json()
        form_data = data.get('formData', {})
        result_data = data.get('resultData', {})
        
        with get_db_conn() as conn:
            conn.execute(text("""
                INSERT INTO PredictionHistory 
                (user_id, sk_id_curr, prediction_code, risk_status, confidence, dti, pti, form_data, result_data)
                VALUES (:uid, :sk, :pc, :rs, :cf, :dti, :pti, :fd, :rd)
            """),
            {
                "uid": user_id,
                "sk": int(form_data.get('SK_ID_CURR', 0)),
                "pc": int(result_data.get('prediction_code', 0)),
                "rs": result_data.get('risk_status', ''),
                "cf": float(result_data.get('confidence', 0)),
                "dti": float(result_data.get('dti', 0)),
                "pti": float(result_data.get('pti', 0)),
                "fd": json.dumps(form_data, ensure_ascii=False),
                "rd": json.dumps(result_data, ensure_ascii=False)
            })
            conn.commit()
        return {"success": True, "message": "Đã lưu lịch sử."}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Save history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/history")
def get_history(request: Request):
    user_id = get_current_user(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Chưa đăng nhập")
    try:
        with get_db_conn() as conn:
            res = conn.execute(text("""
                SELECT id, sk_id_curr, prediction_code, risk_status, confidence, dti, pti, 
                       form_data, result_data, created_at
                FROM PredictionHistory 
                WHERE user_id = :uid 
                ORDER BY created_at DESC
            """), {"uid": user_id})
            rows = res.fetchall()
        
        history = []
        for row in rows:
            history.append({
                "id": row[0],
                "sk_id_curr": row[1],
                "prediction_code": row[2],
                "risk_status": row[3],
                "confidence": row[4],
                "dti": row[5],
                "pti": row[6],
                "formData": json.loads(row[7]) if row[7] else {},
                "resultData": json.loads(row[8]) if row[8] else {},
                "timestamp": row[9].isoformat() if row[9] else None
            })
        return {"success": True, "history": history}
    except HTTPException:
        raise
    except Exception as e:
        print(f"Get history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/history/{entry_id}")
def delete_history_entry(entry_id: int, request: Request):
    user_id = get_current_user(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Chưa đăng nhập")
    try:
        with get_db_conn() as conn:
            conn.execute(text("DELETE FROM PredictionHistory WHERE id = :eid AND user_id = :uid"), {"eid": entry_id, "uid": user_id})
            conn.commit()
        return {"success": True}
    except Exception as e:
        print(f"Delete history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/history")
def clear_all_history(request: Request):
    user_id = get_current_user(request)
    if not user_id:
        raise HTTPException(status_code=401, detail="Chưa đăng nhập")
    try:
        with get_db_conn() as conn:
            conn.execute(text("DELETE FROM PredictionHistory WHERE user_id = :uid"), {"uid": user_id})
            conn.commit()
        return {"success": True, "message": "Đã xóa toàn bộ lịch sử."}
    except Exception as e:
        print(f"Clear history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/api/feature_importance")
def feature_importance():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    if not hasattr(model, 'feature_importances_'):
        raise HTTPException(status_code=500, detail="Model does not support feature importances")
    
    importances = model.feature_importances_.tolist()
    names = feature_columns if feature_columns else [f"Feature {i}" for i in range(len(importances))]
    
    data = [
        {"feature": name, "importance": round(imp * 100, 2)}
        for name, imp in sorted(zip(names, importances), key=lambda x: -x[1])[:15]
    ]
    return {"features": data}

DATA_STATS = {
    "income_mean": 168000,
    "income_75pct": 202500,
    "income_90pct": 270000,
    "employment_mean": 2384,
    "default_rate": 0.0807,
    "total_records": 307511
}

OCCUPATION_MAP = {
    "Laborers": "Lao động phổ thông",
    "Core staff": "Nhân viên nòng cốt / Biên chế",
    "Accountants": "Kế toán",
    "Managers": "Quản lý",
    "Drivers": "Lái xe",
    "Sales staff": "Nhân viên bán hàng",
    "Cleaning staff": "Nhân viên vệ sinh",
    "Cooking staff": "Nhân viên nấu ăn",
    "Private service staff": "Nhân viên dịch vụ tư nhân",
    "Medicine staff": "Nhân viên y tế",
    "Security staff": "Nhân viên an ninh",
    "High skill tech staff": "Kỹ thuật viên bậc cao",
    "Waiters/barmen staff": "Bồi bàn / Pha chế",
    "Low-skill Laborers": "Lao động kỹ năng thấp",
    "Realty agents": "Nhân viên môi giới BĐS",
    "Secretaries": "Thư ký",
    "IT staff": "Nhân viên IT",
    "HR staff": "Nhân viên nhân sự",
}

OCCUPATION_MAP_REVERSE = {v: k for k, v in OCCUPATION_MAP.items()}

@app.get("/api/dashboard-data")
def get_dashboard_data():
    try:
        with get_db_conn() as conn:
            # 1. Thống kê theo độ tuổi (Nhóm tuổi từ cột AGE)
            age_dist = pd.read_sql(text("""
                SELECT 
                    CASE 
                        WHEN AGE BETWEEN 18 AND 25 THEN '18-25'
                        WHEN AGE BETWEEN 26 AND 35 THEN '26-35'
                        WHEN AGE BETWEEN 36 AND 45 THEN '36-45'
                        WHEN AGE BETWEEN 46 AND 55 THEN '46-55'
                        ELSE '55+' 
                    END as AgeGroup,
                    COUNT(*) as Count
                FROM Clients
                GROUP BY 
                    CASE 
                        WHEN AGE BETWEEN 18 AND 25 THEN '18-25'
                        WHEN AGE BETWEEN 26 AND 35 THEN '26-35'
                        WHEN AGE BETWEEN 36 AND 45 THEN '36-45'
                        WHEN AGE BETWEEN 46 AND 55 THEN '46-55'
                        ELSE '55+' 
                    END
                ORDER BY AgeGroup
            """), conn)

            # 2. Tỷ lệ rủi ro (Default Rate) trung bình theo Loại hình Hợp đồng
            risk_by_contract = pd.read_sql(text("""
                SELECT 
                    NAME_CONTRACT_TYPE as ContractType,
                    AVG(CAST(TARGET as float)) * 100 as DefaultRate
                FROM Clients
                GROUP BY NAME_CONTRACT_TYPE
            """), conn)
            
            # 3. Mức thu nhập trung bình theo Nghề nghiệp (Top 5)
            # Lưu ý SQL Server dùng TOP, Postgres dùng LIMIT
            if DATABASE_URL:
                income_query = """
                    SELECT 
                        OCCUPATION_TYPE as Occupation,
                        AVG(AMT_INCOME_TOTAL) as AvgIncome
                    FROM Clients 
                    WHERE OCCUPATION_TYPE IS NOT NULL
                    GROUP BY OCCUPATION_TYPE
                    ORDER BY AvgIncome DESC
                    LIMIT 5
                """
            else:
                 income_query = """
                    SELECT TOP 5
                        OCCUPATION_TYPE as Occupation,
                        AVG(AMT_INCOME_TOTAL) as AvgIncome
                    FROM Clients 
                    WHERE OCCUPATION_TYPE IS NOT NULL
                    GROUP BY OCCUPATION_TYPE
                    ORDER BY AvgIncome DESC
                """
            income_by_occ = pd.read_sql(text(income_query), conn)
            
            # 4. Trạng thái số lượng khoản vay (Tổng hồ sơ)
            total_clients = conn.execute(text("SELECT COUNT(*) FROM Clients")).scalar()
            high_risk_clients = conn.execute(text("SELECT COUNT(*) FROM Clients WHERE TARGET=1")).scalar()

        # Ánh xạ nghề nghiệp sang Tiếng Việt nếu có trong OCCUPATION_MAP
        income_occ_list = income_by_occ.to_dict('records')
        for item in income_occ_list:
             if item['Occupation'] in OCCUPATION_MAP:
                  item['Occupation'] = OCCUPATION_MAP[item['Occupation']]

        return {
            "age_distribution": age_dist.to_dict('records'),
            "risk_by_contract": risk_by_contract.to_dict('records'),
            "income_by_occupation": income_occ_list,
            "overall_stats": {
                "total_clients": int(total_clients),
                "high_risk_clients": int(high_risk_clients),
                "safe_clients": int(total_clients - high_risk_clients),
                "default_rate_overall": round((high_risk_clients/total_clients)*100, 2) if total_clients > 0 else 0
            }
        }
            
    except Exception as e:
        print("SQL Dashboard Error:", e)
        raise HTTPException(status_code=500, detail="Lỗi kết nối hoặc truy vấn Database.")

@app.get("/api/stats")
def get_dashboard_stats():
    return DATA_STATS

@app.get("/lookup/{sk_id}")
def lookup_client(sk_id: int):
    try:
        with get_db_conn() as conn:
            client_df = pd.read_sql(text("SELECT * FROM Clients WHERE SK_ID_CURR = :sid"), conn, params={"sid": sk_id})
            if not client_df.empty:
                data = client_df.iloc[0].dropna().to_dict()
                if data.get("OCCUPATION_TYPE") in OCCUPATION_MAP:
                    data["OCCUPATION_TYPE"] = OCCUPATION_MAP[data["OCCUPATION_TYPE"]]
                return data
    except Exception as e:
        print("SQL Lookup Error:", e)
        # Fallback to CSV if DB fails or table not found
            
    # Chế độ dùng file CSV
    if train_df_sample is None:
        if os.path.exists(TRAIN_DATA_PATH):
            full_df = pd.read_csv(TRAIN_DATA_PATH)
            client_data = full_df[full_df['SK_ID_CURR'] == sk_id]
            if not client_data.empty:
                return client_data.iloc[0].dropna().to_dict()
        raise HTTPException(status_code=404, detail="Client not found")
    
    client_data = train_df_sample[train_df_sample['SK_ID_CURR'] == sk_id]
    if client_data.empty:
        try:
            full_df = pd.read_csv(TRAIN_DATA_PATH)
            client_data = full_df[full_df['SK_ID_CURR'] == sk_id]
        except:
            pass
            
    if not client_data.empty:
        data = client_data.iloc[0].dropna().to_dict()
        if data.get("OCCUPATION_TYPE") in OCCUPATION_MAP:
            data["OCCUPATION_TYPE"] = OCCUPATION_MAP[data["OCCUPATION_TYPE"]]
        return data
    raise HTTPException(status_code=404, detail="Client ID không tồn tại trong hệ thống dữ liệu mẫu.")

@app.post("/api/predict")
def predict(request: LoanRequest):
    if model is None or feature_columns is None:
        raise HTTPException(status_code=500, detail="Model or features configuration not loaded")

    input_data = request.model_dump()
    if input_data.get("OCCUPATION_TYPE") in OCCUPATION_MAP_REVERSE:
        input_data["OCCUPATION_TYPE"] = OCCUPATION_MAP_REVERSE[input_data["OCCUPATION_TYPE"]]
        request.OCCUPATION_TYPE = input_data["OCCUPATION_TYPE"]
    
    historical_features = {}
    if request.SK_ID_CURR > 0:
        try:
            with get_db_conn() as conn:
                client_hist = pd.read_sql(text("SELECT * FROM Clients WHERE SK_ID_CURR = :sid"), conn, params={"sid": request.SK_ID_CURR})
                if not client_hist.empty:
                    historical_features = client_hist.iloc[0].dropna().to_dict()
        except Exception as e:
            print("Prediction history lookup error:", e)
        else:
            try:
                full_df = pd.read_csv(TRAIN_DATA_PATH)
                client_hist = full_df[full_df['SK_ID_CURR'] == request.SK_ID_CURR]
                if not client_hist.empty:
                    historical_features = client_hist.iloc[0].dropna().to_dict()
            except:
                pass

    final_features = {}
    for col in feature_columns:
        if col in input_data:
            final_features[col] = input_data[col]
        elif col in historical_features:
            final_features[col] = historical_features[col]
        else:
            final_features[col] = 0

    df = pd.DataFrame([final_features])

    if encoders:
        cat_cols = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
                    'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']
        for col in cat_cols:
            if col in df.columns and col in encoders:
                try:
                    df[col] = encoders[col].transform(df[col].astype(str))
                except:
                    df[col] = 0

    try:
        high_risk_prob = float(model.predict_proba(df[feature_columns])[0][1])
        
        if high_risk_prob >= 0.7: status, code = "High Risk", 1
        elif high_risk_prob >= 0.4: status, code = "Needs Consideration", 2
        else: status, code = "Low Risk", 0

        reasons = []
        suggestions = []
        
        dti = request.AMT_CREDIT / request.AMT_INCOME_TOTAL if request.AMT_INCOME_TOTAL > 0 else 999
        pti = (request.AMT_ANNUITY / request.AMT_INCOME_TOTAL * 100) if request.AMT_INCOME_TOTAL > 0 else 999
        
        if dti > 6:
            code = 1
            status = "High Risk"
            reasons.append(f"Tỷ lệ nợ/thu nhập (DTI = {dti:.1f}x) quá cao — Vượt mức 6 lần thu nhập.")
        elif dti > 4:
            reasons.append(f"Tỷ lệ nợ/thu nhập (DTI = {dti:.1f}x) ở mức cao.")
            suggestions.append("Giảm số tiền vay hoặc tăng thu nhập để cải thiện DTI.")
        
        if pti > 50:
            reasons.append(f"Tỷ lệ trả nợ/thu nhập (PTI = {pti:.0f}%) vượt 50% — Gánh nặng tài chính lớn.")
        
        if request.DAYS_EMPLOYED < 90:
            reasons.append("Kinh nghiệm làm việc dưới 3 tháng — Thiếu ổn định công việc.")
            suggestions.append("Tích lũy thêm kinh nghiệm làm việc (tối thiểu 6 tháng).")

        if request.AGE < 22:
            reasons.append(f"Tuổi ({request.AGE}) còn trẻ — Ít lịch sử tín dụng.")
            
        if request.ACTIVE_LOANS_COUNT > 0:
            reasons.append(f"Đang có {request.ACTIVE_LOANS_COUNT} khoản vay/mở thẻ tại các tổ chức tín dụng khác.")
            if request.ACTIVE_LOANS_COUNT > 2: code = max(code, 2)
            
        if request.CURRENT_DEBT_TOTAL > 0:
            reasons.append(f"Có dư nợ hiện tại khoảng {int(request.CURRENT_DEBT_TOTAL):,} ₫ ở tổ chức khác.")
            if request.CURRENT_DEBT_TOTAL > request.AMT_INCOME_TOTAL * 2: code = max(code, 1)
        
        if request.MAX_DPD > 0:
            reasons.append(f"Lịch sử từng trễ hạn thanh toán {request.MAX_DPD} ngày (Nợ chú ý).")
            code = max(code, 1)
            status = "High Risk"
            
        if request.OCCUPATION_TYPE and request.OCCUPATION_TYPE != 'nan' and request.OCCUPATION_TYPE != 'None':
            occ_vi = OCCUPATION_MAP.get(request.OCCUPATION_TYPE, request.OCCUPATION_TYPE)
            reasons.append(f"[INFO] Nghề nghiệp: {occ_vi}.")

        # ── Positive Indicators (Điểm cộng) ──
        if dti <= 3: reasons.append("[OK] Tỷ lệ nợ/thu nhập lành mạnh.")
        if request.DAYS_EMPLOYED >= 365: reasons.append("[OK] Công việc ổn định (trên 1 năm).")
        if request.FLAG_OWN_REALTY == 'Y': reasons.append("[OK] Có tài sản bất động sản thế chấp.")
        if request.AGE >= 25 and request.AGE <= 55: reasons.append("[OK] Độ tuổi lao động vàng, ổn định.")
        if request.AMT_INCOME_TOTAL > 200000: reasons.append("[OK] Thu nhập ở mức khá giỏi.")

        # ── FAIRNESS OVERRULE LAYER ──
        ok_count = sum(1 for r in reasons if r.startswith("[OK]"))
        critical_red_flags = request.MAX_DPD > 0 or dti > 6 or pti > 60 or (request.ACTIVE_LOANS_COUNT > 3)
        
        # Hard Overrule for Extremely Low Risk (Perfect Indicators)
        if dti < 0.1 and pti < 1.0 and not critical_red_flags:
            high_risk_prob = 0.05
            reasons.append("[INFO] Phân tích đặc biệt: Hồ sơ có các chỉ số tài chính hoàn hảo — Tự động phê duyệt ưu tiên.")
        elif high_risk_prob > 0.4 and ok_count >= 3 and not critical_red_flags:
            # Nếu hồ sơ quá tốt, ép rủi ro xuống thấp
            if high_risk_prob > 0.7: high_risk_prob = 0.55 # Chuyển từ High sang Needs Consideration
            elif high_risk_prob > 0.4: high_risk_prob = 0.25 # Chuyển từ Needs Consideration sang Low Risk
            reasons.append("[INFO] Hệ thống đã áp dụng quy tắc công bằng (Fairness Overrule) dựa trên các chỉ số tích cực của hồ sơ.")

        # Final re-evaluation of status and code
        if high_risk_prob >= 0.7: status, code = "High Risk", 1
        elif high_risk_prob >= 0.4: status, code = "Needs Consideration", 2
        else: status, code = "Low Risk", 0

        return {
            "risk_status": status,
            "prediction_code": code,
            "confidence": high_risk_prob,
            "reasons": reasons,
            "suggestions": suggestions,
            "client_found": bool(historical_features),
            "dti": round(dti, 2),
            "pti": round(pti, 1)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/chat")
async def chat(request: ChatRequest):
    try:
        model_name = "gemini-3-flash-preview"
        ai_model = genai.GenerativeModel(model_name)
        
        mapped_context = []
        for k, v in request.context.items():
            if k == 'Nghề nghiệp' and v in OCCUPATION_MAP:
                v = OCCUPATION_MAP[v]
            mapped_context.append(f"- {k}: {v}")
        context_str = "\n".join(mapped_context)
        
        prompt = f"""
        GIỚI HẠN BẮT BUỘC KHÔNG THỂ BỎ QUA: 
        Bạn là hệ thống Trợ lý ảo AI Loan Predictor – một hệ thống dự đoán rủi ro tín dụng. 
        Bạn CHỈ ĐƯỢC PHÉP trả lời các câu hỏi liên quan đến dự án này (dự đoán khoản vay, rủi ro tín dụng, hồ sơ vay, điểm tín dụng, thu nhập, nợ, mô hình học máy XGBoost, v.v.).
        Nếu người dùng hỏi BẤT CỨ ĐIỀU GÌ ngoài chủ đề này (ví dụ: Ai là tổng thống Mỹ, thời tiết hôm nay, làm thơ, v.v.), BẠN PHẢI TỪ CHỐI NGAY LẬP TỨC bằng đúng câu sau:
        "Xin lỗi, tôi chỉ hỗ trợ về hệ thống dự đoán khả năng vay vốn."

        QUY TẮC TRẢ LỜI: Ngắn gọn, súc tích, đi thẳng vào vấn đề nhưng phải đầy đủ ý chính.
        Đơn vị tiền tệ mặc định là Việt Nam Đồng (VND/₫).
        
        Dưới đây là thông tin của người dùng vừa nhập vào đồ án / form:
        {context_str}
        
        Người dùng hỏi: "{request.message}"
        
        Hãy trả lời bằng tiếng Việt, thân thiện và chuyên nghiệp. 
        Dựa vào dữ liệu cụ thể ở trên để giải thích kết quả dự đoán:
        - Nếu rủi ro cao: Giải thích lý do và cách khắc phục.
        - Nếu cần xem xét: Chỉ ra điểm yếu và điểm mạnh.
        - Nếu an toàn: Chúc mừng và lưu ý giữ vững tài chính.
        
        Sử dụng các số liệu thực tế họ đã nhập (thu nhập, khoản vay, tuổi, v.v.) khi trả lời.
        Nhớ định dạng Markdown cơ bản như **chữ đậm**, *chữ nghiêng*, hoặc danh sách.
        """
        
        response = ai_model.generate_content(prompt)
        return {"response": response.text}
    except Exception as e:
        print(f"Chat error: {e}")
        return {"response": "Xin lỗi, tôi gặp sự cố khi kết nối với bộ não AI. Vui lòng thử lại sau."}

@app.post("/api/export-pdf")
async def export_pdf(request: Request):
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch, mm
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
        from reportlab.lib import colors
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
        from io import BytesIO
        from datetime import datetime
        
        # --- VIETNAMESE SUPPORT ---
        FONT_PATH = r"C:\Windows\Fonts\arial.ttf"
        FONT_BOLD_PATH = r"C:\Windows\Fonts\arialbd.ttf"
        pdfmetrics.registerFont(TTFont('Arial', FONT_PATH))
        pdfmetrics.registerFont(TTFont('Arial-Bold', FONT_BOLD_PATH))

        TRANSLATIONS = {
            "NAME_CONTRACT_TYPE": {"Cash loans": "Khoản vay tiền mặt", "Revolving loans": "Khoản vay tiêu dùng"},
            "CODE_GENDER": {"M": "Nam", "F": "Nữ", "XNA": "Khác"},
            "NAME_EDUCATION_TYPE": {
                "Higher education": "Đại học",
                "Secondary / secondary special": "Trung học / Trung cấp",
                "Incomplete higher": "Chưa tốt nghiệp ĐH",
                "Lower secondary": "Dưới trung học",
                "Academic degree": "Bằng học thuật"
            },
            "NAME_FAMILY_STATUS": {
                "Married": "Đã kết hôn",
                "Single / not married": "Độc thân",
                "Civil marriage": "Kết hôn dân sự",
                "Separated": "Ly thân",
                "Widow": "Góa"
            },
            "NAME_HOUSING_TYPE": {
                "House / apartment": "Nhà riêng / Căn hộ",
                "With parents": "Ở cùng cha mẹ",
                "Rented apartment": "Nhà thuê",
                "Office apartment": "Căn hộ văn phòng",
                "Municipal apartment": "Căn hộ thuộc nhà nước",
                "Co-op apartment": "Căn hộ tập thể"
            }
        }

        def trans(key, val):
            return TRANSLATIONS.get(key, {}).get(val, val)

        data = await request.json()
        form_data = data.get('formData', {})
        result_data = data.get('resultData', {})
        
        # Create PDF in memory
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, leftMargin=10*mm, rightMargin=10*mm, topMargin=10*mm, bottomMargin=10*mm)
        
        # Create styles
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontName='Arial-Bold',
            fontSize=16,
            textColor=colors.HexColor('#6B46C1'),
            spaceAfter=6,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontName='Arial-Bold',
            fontSize=11,
            textColor=colors.HexColor('#2D3748'),
            spaceAfter=6,
            spaceBefore=8
        )
        
        normal_style = ParagraphStyle(
            'CustomNormal',
            parent=styles['Normal'],
            fontName='Arial',
            fontSize=9,
            spaceAfter=4
        )
        
        # Build content
        flowables = []
        
        # Header
        now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
        flowables.append(Paragraph("AI LOAN PREDICTOR", title_style))
        flowables.append(Paragraph("Báo Cáo Phân Tích Rủi Ro Tín Dụng", normal_style))
        flowables.append(Paragraph(f"Ngày xuất: {now}", normal_style))
        flowables.append(Spacer(1, 0.3*inch))
        
        # Status banner
        code = result_data.get('prediction_code', 1)
        if code == 0:
            status_text = "✓ ĐƯỢC DUYỆT - RỦI RO THẤP"
            status_color = colors.HexColor('#22C55E')
        elif code == 2:
            status_text = "⚠ CÂN NHẮC - RỦI RO TRUNG BÌNH"
            status_color = colors.HexColor('#F59E0B')
        else:
            status_text = "✗ TỪ CHỐI - RỦI RO CAO"
            status_color = colors.HexColor('#EF4444')
        
        confidence = result_data.get('confidence', 0)
        dti = result_data.get('dti', 0)
        pti = result_data.get('pti', 0)
        
        status_data = [[Paragraph(status_text, ParagraphStyle('status', parent=normal_style, fontName='Arial-Bold', textColor=colors.whitesmoke, alignment=TA_CENTER))]]
        status_table = Table(status_data, colWidths=[190*mm])
        status_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), status_color),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ('TOPPADDING', (0, 0), (-1, -1), 8)
        ]))
        flowables.append(status_table)
        flowables.append(Spacer(1, 0.1*inch))
        
        metric_text = f"Xác suất rủi ro: {(confidence*100):.1f}% | DTI: {dti}x | PTI: {pti}%"
        flowables.append(Paragraph(metric_text, ParagraphStyle('metric', parent=normal_style, alignment=TA_CENTER)))
        flowables.append(Spacer(1, 0.2*inch))
        
        # Info section
        flowables.append(Paragraph("THÔNG TIN HỒ SƠ", heading_style))
        
        info_rows = [
            ["Mã hồ sơ:", str(form_data.get('SK_ID_CURR', 'N/A'))],
            ["Giới tính:", "Nam" if form_data.get('CODE_GENDER') == 'M' else "Nữ"],
            ["Tuổi:", f"{form_data.get('AGE', '')} tuổi"],
            ["Học vấn:", trans("NAME_EDUCATION_TYPE", form_data.get('NAME_EDUCATION_TYPE', ''))],
            ["Hôn nhân:", trans("NAME_FAMILY_STATUS", form_data.get('NAME_FAMILY_STATUS', ''))],
            ["Nơi ở:", trans("NAME_HOUSING_TYPE", form_data.get('NAME_HOUSING_TYPE', ''))],
            ["Hợp đồng:", trans("NAME_CONTRACT_TYPE", form_data.get('NAME_CONTRACT_TYPE', ''))],
            ["Thu nhập/năm:", f"{form_data.get('AMT_INCOME_TOTAL', 0):,.0f} ₫"],
            ["Số tiền vay:", f"{form_data.get('AMT_CREDIT', 0):,.0f} ₫"],
            ["Nợ hiện tại:", f"{form_data.get('CURRENT_DEBT_TOTAL', 0):,.0f} ₫"],
            ["Số khoản nợ:", str(form_data.get('ACTIVE_LOANS_COUNT', 0))],
            ["Trễ hạn tối đa:", f"{form_data.get('MAX_DPD', 0)} ngày"],
        ]
        
        info_table = Table(info_rows, colWidths=[60*mm, 130*mm])
        info_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F3F4F6')),
            ('BACKGROUND', (1, 0), (1, -1), colors.HexColor('#FFFFFF')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#374151')),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Arial-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Arial'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#E5E7EB'))
        ]))
        flowables.append(info_table)
        flowables.append(Spacer(1, 0.15*inch))
        
        # Reasons section
        flowables.append(Paragraph("PHÂN TÍCH CHI TIẾT", heading_style))
        reasons = result_data.get('reasons', [])
        for reason in reasons:
            clean_reason = reason.replace('[OK] ', '').replace('[INFO] ', '')
            flowables.append(Paragraph(f"• {clean_reason}", normal_style))
        
        flowables.append(Spacer(1, 0.1*inch))
        
        # Suggestions section
        suggestions = result_data.get('suggestions', [])
        if suggestions:
            flowables.append(Paragraph("ĐỀ XUẤT CẢI THIỆN", heading_style))
            for suggestion in suggestions:
                flowables.append(Paragraph(f"→ {suggestion}", normal_style))
        
        flowables.append(Spacer(1, 0.2*inch))
        
        # Footer
        footer_text = "AI Loan Predictor - Báo cáo tự động"
        flowables.append(Paragraph(footer_text, ParagraphStyle('footer', parent=normal_style, fontSize=7, textColor=colors.grey, alignment=TA_CENTER)))
        
        # Build PDF
        doc.build(flowables)
        buffer.seek(0)
        
        # Return PDF file
        from fastapi.responses import FileResponse
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
            tmp.write(buffer.getvalue())
            tmp_path = tmp.name
        
        return FileResponse(tmp_path, media_type='application/pdf', filename=f"BaoCao-AI-{form_data.get('SK_ID_CURR', 'KH')}.pdf")
    
    except Exception as e:
        print(f"PDF Export error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"PDF generation error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
