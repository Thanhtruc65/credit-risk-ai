# 🚀 AI Loan Predictor - Hệ Thống Dự Đoán Vay Vốn Thông Minh

Một ứng dụng web hiện đại sử dụng Machine Learning để đánh giá rủi ro tín dụng và dự đoán khả năng phê duyệt khoản vay dựa trên hồ sơ khách hàng.

![Dashboard Preview](https://supabase.com/docs/img/guides/database/connection-string-uri.png) <!-- Bạn có thể thay bằng screenshot web của mình -->

## ✨ Tính năng nổi bật
- **Dự đoán Real-time:** Sử dụng mô hình XGBoost/Random Forest để đưa ra kết quả ngay lập tức.
- **Dashboard Phân Tích:** Biểu đồ trực quan về độ tuổi, thu nhập và tỷ lệ rủi ro.
- **Chatbot AI:** Tích hợp Google Gemini để tư vấn tài chính thông minh.
- **Quản lý linh hoạt:** Hỗ trợ cả SQL Server (Local) và PostgreSQL (Cloud/Supabase).
- **Hệ thống Auth:** Đăng ký, đăng nhập và lưu trữ lịch sử dự đoán riêng biệt cho từng người dùng.

## 📂 Cấu trúc dự án (Đã tối ưu)
```text
├── app.py                # Server chính (FastAPI)
├── requirements.txt      # Danh sách thư viện cần cài đặt
├── .env                  # Cấu hình biến môi trường (Bảo mật)
├── .gitignore            # Các file không đẩy lên GitHub
├── README.md             # Tài liệu dự án
│
├── models/               # Chứa các file "bộ não" AI (.pkl)
├── data/                 # Chứa dữ liệu mẫu (.csv)
├── static/               # Assets (CSS, JS, Images, Meteors effect)
├── templates/            # Giao diện HTML (Base, Dashboard, Predict,...)
│
├── scripts/              # Các kịch bản phụ trợ
│   ├── train.py          # Huấn luyện mô hình
│   ├── preprocess.py     # Tiền xử lý dữ liệu
│   ├── setup_cloud_db.py # Khởi tạo Database trên Cloud
│   └── ...               # Các file cài đặt khác
└── tests/                # Chứa các file kiểm thử (test_predict.py)
```

## 🛠 Hướng dẫn cài đặt

### 1. Cài đặt môi trường
```powershell
# Clone dự án
git clone https://github.com/Thanhtruc65/credit-risk-ai.git
cd credit-risk-ai

# Cài đặt thư viện
pip install -r requirements.txt
```

### 2. Cấu hình .env
Tạo file `.env` và điền các thông tin sau:
```text
DATABASE_URL=your_postgresql_url
GEMINI_API_KEY=your_google_ai_key
SMTP_EMAIL=your_email
SMTP_PASSWORD=your_app_password
```

### 3. Khởi chạy
```powershell
python app.py
```

## 🚀 Triển khai (Deployment)
Dự án đã được tối ưu để chạy trên **Render.com** kết hợp với **Supabase**.
- **Backend:** Render (Python/Gunicorn)
- **Database:** Supabase (PostgreSQL)

## 👤 Tác giả
- **Thanh Trúc** - [GitHub Profile](https://github.com/Thanhtruc65)

---
*Dự án này được phát triển cho mục đích đồ án và nghiên cứu khoa học.*
