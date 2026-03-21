# 🚀 AI Loan Predictor - Hệ Thống Dự Đoán Vay Vốn Thông Minh
> **Nền tảng phân tích tài chính và đánh giá rủi ro tín dụng ứng dụng Trí tuệ nhân tạo (Mô hình XGBoost & Gemini AI)**

[![Status](https://img.shields.io/badge/Status-Online-brightgreen)](https://huggingface.co/spaces/truc0605/credit-risk-ai-demo)
[![Demo](https://img.shields.io/badge/Demo-Live-blue)](https://huggingface.co/spaces/truc0605/credit-risk-ai-demo)

---

## 📖 Giới thiệu
**AI Loan Predictor** là giải pháp công nghệ số tiên tiến nhằm hỗ trợ các tổ chức tài chính và cá nhân trong việc đánh giá khả năng phê duyệt khoản vay. Hệ thống kết hợp sức mạnh của **Machine Learning (XGBoost)** để phân tích dữ liệu lịch sử và **Generative AI (Gemini)** để cung cấp trải nghiệm tư vấn thông minh, cá nhân hóa. Với giao diện Dashboard hiện đại theo phong cách **Glassmorphism**, dự án mang lại một không gian làm việc số chuyên nghiệp và trực quan.

---

## ✨ Tính năng nổi bật (Key Features)

### 1. 🤖 Lõi AI Dự Đoán Siêu Tốc
Sử dụng thuật toán **XGBoost** đã được tối ưu hóa cho dữ liệu tín dụng. Tốc độ xử lý kết quả cực nhanh với độ tin cậy cao, dựa trên các chỉ số tài chính thực tế của khách hàng.

### 2. 📊 Dashboard Tài Chính Đa Chiều
Hệ thống cung cấp các biểu đồ trực quan (Pie, Bar, Line) giúp theo dõi:
- Phân bổ rủi ro theo nhóm khách hàng.
- Tương quan giữa thu nhập và khả năng trả nợ.
- Thống kê lịch sử dự đoán theo thời gian thực.

### 3. 💬 Trợ Lý Ảo AI Thông Minh (Gemini 1.5 Flash)
Tích hợp Chatbot AI không chỉ trả lời các thắc mắc chung mà còn:
- Giải thích trực tiếp kết quả dự đoán dựa trên số liệu người dùng nhập.
- Tư vấn các chỉ số **DTI (Nợ/Thu nhập)** và **PTI (Trả khoản vay/Thu nhập)**.
- Đưa ra lời khuyên tài chính ngắn gọn, đúng trọng tâm.

### 4. 📝 Báo Cáo Chuyên Nghiệp (PDF Export)
Tự động xuất báo cáo chẩn đoán tài chính ra file PDF chuyên nghiệp với đầy đủ thông tin chi tiết, hỗ trợ lưu trữ và đối chiếu hồ sơ.

### 5. 🌍 Giao Diện Tùy Biến (Modern UI/UX)
- Cấu trúc **SPA (Single Page Application)** chuyển tab mượt mà.
- Hiệu ứng **Meteors & Glassmorphism** mang lại cảm giác cao cấp.
- Hệ thống **Auth** bảo mật, lưu trữ lịch sử chẩn đoán riêng biệt cho từng tài khoản.

---

## 🛠 Công nghệ sử dụng (Tech Stack)
- **Backend (Máy chủ)**: Python, FastAPI, Uvicorn (Asynchronous Processing).
- **Trí tuệ nhân tạo (AI)**: XGBoost, Scikit-learn, Google Gemini AI Engine.
- **Cơ sở dữ liệu**: Microsoft SQL Server (Local), Supabase/PostgreSQL (Cloud).
- **Frontend**: HTML5, CSS3 (Vanilla), JavaScript ES6+.
- **Deployment**: Local Service, Cloudflare Tunnel (HTTPS).

---

## 🚀 Hướng dấn cài đặt

### 1. Cấu hình môi trường (.env)
Tạo file `.env` tại thư mục gốc của dự án:
```powershell
GEMINI_API_KEY=your_google_ai_key
SMTP_EMAIL=your_email@gmail.com
SMTP_PASSWORD=your_app_password
```

### 2. Khởi chạy ứng dụng
```powershell
# Cài đặt thư viện
pip install -r requirements.txt

# Chạy server
python app.py
```
👉 Truy cập ngay tại: **https://huggingface.co/spaces/truc0605/credit-risk-ai-demo**
Do hệ thống demo hiện được triển khai trên nền tảng máy chủ miễn phí, nên trong một số thời điểm có thể xảy ra tình trạng không truy cập được hoặc phản hồi chậm.Nếu khong hiện kính mong thầy thử lại sau ạ!!
---

## ⚠️ Lưu ý quan trọng về Dữ liệu & Mô hình 

> Do kích thước Mô hình AI và Dataset khá lớn, vui lòng tải tại link Google Drive sau đây:
> [👉 Tải Dataset & Model tại đây](https://drive.google.com/drive/folders/1sQP6CtcF2bNDsCqHa2BKOMRzlWlt-fQU?usp=sharing)

### Sau khi tải về, bạn cần:

1.  Chép các tệp dữ liệu huấn luyện (`application_train.csv`, `bureau.csv`,...) vào thư mục **`data/`**.

---
*(Hệ thống được thiết kế nhằm mục đích nghiên cứu và hỗ trợ ra quyết định trong lĩnh vực tài chính tín dụng)*
