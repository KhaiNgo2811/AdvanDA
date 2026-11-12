# Demo cảnh báo rủi ro vỡ nợ SBA 7(a) - Nhóm 01 (PTDLNC, UEL)

Ứng dụng Streamlit trình bày kết quả đồ án: so sánh mô hình, decile lift, SHAP, data leakage, dự báo chuỗi thời gian, phân tầng Đỏ/Vàng/Xanh và trang chấm điểm hồ sơ.

## Chạy local
    pip install -r requirements.txt
    streamlit run app.py

## Đưa lên GitHub rồi deploy
1. Tạo repo mới trên GitHub, đẩy toàn bộ thư mục này lên (app.py ở thư mục gốc).
2. Vào https://share.streamlit.io, đăng nhập bằng GitHub, chọn repo, branch main, file `app.py`, bấm Deploy (khoảng 2 phút).

Lưu ý: Vercel chạy hàm serverless, không chạy được server Streamlit (cần WebSocket liên tục). Hãy deploy ở Streamlit Community Cloud; có thể dán link đó vào Vercel/trang cá nhân nếu cần.

## Dùng mô hình thật cho trang chấm điểm
Trong notebook 06/07 trên Colab, xuất `models/model_bundle.joblib` (mô hình XGBoost, bộ tiền xử lý, isotonic, ngưỡng) và đặt vào thư mục `models/`. Khi có file, app báo "Đang dùng mô hình thật"; hiện phần chấm điểm thật chưa nối vì dữ liệu huấn luyện (SBA, ~1 GB) nằm trên Drive của nhóm.
