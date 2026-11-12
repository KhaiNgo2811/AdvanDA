"""Demo hệ thống cảnh báo rủi ro vỡ nợ khoản vay SBA 7(a) - Nhóm 01, PTDLNC (UEL).
Số liệu lấy từ báo cáo cuối kỳ (Bảng 3.18-3.23, 4.9). Trang 'Chấm điểm' là mô phỏng minh họa,
tự dùng mô hình thật nếu có file models/model_bundle.joblib (xem README)."""
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="SBA 7(a) · Cảnh báo rủi ro vỡ nợ", page_icon="📊", layout="wide")
NAVY, BLUE, ORANGE = "#0F2A47", "#15508C", "#C2410C"
RED, YEL, GRN = "#C0392B", "#D9A400", "#2F6B3A"

st.markdown(f"""<style>
h1,h2,h3{{color:{NAVY}}} .tier{{border-left:10px solid;padding:10px 16px;margin:6px 0;background:#fff;border-radius:10px}}
</style>""", unsafe_allow_html=True)

st.title("Dự báo rủi ro vỡ nợ khoản vay SBA 7(a)")
st.caption("Nhóm 01 · Phân tích dữ liệu nâng cao · UEL · GVHD: TS. Nguyễn Thôn Dã · FY2000–FY2019 + FRED-MD")

MODELS = pd.DataFrame({
    "Mô hình": ["Logistic (cơ sở)", "Logistic (+vĩ mô)", "Random Forest (+vĩ mô)", "XGBoost (cơ sở)", "XGBoost (+vĩ mô)"],
    "ROC-AUC valid": [0.535, 0.545, 0.554, 0.574, 0.590],
    "ROC-AUC test": [0.549, 0.558, 0.573, 0.592, 0.603],
    "PR-AUC valid": [0.073, 0.076, 0.077, 0.080, 0.087],
    "PR-AUC test": [0.084, 0.084, 0.088, 0.094, 0.097],
    "KS test": [0.067, 0.084, 0.107, 0.136, 0.150],
})
DECILE = pd.DataFrame({
    "Decile": range(1, 11),
    "Số khoản": [9607, 9606, 9607, 9606, 9607, 9606, 9606, 9607, 9606, 9607],
    "Tỷ lệ xóa nợ (%)": [10.9, 10.1, 8.8, 7.8, 7.1, 6.6, 6.1, 5.1, 4.5, 3.2],
    "% khoản xóa nợ bắt được": [15.5, 14.5, 12.5, 11.2, 10.2, 9.4, 8.7, 7.2, 6.4, 4.5],
    "Lift": [1.55, 1.45, 1.25, 1.12, 1.02, 0.94, 0.87, 0.72, 0.64, 0.45],
})
SHAP = pd.DataFrame({
    "Đặc trưng": ["naics_sector", "BorrState", "busloans_yoy_lag1", "fedfunds_lag1", "ProcessingMethod",
                  "BusinessAge", "log_gross", "unrate_lag1", "log_jobs", "BusinessType"],
    "Tỷ trọng SHAP (%)": [15.0, 14.0, 13.2, 9.9, 9.4, 7.7, 7.6, 5.4, 2.8, 2.5],
    "Loại": ["Khoản vay / doanh nghiệp", "Khoản vay / doanh nghiệp", "Vĩ mô", "Vĩ mô", "Khoản vay / doanh nghiệp",
             "Khoản vay / doanh nghiệp", "Khoản vay / doanh nghiệp", "Vĩ mô", "Khoản vay / doanh nghiệp", "Khoản vay / doanh nghiệp"],
})
TIERS = pd.DataFrame({
    "Tầng": ["Đỏ", "Vàng", "Xanh"], "% khoản vay": [15.7, 45.5, 38.8], "Tỷ lệ xóa nợ (%)": [10.8, 7.7, 4.7],
    "Hành động": ["Rà soát thủ công, ghi lý do nếu vẫn duyệt; mô hình không tự từ chối",
                  "Quy trình thẩm định chuẩn", "Có thể ưu tiên tiếp cận để mở rộng danh mục"],
})

tabs = st.tabs(["Tổng quan", "Mô hình", "Decile & SHAP", "Data leakage", "Dự báo chuỗi thời gian", "Phân tầng & chấm điểm"])

with tabs[0]:
    c = st.columns(4)
    c[0].metric("Bài toán", "Phân loại nhị phân", "có giám sát")
    c[1].metric("Nhãn", "CHGOFF = 1 / PIF = 0")
    c[2].metric("ROC-AUC test (XGBoost)", "0,603")
    c[3].metric("PR-AUC test (chỉ số chính)", "0,097", "≈1,4× ngẫu nhiên (0,070)")
    st.markdown("""
**Học máy có giám sát (supervised):** mỗi khoản vay trong quá khứ có nhãn thật (xóa nợ hay trả đủ), mô hình học quan hệ giữa đặc trưng lúc phê duyệt và nhãn.

**Quy trình:** chia dữ liệu theo thời gian (train / valid / test theo năm tài khóa) → Logistic Regression, Random Forest, XGBoost → trọng số lớp (scale_pos_weight ≈ 3,55), không dùng SMOTE → hiệu chỉnh xác suất bằng isotonic → giải thích bằng SHAP → phân tầng Đỏ/Vàng/Xanh → giám sát PSI và ROC-AUC theo cohort.

**Kết luận chính:** khi chỉ dùng thông tin biết lúc phê duyệt, khả năng phân biệt chỉ ở mức khiêm tốn, nên mô hình dùng để **quyết định mức rà soát**, không để tự động từ chối.
""")

with tabs[1]:
    st.subheader("So sánh mô hình (Bảng 3.18, đã loại biến kỳ hạn)")
    st.dataframe(MODELS, hide_index=True)
    m = st.radio("Chỉ số", ["PR-AUC test", "ROC-AUC test", "KS test"], horizontal=True)
    fig = px.bar(MODELS, x="Mô hình", y=m, text=m, color_discrete_sequence=[BLUE])
    fig.update_traces(textposition="outside")
    if m == "PR-AUC test":
        fig.add_hline(y=0.070, line_dash="dash", line_color=ORANGE, annotation_text="Mức ngẫu nhiên 0,070")
    st.plotly_chart(fig)
    st.info("Mô hình chọn theo PR-AUC trên valid (không nhìn test): XGBoost (+vĩ mô). Biến vĩ mô làm ROC-AUC tăng khoảng 0,011 nhưng PR-AUC không cải thiện có ý nghĩa.")

with tabs[2]:
    a, b = st.columns(2)
    with a:
        st.subheader("Decile lift (Bảng 3.20)")
        fig = px.bar(DECILE, x="Decile", y="Lift", text="Lift", color_discrete_sequence=[BLUE])
        fig.update_traces(textposition="outside"); fig.add_hline(y=1, line_dash="dash", line_color=ORANGE)
        st.plotly_chart(fig)
        st.caption("10% khoản vay rủi ro nhất có tỷ lệ xóa nợ 10,9% (lift 1,55); 10% thấp nhất chỉ 3,2% (lift 0,45).")
    with b:
        st.subheader("Top 10 đặc trưng theo SHAP (Bảng 3.21)")
        fig = px.bar(SHAP[::-1], x="Tỷ trọng SHAP (%)", y="Đặc trưng", orientation="h", color="Loại",
                     color_discrete_map={"Vĩ mô": ORANGE, "Khoản vay / doanh nghiệp": BLUE}, text="Tỷ trọng SHAP (%)")
        st.plotly_chart(fig)
        st.caption("Sáu biến FRED-MD cộng lại chiếm 34,0% đóng góp SHAP. SHAP là đóng góp trong mô hình, không phải quan hệ nhân quả.")
    st.dataframe(DECILE, hide_index=True)

with tabs[3]:
    st.subheader("Điểm mấu chốt: data leakage từ biến kỳ hạn (term)")
    d = pd.DataFrame({"Kịch bản": ["Loại feature term", "Giữ feature term", "Chỉ với feature term"], "ROC-AUC": [0.601, 0.965, 0.954]})
    fig = px.bar(d, x="Kịch bản", y="ROC-AUC", text="ROC-AUC", color="Kịch bản",
                 color_discrete_sequence=[BLUE, ORANGE, ORANGE])
    fig.update_traces(textposition="outside"); fig.update_layout(showlegend=False, yaxis_range=[0, 1.05])
    st.plotly_chart(fig)
    st.markdown("""
- **Data leakage:** feature chứa thông tin chỉ có sau khi đã biết target, nên điểm cao giả tạo.
- **Vì sao term là leakage:** kỳ hạn thực tế chỉ biết khi khoản vay đã kết thúc, lúc phê duyệt chưa có.
- **Ablation:** chạy cùng một XGBoost có và không có term; ROC-AUC nhảy từ 0,601 lên 0,965, nên nhóm loại feature này. **0,601 mới là hiệu năng thực tế tại thời điểm phê duyệt.**
""")

with tabs[4]:
    st.subheader("RQ2: số khoản vay theo tháng và khả năng dự báo")
    st.markdown("""
- **Xu hướng:** tăng đến 2006–2007, sau đó **giảm 78%** đến tháng 01/2011 (khủng hoảng tài chính).
- **Mùa vụ:** lặp lại mỗi năm, nhiều khoản vay nhất vào tháng 9, ít nhất vào tháng 1.
- **Walk-forward** (dự báo lùi nhiều lần, mỗi lần chỉ dùng dữ liệu đến thời điểm đó): ETS có sai số thấp nhất.
""")
    ts = pd.DataFrame({"Mô hình": ["ETS", "SARIMA", "SARIMAX (+vĩ mô)"], "RMSE (khoản/tháng)": [606, 646, 723]})
    fig = px.bar(ts, x="Mô hình", y="RMSE (khoản/tháng)", text="RMSE (khoản/tháng)", color_discrete_sequence=[BLUE])
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig)
    st.caption("Biến vĩ mô không giúp thêm: SARIMAX (723) kém hơn SARIMA (646). Trên hold-out 36 tháng, seasonal naive tốt nhất (RMSE 801), nên chênh lệch giữa các mô hình không lớn.")

with tabs[5]:
    st.subheader("Phân tầng rủi ro (Bảng 4.9)")
    cols = st.columns(3)
    for col, (_, r), colr in zip(cols, TIERS.iterrows(), [RED, YEL, GRN]):
        col.markdown(f"<div class='tier' style='border-color:{colr}'><b style='color:{colr}'>{r['Tầng'].upper()}</b> · {r['% khoản vay']}% khoản vay<br>"
                     f"Tỷ lệ xóa nợ <b>{r['Tỷ lệ xóa nợ (%)']}%</b><br><span style='font-size:0.9em'>{r['Hành động']}</span></div>",
                     unsafe_allow_html=True)
    fig = go.Figure()
    for (_, r), colr in zip(TIERS.iterrows(), [RED, YEL, GRN]):
        fig.add_bar(y=["Khoản vay"], x=[r["% khoản vay"]], name=r["Tầng"], orientation="h", marker_color=colr, text=f"{r['% khoản vay']}%")
    fig.update_layout(barmode="stack", height=160, margin=dict(t=10, b=10), xaxis_title="% khoản vay (test)")
    st.plotly_chart(fig)

    st.divider()
    st.subheader("Chấm điểm một hồ sơ")
    bundle = "models/model_bundle.joblib"
    real = os.path.exists(bundle)
    if real:
        st.success("Đang dùng mô hình thật (models/model_bundle.joblib).")
    else:
        st.warning("**Chế độ mô phỏng minh họa:** chưa có file mô hình thật nên điểm bên dưới được ước lượng từ các lift nhóm trong báo cáo (Bảng 3.14), "
                   "không phải xác suất của XGBoost. Dùng để minh họa luồng sử dụng; để dùng mô hình thật, xem README.")
    c1, c2, c3 = st.columns(3)
    method = c1.selectbox("Kênh xử lý (ProcessingMethod)", ["Thông thường", "Community Express", "International Trade Loans",
                                                           "Working Capital CAPLine", "7a with EWCP", "Preferred Lenders with EWCP"])
    btype = c2.selectbox("Loại hình doanh nghiệp", ["CORPORATION", "INDIVIDUAL", "PARTNERSHIP"])
    amount = c3.slider("Số tiền vay (USD)", 5_000, 2_000_000, 150_000, step=5_000)
    sector = st.selectbox("Ngành", ["Khác", "Y tế (62)", "Khai khoáng (21)"])
    LIFT_M = {"Thông thường": 1.0, "Community Express": 2.30, "International Trade Loans": 1.65,
              "Working Capital CAPLine": 0.27, "7a with EWCP": 0.21, "Preferred Lenders with EWCP": 0.12}
    LIFT_B = {"CORPORATION": 0.9, "INDIVIDUAL": 1.26, "PARTNERSHIP": 1.0}
    lift = LIFT_M[method] ** 0.5 * LIFT_B[btype] ** 0.5 * (1.15 if amount < 50_000 else 1.0) * (0.8 if sector.startswith("Y tế") else 1.0)
    p = min(max(0.070 * lift, 0.01), 0.40)
    tier = "Đỏ" if p >= 0.10 else ("Vàng" if p >= 0.065 else "Xanh")
    colr = {"Đỏ": RED, "Vàng": YEL, "Xanh": GRN}[tier]
    m1, m2 = st.columns(2)
    m1.metric("Xác suất xóa nợ ước lượng", f"{p:.1%}", f"{p/0.070:.2f}× mức chung (7,0%)")
    m2.markdown(f"<div class='tier' style='border-color:{colr}'><b style='color:{colr}'>Tầng {tier.upper()}</b><br>"
                f"{TIERS.set_index('Tầng').loc[tier, 'Hành động']}</div>", unsafe_allow_html=True)
    st.caption("Mô hình chỉ hỗ trợ quyết định mức rà soát, không tự động từ chối. Cần kiểm tra thêm công bằng và quyền riêng tư; "
               "giám sát hằng tháng bằng PSI (cảnh báo > 0,25) và ROC-AUC theo cohort (giảm > 0,03 so với lúc test).")
