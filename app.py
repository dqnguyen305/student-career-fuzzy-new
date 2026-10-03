import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

from src.config import (
    DATA_PROCESSED_DIR,
    SUBJECT_MAP_VN,
    FCM_FUZZINESS,
)
from src.visualization.radar import plot_student_radar
from src.counseling.combination_mapper import extract_subject_averages
from src.clustering.fcm import calculate_fcm_membership
from src.counseling.manfis import build_manfis_input

st.set_page_config(
    page_title="Hệ Thống Tư Vấn Môn Học Mờ (Fuzzy Career)",
    page_icon="🎓",
    layout="wide"
)

# ==========================================
# HÀM VẼ BIỂU ĐỒ BAR MEMBERSHIP LÀM SẠCH TÊN NHÃN CỤM
# ==========================================
def plot_membership_bar(membership_series):
    """
    Vẽ biểu đồ Cụm Mờ và làm sạch tên nhãn cụm (loại bỏ suffix .1, .2 nếu có).
    """
    data = {}
    for col, val in membership_series.items():
        if col in ["student_id", "student_name", "class", "display_name"]:
            continue
        try:
            val_float = float(val)
            clean_cluster_name = str(col).replace("membership_", "").split(".")[0]
            data[clean_cluster_name] = val_float
        except (ValueError, TypeError):
            continue

    df_bar = pd.DataFrame({
        "Cụm Năng Lực": list(data.keys()),
        "Độ Thuộc (Membership)": list(data.values())
    })

    fig = px.bar(
        df_bar,
        x="Cụm Năng Lực",
        y="Độ Thuộc (Membership)",
        color="Cụm Năng Lực",
        text_auto=".2%",
        range_y=[0, 1.05],
        title="Mức Độ Thuộc Các Cụm Năng Lực Mờ"
    )
    fig.update_layout(showlegend=False, height=380, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def membership_details(membership_series, distance_series):
    """Create a comparable table of centroid distance and FCM membership."""
    cluster_names = distance_series.index.tolist()
    membership_values = [
        float(membership_series.get(f"membership_{name}", 0.0))
        for name in cluster_names
    ]
    return pd.DataFrame({
        "Cụm năng lực": cluster_names,
        "Khoảng cách theo hướng": distance_series.to_numpy(dtype=float),
        "Membership": membership_values
    })


def membership_summary(membership_series):
    """Summarize the strongest and secondary fuzzy memberships."""
    values = membership_series.astype(float).sort_values(ascending=False)
    top_cluster = values.index[0].replace("membership_", "").split(".")[0]
    second_cluster = values.index[1].replace("membership_", "").split(".")[0]
    overlap = 1.0 - float(values.iloc[0])
    entropy = float(-np.sum(values * np.log(np.clip(values, 1e-12, 1.0))) / np.log(len(values)))
    return {
        "top_cluster": top_cluster,
        "top_score": float(values.iloc[0]),
        "second_cluster": second_cluster,
        "second_score": float(values.iloc[1]),
        "overlap": overlap,
        "entropy": entropy
    }


def load_data():
    features_path = os.path.join(DATA_PROCESSED_DIR, "features.csv")
    membership_path = os.path.join(DATA_PROCESSED_DIR, "membership.csv")
    final_path = os.path.join(DATA_PROCESSED_DIR, "final_counseling_results.csv")
    manfis_final_path = os.path.join(DATA_PROCESSED_DIR, "manfis_counseling_results.csv")
    centroids_path = os.path.join(DATA_PROCESSED_DIR, "centroids.csv")
    manfis_model_path = os.path.join(DATA_PROCESSED_DIR, "manfis_model.pkl")
    manfis_metrics_path = os.path.join(DATA_PROCESSED_DIR, "manfis_metrics.csv")
    
    features_df = pd.read_csv(features_path)
    membership_df = pd.read_csv(membership_path)
    final_df = pd.read_csv(final_path)
    manfis_final_df = (
        pd.read_csv(manfis_final_path)
        if os.path.exists(manfis_final_path)
        else None
    )
    centroids_df = pd.read_csv(centroids_path) if os.path.exists(centroids_path) else None
    manfis_model = joblib.load(manfis_model_path) if os.path.exists(manfis_model_path) else None
    manfis_metrics = (
        pd.read_csv(manfis_metrics_path).iloc[0].to_dict()
        if os.path.exists(manfis_metrics_path)
        else None
    )

    features_df["student_id"] = features_df["student_id"].astype(str)
    membership_df["student_id"] = membership_df["student_id"].astype(str)
    final_df["student_id"] = final_df["student_id"].astype(str)
    if manfis_final_df is not None:
        manfis_final_df["student_id"] = manfis_final_df["student_id"].astype(str)

    return (
        features_df,
        membership_df,
        final_df,
        centroids_df,
        manfis_model,
        manfis_metrics,
        manfis_final_df,
    )

try:
    (
        features_df,
        membership_df,
        final_df,
        centroids_df,
        manfis_model,
        manfis_metrics,
        manfis_final_df,
    ) = load_data()
except Exception as e:
    st.error("❌ Chưa tìm thấy dữ liệu đã xử lý. Hãy chạy `python src/main.py` trước!")
    st.stop()


def predict_fcm_membership(user_avg_dict, centroids_df, m=FCM_FUZZINESS):
    """
    Dự đoán độ thuộc FCM Real-time đảm bảo khớp 100% thứ tự cột trong centroids.csv.
    """
    # 1. Tính điểm trung bình 3 miền năng lực [0, 10]
    nat_score = np.mean([user_avg_dict.get(m, user_avg_dict.get(SUBJECT_MAP_VN.get(m, m), 0.0)) for m in ["math", "physics", "chemistry", "biology", "informatics"]])
    soc_score = np.mean([user_avg_dict.get(m, user_avg_dict.get(SUBJECT_MAP_VN.get(m, m), 0.0)) for m in ["literature", "history", "geography"]])
    eng_score = user_avg_dict.get("english", user_avg_dict.get("Anh", 0.0))

    # 2. Tạo dictionary chuẩn hóa [0, 1]
    scaled_features = {
        "natural": float(nat_score) / 10.0,
        "social": float(soc_score) / 10.0,
        "english": float(eng_score) / 10.0
    }
    domain_mean = np.mean(list(scaled_features.values()))
    scaled_features = {
        key: value - domain_mean for key, value in scaled_features.items()
    }
    # 3. Dùng đúng thứ tự đặc trưng của centroid để tính khoảng cách và membership.
    feature_cols = [col for col in centroids_df.columns if col != "Cluster_Name"]
    user_vector = np.array([scaled_features[col] for col in feature_cols], dtype=float)
    res_membership, res_distances = calculate_fcm_membership(user_vector, centroids_df, m=m)

    # 6. Dữ liệu bổ trợ cho biểu đồ Radar
    radar_data = {}
    subjects_map = {
        'math': 'Toán', 'physics': 'Lý', 'chemistry': 'Hóa', 'biology': 'Sinh',
        'informatics': 'Tin học', 'literature': 'Văn', 'history': 'Sử',
        'geography': 'Địa', 'english': 'Anh'
    }
    for eng_k, vn_k in subjects_map.items():
        score = user_avg_dict.get(eng_k, user_avg_dict.get(vn_k, 0.0))
        radar_data[eng_k] = score
        radar_data[f"{eng_k}_avg"] = score
        radar_data[f"{eng_k}_scaled"] = score / 10.0
        radar_data[vn_k] = score

    res_features = pd.Series(radar_data)

    return res_membership, res_features, res_distances, {
        "natural": nat_score,
        "social": soc_score,
        "english": eng_score
    }


st.title("🎓 Hệ Thống Tư Vấn Hướng Nghiệp & Chọn Môn Thi THPT (Fuzzy Logic)")

tab1, tab2 = st.tabs(["📋 Tra Cứu Theo Danh Sách Lớp", "✍️ Nhập Điểm & Dự Đoán Phân Cụm FCM Real-time"])

# ==========================================
# TAB 1: TRA CỨU DANH SÁCH LỚP
# ==========================================
with tab1:
    st.sidebar.header("🔍 Lọc & Tra Cứu Học Sinh")
    available_classes = ["Tất cả"] + sorted(list(final_df["class"].dropna().unique()))
    selected_class = st.sidebar.selectbox("Chọn Lớp:", options=available_classes)

    filtered_df = final_df[final_df["class"] == selected_class].copy() if selected_class != "Tất cả" else final_df.copy()
    
    if filtered_df.empty:
        st.warning("⚠️ Không tìm thấy dữ liệu học sinh.")
    else:
        filtered_df["display_name"] = filtered_df["student_name"] + " (ID: " + filtered_df["student_id"] + ")"
        student_dict = dict(zip(filtered_df["display_name"], filtered_df["student_id"]))

        selected_display = st.sidebar.selectbox("Chọn Học Sinh:", options=list(student_dict.keys()))
        selected_s_id = student_dict[selected_display]

        student_info = final_df[final_df["student_id"] == selected_s_id].iloc[0]
        s_features = features_df[features_df["student_id"] == selected_s_id].iloc[0]
        s_domain_values = s_features[[
            "natural_score", "social_score", "english_score"
        ]].to_numpy(dtype=float) / 10.0
        s_domain_values -= s_domain_values.mean()
        s_membership, s_distances = calculate_fcm_membership(
            s_domain_values, centroids_df, m=FCM_FUZZINESS
        )

        top1_vn = SUBJECT_MAP_VN.get(str(student_info['top1_subject']).lower(), student_info['top1_subject'])
        top2_vn = SUBJECT_MAP_VN.get(str(student_info['top2_subject']).lower(), student_info['top2_subject'])

        st.subheader(f"📌 Kết Quả Tư Vấn: {student_info['student_name']} (Lớp {student_info['class']} - Mã HS: {selected_s_id})")

        col1, col2, col3 = st.columns([1, 1, 1.5])
        with col1:
            st.metric(label="🎯 Môn Ưu Tiên 1", value=f"{top1_vn.upper()}", delta=f"Score: {float(student_info['top1_score']):.4f}")
        with col2:
            st.metric(label="🎯 Môn Ưu Tiên 2", value=f"{top2_vn.upper()}", delta=f"Score: {float(student_info['top2_score']):.4f}")
        with col3:
            st.markdown("**📚 Top Tổ Hợp Xét Tuyển Đề Xuất:**")
            combis = [c.strip() for c in str(student_info['suggested_combinations']).split('|')]
            for combi in combis:
                st.markdown(f"🔹 `{combi}`")

        if manfis_final_df is not None:
            manfis_info_rows = manfis_final_df[
                manfis_final_df["student_id"] == selected_s_id
            ]
            if not manfis_info_rows.empty:
                manfis_info = manfis_info_rows.iloc[0]
                manfis_top1_vn = SUBJECT_MAP_VN.get(
                    str(manfis_info["top1_subject"]).lower(),
                    manfis_info["top1_subject"],
                )
                manfis_top2_vn = SUBJECT_MAP_VN.get(
                    str(manfis_info["top2_subject"]).lower(),
                    manfis_info["top2_subject"],
                )
                st.markdown("**Đề xuất MANFIS**")
                st.caption(
                    "MANFIS dùng luật mờ khởi tạo bằng FCM và được huấn luyện trên kết quả tư vấn hiện có."
                )
                manfis_col1, manfis_col2, manfis_col3 = st.columns([1, 1, 1.5])
                with manfis_col1:
                    st.metric(
                        "MANFIS Top 1",
                        manfis_top1_vn.upper(),
                        f"Score: {float(manfis_info['top1_score']):.4f}",
                    )
                with manfis_col2:
                    st.metric(
                        "MANFIS Top 2",
                        manfis_top2_vn.upper(),
                        f"Score: {float(manfis_info['top2_score']):.4f}",
                    )
                with manfis_col3:
                    st.markdown("**Tổ hợp MANFIS:**")
                    manfis_combinations = [
                        item.strip()
                        for item in str(manfis_info["suggested_combinations"]).split("|")
                    ]
                    for combination in manfis_combinations:
                        st.markdown(f"🔹 `{combination}`")

        st.divider()
        col_left, col_right = st.columns(2)
        with col_left:
            st.write("### 🕸️ Biểu Đồ Radar Năng Lực Các Môn")
            fig_radar_tab1 = plot_student_radar(s_features, student_info['student_name'])
            st.plotly_chart(fig_radar_tab1, use_container_width=True, key="radar_chart_tab1")
            
        with col_right:
            st.write("### 📊 Mức Độ Thuộc Cụm Mờ (FCM Membership)")
            summary = membership_summary(s_membership)
            st.caption(
                f"Top 1: **{summary['top_cluster']} ({summary['top_score']:.1%})** · "
                f"Top 2: **{summary['second_cluster']} ({summary['second_score']:.1%})** · "
                f"Giao thoa: **{summary['overlap']:.1%}** · "
                f"Entropy: **{summary['entropy']:.1%}**"
            )
            fig_bar_tab1 = plot_membership_bar(s_membership)
            st.plotly_chart(fig_bar_tab1, use_container_width=True, key="membership_bar_tab1")
            st.dataframe(
                membership_details(s_membership, s_distances).style.format({
                    "Khoảng cách theo hướng": "{:.6f}",
                    "Membership": "{:.2%}"
                }),
                hide_index=True,
                use_container_width=True
            )

# ==========================================
# TAB 2: NHẬP ĐIỂM & DỰ ĐOÁN PHÂN CỤM REAL-TIME
# ==========================================
with tab2:
    st.subheader("📝 Nhập Bảng Điểm 3 Học Kỳ - FCM & MANFIS")
    
    subjects_list = [
        ('Toán', 'math'), ('Văn', 'literature'), ('Lý', 'physics'), ('Hóa', 'chemistry'),
        ('Sinh', 'biology'), ('Tin học', 'informatics'), ('Sử', 'history'),
        ('Địa', 'geography'), ('Anh', 'english')
    ]

    with st.form("form_fcm_predict"):
        raw_row = {}
        c_h1, c_h2, c_h3, c_h4 = st.columns([1.5, 1, 1, 1])
        c_h1.markdown("**Môn Học**")
        c_h2.markdown("**Lớp 10**")
        c_h3.markdown("**Lớp 11**")
        c_h4.markdown("**Lớp 12 HK1**")

        for vn_name, eng_key in subjects_list:
            c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
            c1.markdown(f"📚 **{vn_name}**")
            raw_row[f"{eng_key}_10"] = c2.number_input(f"10_{eng_key}", 0.0, 10.0, 8.0, 0.1, label_visibility="collapsed")
            raw_row[f"{eng_key}_11"] = c3.number_input(f"11_{eng_key}", 0.0, 10.0, 8.5, 0.1, label_visibility="collapsed")
            raw_row[f"{eng_key}_12_hk1"] = c4.number_input(f"12_{eng_key}", 0.0, 10.0, 8.5, 0.1, label_visibility="collapsed")

        btn_predict = st.form_submit_button("🚀 Chạy Thuật Toán Dự Đoán Phân Cụm FCM", type="primary")

    if btn_predict:
        if centroids_df is None:
            st.error("❌ Thiếu file `centroids.csv`. Hãy chạy lại `python src/main.py` để khởi tạo tâm cụm!")
        else:
            student_series = pd.Series(raw_row)
            user_avg_dict = extract_subject_averages(student_series)

            pred_membership, pred_features, pred_distances, _ = predict_fcm_membership(
                user_avg_dict, centroids_df, m=FCM_FUZZINESS
            )

            st.markdown("---")

            if manfis_model is not None:
                manfis_averages = {
                    subject: float(user_avg_dict.get(vietnamese_name, 0.0))
                    for subject, vietnamese_name in {
                        "math": "Toán", "physics": "Lý", "chemistry": "Hóa",
                        "biology": "Sinh", "informatics": "Tin học",
                        "literature": "Văn", "history": "Sử",
                        "geography": "Địa", "english": "Anh",
                    }.items()
                }
                manfis_features = build_manfis_input(manfis_averages, raw_row)
                manfis_prediction = manfis_model.predict(manfis_features).iloc[0]
                st.write("### Xếp hạng môn bằng MANFIS")
                st.caption(
                    "MANFIS kết hợp luật mờ khởi tạo từ FCM với các đặc trưng điểm học sinh."
                )
                manfis_col1, manfis_col2 = st.columns(2)
                with manfis_col1:
                    st.metric(
                        "MANFIS Top 1",
                        SUBJECT_MAP_VN.get(manfis_prediction["top1_subject"], manfis_prediction["top1_subject"]),
                        f"Score: {float(manfis_prediction['top1_score']):.4f}",
                    )
                with manfis_col2:
                    st.metric(
                        "MANFIS Top 2",
                        SUBJECT_MAP_VN.get(manfis_prediction["top2_subject"], manfis_prediction["top2_subject"]),
                        f"Score: {float(manfis_prediction['top2_score']):.4f}",
                    )
                if manfis_metrics is not None:
                    st.caption(
                        "Hold-out trên nhãn giả: "
                        f"Top 1 accuracy {float(manfis_metrics['test_top1_accuracy']):.1%}, "
                        f"Top 2 accuracy {float(manfis_metrics['test_top2_accuracy']):.1%}."
                    )
            else:
                st.info("Chưa có model MANFIS. Hãy chạy lại `python src/main.py` để huấn luyện.")

            col_res1, col_res2 = st.columns(2)
            
            with col_res1:
                st.write("### 🕸️ Biểu Đồ Radar Năng Lực Chuẩn Hóa")
                try:
                    fig_radar_tab2 = plot_student_radar(pred_features, "Học Sinh Nhập Trực Tiếp")
                    st.plotly_chart(fig_radar_tab2, use_container_width=True, key="radar_chart_tab2")
                except Exception as err_radar:
                    st.error(f"Lỗi hiển thị Biểu đồ Radar: {err_radar}")

            with col_res2:
                st.write("### 📊 Mức Độ Thuộc Cụm FCM (Membership Degrees)")
                try:
                    fig_bar_tab2 = plot_membership_bar(pred_membership)
                    st.plotly_chart(fig_bar_tab2, use_container_width=True, key="membership_bar_tab2")
                    st.dataframe(
                        membership_details(pred_membership, pred_distances).style.format({
                            "Khoảng cách theo hướng": "{:.6f}",
                            "Membership": "{:.2%}"
                        }),
                        hide_index=True,
                        use_container_width=True
                    )
                except Exception as err_bar:
                    st.error(f"Lỗi hiển thị Biểu đồ Cụm Mờ: {err_bar}")