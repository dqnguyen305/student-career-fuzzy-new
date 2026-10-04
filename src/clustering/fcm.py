import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import numpy as np
import skfuzzy as fuzzy
from src.config import (
    DATA_PROCESSED_DIR,
    FCM_FUZZINESS,
    MEMBERSHIP_SCORE_SCALE,
)

def run_fcm_clustering(
    X_scaled_df: pd.DataFrame, 
    n_clusters: int = 3, 
    m: float = FCM_FUZZINESS,
    error: float = 0.005, 
    maxiter: int = 1000,
    seed: int = 42
) -> tuple[np.ndarray, np.ndarray, float, dict]:
    """
    Chạy thuật toán FCM trên độ lệch tương đối của 3 miền [natural, social, english]
    và ánh xạ duy nhất từng cụm năng lực tương ứng.
    """
    data_matrix = X_scaled_df.T.values

    cntr, u, u0, d, jm, p, fpc = fuzzy.cmeans(
        data=data_matrix,
        c=n_clusters,
        m=m,
        error=error,
        maxiter=maxiter,
        init=None,
        seed=seed
    )
    raw_u = u.copy()
    u = calculate_membership_matrix(X_scaled_df.to_numpy(), cntr, m=m)
    fpc = float(np.sum(u ** 2) / u.shape[1])

    cols = list(X_scaled_df.columns) # ['natural', 'social', 'english']
    
    if n_clusters != len(cols):
        raise ValueError("Số cụm phải bằng số miền năng lực để ánh xạ nhãn.")

    domain_labels = {
        "natural": "Nhóm Tự nhiên",
        "social": "Nhóm Xã hội",
        "english": "Nhóm Ngoại ngữ"
    }
    centroid_dominance = cntr - cntr.mean(axis=1, keepdims=True)
    ranked_assignments = sorted(
        (
            float(centroid_dominance[i, j]),
            i,
            cols[j]
        )
        for i in range(n_clusters)
        for j in range(n_clusters)
    )[::-1]
    cluster_mapping = {}
    used_clusters = set()
    used_domains = set()
    for _, cluster_index, domain in ranked_assignments:
        if cluster_index not in used_clusters and domain not in used_domains:
            cluster_mapping[cluster_index] = domain_labels[domain]
            used_clusters.add(cluster_index)
            used_domains.add(domain)
    remaining_labels = [domain_labels[domain] for domain in cols if domain not in used_domains]
    for cluster_index in range(n_clusters):
        if cluster_index not in cluster_mapping:
            cluster_mapping[cluster_index] = remaining_labels.pop(0)

    label_to_domain = {label: domain for domain, label in domain_labels.items()}
    semantic_labels = [cluster_mapping[i] for i in range(n_clusters)]
    semantic_domains = [label_to_domain[label] for label in semantic_labels]
    semantic_centroids = domain_prototypes(semantic_labels, cols)
    u = calculate_membership_matrix(
        X_scaled_df.to_numpy(),
        semantic_centroids,
        m=m,
        score_order=[cols.index(name) for name in semantic_domains]
    )
    fpc = float(np.sum(u ** 2) / u.shape[1])

    return cntr, u, fpc, cluster_mapping, raw_u


def calculate_fcm_membership(
    feature_vector: np.ndarray,
    centroids_df: pd.DataFrame,
    m: float = FCM_FUZZINESS
) -> tuple[pd.Series, pd.Series]:
    """Calculate centroid distances and FCM memberships for one student."""
    if m <= 1.0:
        raise ValueError("Tham số m của FCM phải lớn hơn 1.")
    if "Cluster_Name" not in centroids_df.columns:
        raise ValueError("centroids_df phải có cột Cluster_Name.")

    feature_cols = [col for col in centroids_df.columns if col != "Cluster_Name"]
    user_vector = np.asarray(feature_vector, dtype=float)
    centroids = domain_prototypes(centroids_df["Cluster_Name"].tolist(), feature_cols)

    if user_vector.shape != (len(feature_cols),):
        raise ValueError("Vector đầu vào không khớp số chiều của centroid.")

    cluster_names = centroids_df["Cluster_Name"].tolist()
    label_to_domain = {
        "Nhóm Tự nhiên": "natural",
        "Nhóm Xã hội": "social",
        "Nhóm Ngoại ngữ": "english",
    }
    memberships = calculate_membership_matrix(
        user_vector[None, :],
        centroids,
        m=m,
        score_order=[feature_cols.index(label_to_domain[name]) for name in cluster_names]
    )[:, 0]
    distances = directional_distances(user_vector, centroids)

    membership_series = pd.Series(
        memberships,
        index=[f"membership_{name}" for name in cluster_names],
        dtype=float
    )
    distance_series = pd.Series(distances, index=cluster_names, dtype=float)
    return membership_series, distance_series


def domain_prototypes(cluster_names: list[str], feature_cols: list[str]) -> np.ndarray:
    """Return semantic direction prototypes for the three business domains."""
    prototype_by_domain = {
        "Nhóm Tự nhiên": np.array([2.0, -1.0, -1.0]),
        "Nhóm Xã hội": np.array([-1.0, 2.0, -1.0]),
        "Nhóm Ngoại ngữ": np.array([-1.0, -1.0, 2.0]),
    }
    try:
        prototypes = np.array([prototype_by_domain[name] for name in cluster_names])
    except KeyError as error:
        raise ValueError(f"Nhãn cụm không hợp lệ: {error.args[0]}") from error
    if feature_cols != ["natural", "social", "english"]:
        raise ValueError("Membership cần đúng thứ tự natural, social, english.")
    return prototypes


def directional_distances(
    feature_matrix: np.ndarray,
    centroids: np.ndarray
) -> np.ndarray:
    """Measure distance between normalized directions, not vector magnitudes."""
    features = np.asarray(feature_matrix, dtype=float)
    centers = np.asarray(centroids, dtype=float)
    features_2d = np.atleast_2d(features)
    feature_norms = np.linalg.norm(features_2d, axis=1, keepdims=True)
    center_norms = np.linalg.norm(centers, axis=1, keepdims=True)
    normalized_features = np.divide(
        features_2d, feature_norms, out=np.zeros_like(features_2d), where=feature_norms > 1e-12
    )
    normalized_centers = np.divide(
        centers, center_norms, out=np.zeros_like(centers), where=center_norms > 1e-12
    )
    distances = np.linalg.norm(
        normalized_features[:, None, :] - normalized_centers[None, :, :], axis=2
    )
    zero_features = feature_norms[:, 0] <= 1e-12
    if np.any(zero_features):
        distances[zero_features, :] = 1.0
    return distances[0] if np.ndim(feature_matrix) == 1 else distances


def calculate_membership_matrix(
    feature_matrix: np.ndarray,
    centroids: np.ndarray,
    m: float = FCM_FUZZINESS,
    score_order: list[int] | None = None
) -> np.ndarray:
    """Calculate fuzzy memberships from directional centroid distances."""
    if m <= 1.0:
        raise ValueError("Tham số m của FCM phải lớn hơn 1.")
    distances = directional_distances(feature_matrix, centroids)
    distances_2d = np.atleast_2d(distances)
    memberships = np.zeros_like(distances_2d)
    zero_distance = distances_2d <= 1e-12
    scores_2d = np.atleast_2d(np.asarray(feature_matrix, dtype=float))
    if score_order is not None:
        scores_2d = scores_2d[:, score_order]
    for row_index, row in enumerate(distances_2d):
        scores = scores_2d[row_index]
        scaled_scores = MEMBERSHIP_SCORE_SCALE * (scores - np.max(scores))
        memberships[row_index] = np.exp(scaled_scores)
        memberships[row_index] /= memberships[row_index].sum()
    return memberships.T


def build_cluster_label_diagnostics(
    centroids: np.ndarray,
    cluster_mapping: dict[int, str],
    feature_names: list[str]
) -> pd.DataFrame:
    """Report the dominant domain and confidence of each assigned label."""
    rows = []
    for cluster_index, centroid in enumerate(centroids):
        domain_values = pd.Series(centroid, index=feature_names, dtype=float)
        ordered = domain_values.sort_values(ascending=False)
        dominant_domain = str(ordered.index[0])
        second_value = float(ordered.iloc[1])
        dominance_margin = float(ordered.iloc[0] - second_value)
        rows.append({
            "Cluster_Index": cluster_index,
            "Assigned_Label": cluster_mapping[cluster_index],
            "Dominant_Domain": dominant_domain,
            "Dominance_Margin": round(dominance_margin, 6),
            "natural": float(centroid[feature_names.index("natural")]),
            "social": float(centroid[feature_names.index("social")]),
            "english": float(centroid[feature_names.index("english")])
        })
    return pd.DataFrame(rows)


def generate_cluster_profiles(
    features_df: pd.DataFrame, 
    membership_df: pd.DataFrame, 
    subject_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Thống kê mô tả đặc điểm từng cụm học sinh chuyên sâu.
    """
    membership_cols = [c for c in membership_df.columns if c.startswith("membership_")]
    
    merged_df = features_df.copy()
    merged_df["Primary_Cluster"] = membership_df[membership_cols].idxmax(axis=1).apply(lambda x: str(x).replace("membership_", "").split(".")[0])
    merged_df["Max_Membership"] = membership_df[membership_cols].max(axis=1)

    total_students = len(merged_df)
    
    raw_sub_cols = [c for c in subject_cols if c.endswith("_avg")]
    if not raw_sub_cols:
        raw_sub_cols = ["math_avg", "physics_avg", "chemistry_avg", "biology_avg", "literature_avg", "history_avg", "english_avg"]

    overall_means = merged_df[raw_sub_cols].mean()
    profiles = []

    for cluster_name, group in merged_df.groupby("Primary_Cluster"):
        n_count = len(group)
        pct = (n_count / total_students) * 100
        avg_membership = group["Max_Membership"].mean()

        sub_means = group[raw_sub_cols].mean()
        sub_stds = group[raw_sub_cols].std()

        diff_from_overall = sub_means - overall_means
        top_subject = diff_from_overall.idxmax().replace("_avg", "")
        weak_subject = diff_from_overall.idxmin().replace("_avg", "")

        profile_row = {
            "Nhóm năng lực": cluster_name,
            "Số học sinh": n_count,
            "Tỷ lệ (%)": round(pct, 2),
            "Độ thuộc TB": round(avg_membership, 4),
            "Môn thế mạnh nhất": top_subject,
            "Môn yếu nhất": weak_subject
        }

        for sub in raw_sub_cols:
            sub_clean = sub.replace("_avg", "")
            profile_row[f"TB_{sub_clean}"] = round(sub_means[sub], 2)
            profile_row[f"SD_{sub_clean}"] = round(sub_stds[sub], 2)

        profiles.append(profile_row)

    profile_df = pd.DataFrame(profiles)

    class_dist_df = pd.crosstab(
        merged_df["class"], 
        merged_df["Primary_Cluster"], 
        margins=True, 
        margins_name="Tổng số"
    )

    return profile_df, class_dist_df


if __name__ == "__main__":
    try:
        scaled_path = os.path.join(DATA_PROCESSED_DIR, "normalized_scores.csv")
        features_path = os.path.join(DATA_PROCESSED_DIR, "features.csv")
        
        if not os.path.exists(scaled_path):
            raise FileNotFoundError("❌ Chưa có file normalized_scores.csv, hãy chạy feature_engineering.py trước.")

        X_scaled_df = pd.read_csv(scaled_path)
        features_df = pd.read_csv(features_path)

        print("🔄 Đang thực hiện phân cụm mờ Fuzzy C-Means (FCM)...")
        cntr, u, fpc, cluster_mapping, _ = run_fcm_clustering(
            X_scaled_df, n_clusters=3, m=FCM_FUZZINESS
        )

        print(f"✅ Phân cụm hoàn tất! Chỉ số FPC: {fpc:.4f}")
        for idx, name in cluster_mapping.items():
            print(f"Cluster {idx} ──> Nhóm năng lực: {name}")

        # Tạo DataFrame Tâm cụm và lưu chuẩn theo thứ tự các cột đặc trưng
        cntr_df = pd.DataFrame(cntr, columns=X_scaled_df.columns)
        cntr_df["Cluster_Name"] = [cluster_mapping[i] for i in range(len(cntr))]
        
        cols_order = [c for c in ['natural', 'social', 'english'] if c in cntr_df.columns] + ['Cluster_Name']
        cntr_df = cntr_df[cols_order]

        cntr_output_path = os.path.join(DATA_PROCESSED_DIR, "centroids.csv")
        cntr_df.to_csv(cntr_output_path, index=False, encoding="utf-8-sig")

        u_df = pd.DataFrame(u.T, columns=[f"membership_{cluster_mapping[i]}" for i in range(len(cntr))])
        membership_result = pd.concat([features_df[["student_id", "student_name", "class"]], u_df], axis=1)
        output_path = os.path.join(DATA_PROCESSED_DIR, "membership.csv")
        membership_result.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"✅ Đã lưu ma trận độ thuộc Membership U tại: {output_path}")

        profile_df, class_dist_df = generate_cluster_profiles(features_df, membership_result, list(features_df.columns))
        
        profile_path = os.path.join(DATA_PROCESSED_DIR, "cluster_profile_summary.csv")
        profile_df.to_csv(profile_path, index=False, encoding="utf-8-sig")
        
        class_dist_path = os.path.join(DATA_PROCESSED_DIR, "cluster_class_distribution.csv")
        class_dist_df.to_csv(class_dist_path, encoding="utf-8-sig")

        print(f"✅ Đã xuất bảng phân tích đặc điểm các nhóm học sinh tại: {profile_path}")
        print(f"✅ Đã xuất bảng phân bố theo lớp tại: {class_dist_path}")

    except Exception as err:  
        print(f"❌ Lỗi chạy FCM: {err}")