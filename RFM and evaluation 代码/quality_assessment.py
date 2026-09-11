import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


rfm = pd.read_csv('optimized_rfm_cluster_results.csv')
K_FINAL = 4

# 4.1 计算更多聚类质量指标
def comprehensive_cluster_evaluation(rfm_df, features=['Recency_log', 'Frequency_log', 'Monetary_log']):
    """
    全面评估聚类质量
    """
    X = rfm_df[features].values
    labels = rfm_df['Cluster'].values

    # 4.1.1 Silhouette Score (已计算，这里重新计算确保一致性)
    sil_score = silhouette_score(X, labels)

    # 4.1.2 Calinski-Harabasz指数 - 值越大表示聚类质量越好
    from sklearn.metrics import calinski_harabasz_score
    ch_score = calinski_harabasz_score(X, labels)

    # 4.1.3 Davies-Bouldin指数 - 值越小表示聚类质量越好
    from sklearn.metrics import davies_bouldin_score
    db_score = davies_bouldin_score(X, labels)

    # 4.1.4 计算簇内距离与簇间距离的比例
    km = KMeans(n_clusters=len(np.unique(labels)), random_state=42, n_init=10)
    km.fit(X)

    # 簇内距离 (SSE)
    within_cluster_distance = km.inertia_

    # 计算簇间距离
    centers = km.cluster_centers_
    between_cluster_distance = 0
    total_points = len(X)
    counts = np.bincount(labels)

    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dist = np.linalg.norm(centers[i] - centers[j])
            weight = (counts[i] * counts[j]) / (total_points ** 2)
            between_cluster_distance += dist * weight

    # 计算比例
    distance_ratio = within_cluster_distance / (between_cluster_distance + 1e-10)  # 避免除零

    # 4.1.5 显示结果
    print("聚类质量综合评估:")
    print("-" * 50)
    print(f"Silhouette Score: {sil_score:.4f} (越大越好，范围[-1,1])")
    print(f"Calinski-Harabasz Score: {ch_score:.2f} (越大越好)")
    print(f"Davies-Bouldin Index: {db_score:.4f} (越小越好)")
    print(f"簇内/簇间距离比例: {distance_ratio:.2f} (越小越好)")

    # 4.1.6 可视化聚类质量指标
    plt.figure(figsize=(12, 5))

    # 1. Silhouette分析
    from sklearn.metrics import silhouette_samples
    silhouette_vals = silhouette_samples(X, labels)

    plt.subplot(1, 2, 1)
    y_lower, y_upper = 0, 0
    for i in range(K_FINAL):
        cluster_silhouette_vals = silhouette_vals[labels == i]
        cluster_silhouette_vals.sort()

        y_upper += len(cluster_silhouette_vals)
        color = plt.cm.nipy_spectral(float(i) / K_FINAL)
        plt.barh(range(y_lower, y_upper), cluster_silhouette_vals,
                 height=1.0, edgecolor='none', color=color)

        # 在每个聚类中间标记聚类编号
        plt.text(-0.05, (y_lower + y_upper) / 2, str(i))

        y_lower = y_upper

    plt.axvline(x=sil_score, color="red", linestyle="--")
    plt.title(f"Silhouette Analysis (Average = {sil_score:.2f})")
    plt.xlabel("Silhouette Coefficient")
    plt.ylabel("Cluster")
    plt.xlim([-0.1, 1])

    # 2. 距离比例可视化
    plt.subplot(1, 2, 2)
    plt.bar(['Intra-cluster Distance', 'Inter-cluster Distance'],
            [within_cluster_distance, between_cluster_distance * 100],
            color=['red', 'green'])
    plt.title('Intra-cluster Distance vs Inter-cluster Distance')
    plt.ylabel('Distance (Unit)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    return {
        'silhouette_score': sil_score,
        'calinski_harabasz_score': ch_score,
        'davies_bouldin_index': db_score,
        'within_cluster_distance': within_cluster_distance,
        'between_cluster_distance': between_cluster_distance,
        'distance_ratio': distance_ratio
    }


# 执行聚类质量综合评估
quality_metrics = comprehensive_cluster_evaluation(rfm.copy())