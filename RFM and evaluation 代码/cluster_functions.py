import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 准备数据 (使用您的rfm_scaled)
import numpy as np
from sklearn.preprocessing import StandardScaler

# 假设您已经计算了基础RFM指标
# rfm 应该是一个包含 'Recency', 'Frequency', 'Monetary' 列的DataFrame
from report_generate import file_name
rfm = pd.read_csv(file_name)

rfm['Recency_log'] = np.log1p(rfm['Recency'])
rfm['Frequency_log'] = np.log1p(rfm['Frequency'])
rfm['Monetary_log'] = np.log1p(rfm['Monetary'])

# 2. 准备用于标准化的特征
features = ['Recency_log', 'Frequency_log', 'Monetary_log']

# 3. 应用标准化处理
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[features])

X = rfm_scaled[:, :3]  # 取Recency, Frequency, Monetary

# 2. 定义多种聚类方法
clustering_algorithms = [
    ("KMeans", KMeans(n_clusters=3, random_state=42, n_init=50)),
    ("DBSCAN", DBSCAN(eps=0.3, min_samples=10)),
    ("Agglomerative", AgglomerativeClustering(n_clusters=3, linkage='ward')),
    ("GaussianMixture", GaussianMixture(n_components=3, random_state=42, covariance_type='full')),
    ("Birch", Birch(n_clusters=3, threshold=0.5))
]

# 3. 存储结果
results = []

for name, algorithm in clustering_algorithms:
    # 执行聚类
    algorithm.fit(X)

    # 获取标签 (处理不同算法的输出差异)
    if hasattr(algorithm, 'labels_'):
        labels = algorithm.labels_
    else:
        labels = algorithm.predict(X)

    # 跳过无效聚类 (如DBSCAN可能只产生1个簇)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters <= 1:
        print(f"Skipping {name} - only {n_clusters} valid clusters")
        continue

    # 计算评估指标
    sil_score = silhouette_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    db_score = davies_bouldin_score(X, labels)

    # 保存结果
    results.append({
        'Algorithm': name,
        'Clusters': n_clusters,
        'Silhouette': sil_score,
        'Calinski-Harabasz': ch_score,
        'Davies-Bouldin': db_score
    })

    # 临时保存标签用于可视化
    rfm[f'Cluster_{name}'] = labels

# 4. 创建结果DataFrame
results_df = pd.DataFrame(results)
print("聚类算法评估结果:")
print(results_df.round(3))

# 5. 可视化比较
plt.figure(figsize=(12, 6))
metrics = ['Silhouette', 'Calinski-Harabasz', 'Davies-Bouldin']
for i, metric in enumerate(metrics, 1):
    plt.subplot(1, 3, i)
    sns.barplot(data=results_df, x='Algorithm', y=metric)
    plt.title(f'{metric} Score by Algorithm')
    plt.xticks(rotation=45)
    if metric == 'Davies-Bouldin':
        plt.gca().invert_yaxis()  # 该指标越小越好
plt.tight_layout()
plt.show()

# 6. 最佳算法选择 (基于综合评分)
results_df['Davies-Bouldin_Normalized'] = 1 - (results_df['Davies-Bouldin'] - results_df['Davies-Bouldin'].min()) / (
            results_df['Davies-Bouldin'].max() - results_df['Davies-Bouldin'].min())
results_df['Composite_Score'] = (results_df['Silhouette'] +
                                 (results_df['Calinski-Harabasz'] - results_df['Calinski-Harabasz'].min()) / (
                                             results_df['Calinski-Harabasz'].max() - results_df[
                                         'Calinski-Harabasz'].min()) +
                                 results_df['Davies-Bouldin_Normalized']) / 3

best_algorithm = results_df.loc[results_df['Composite_Score'].idxmax(), 'Algorithm']
print(f"最佳聚类算法: {best_algorithm}")
print("综合评分详情:")
print(results_df[['Algorithm', 'Composite_Score']].round(3))