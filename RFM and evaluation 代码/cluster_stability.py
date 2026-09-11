import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


rfm = pd.read_csv('optimized_rfm_cluster_results.csv')
# 设置随机种子确保可重现性
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# 1.1 Bootstrap聚类稳定性测试
def cluster_stability_test(rfm_df, n_iterations=50, sample_fraction=0.8):
    """
    通过bootstrap抽样测试聚类稳定性
    返回每个样本在不同迭代中被分配到相同聚类的一致性比例
    """
    # 准备数据
    features = ['Recency_log', 'Frequency_log', 'Monetary_log']
    X = rfm_df[features].values

    # 存储每次迭代的聚类标签
    all_labels = np.zeros((n_iterations, len(rfm_df)))

    for i in range(n_iterations):
        # 随机抽样
        sample_indices = np.random.choice(len(rfm_df),
                                          size=int(len(rfm_df) * sample_fraction),
                                          replace=False)
        sample_data = X[sample_indices]

        # 聚类
        km = KMeans(n_clusters=3, random_state=RANDOM_STATE + i, n_init=10)
        labels = km.fit_predict(sample_data)

        # 存储结果 (为未抽样的样本设为-1)
        iteration_labels = np.full(len(rfm_df), -1)
        iteration_labels[sample_indices] = labels
        all_labels[i] = iteration_labels

    # 计算稳定性得分
    stability_scores = []
    for i in range(len(rfm_df)):
        # 获取样本i在所有迭代中的标签
        sample_labels = all_labels[:, i]
        # 只考虑非-1的标签
        valid_labels = sample_labels[sample_labels != -1]
        if len(valid_labels) > 0:
            # 计算最常见标签的比例
            most_common = np.bincount(valid_labels.astype(int)).max()
            stability = most_common / len(valid_labels)
            stability_scores.append(stability)
        else:
            stability_scores.append(0)

    rfm_df['StabilityScore'] = stability_scores
    avg_stability = np.mean(stability_scores)

    print(f"平均聚类稳定性得分: {avg_stability:.3f}")
    print(f"稳定性得分分布:")
    print(pd.Series(stability_scores).describe())

    # 可视化稳定性分布
    plt.figure(figsize=(10, 6))
    sns.histplot(stability_scores, bins=20, kde=True)
    plt.title('Distribution of Clustering Stability Scores')
    plt.xlabel('Stability Score')
    plt.ylabel('Number of Customers')
    plt.axvline(x=avg_stability, color='r', linestyle='--', label=f'avg={avg_stability:.2f}')
    plt.legend()
    plt.show()

    return rfm_df


# 执行稳定性测试
rfm = cluster_stability_test(rfm.copy())