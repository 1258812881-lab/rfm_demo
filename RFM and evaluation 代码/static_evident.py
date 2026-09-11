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


# 2.1 对每个RFM维度执行Kruskal-Wallis检验（非参数检验，适合非正态分布数据）
def cluster_significance_test(rfm_df):
    """
    执行Kruskal-Wallis检验验证聚类间差异的统计显著性
    """
    results = {}
    metrics = ['Recency', 'Frequency', 'Monetary']

    print("聚类间差异的Kruskal-Wallis检验结果:")
    print("-" * 60)

    for metric in metrics:
        # 获取每个聚类的指标值
        clusters = sorted(rfm_df['Cluster'].unique())
        cluster_data = [rfm_df[rfm_df['Cluster'] == c][metric].values for c in clusters]

        # 执行Kruskal-Wallis检验
        stat, p_value = stats.kruskal(*cluster_data)
        results[metric] = {'statistic': stat, 'p_value': p_value}

        print(f"{metric}:")
        print(f"  统计量 = {stat:.4f}, p值 = {p_value:.6f}")
        if p_value < 0.001:
            print(f"  *** 聚类间存在极显著差异 (p < 0.001) ***")
        elif p_value < 0.01:
            print(f"  ** 聚类间存在显著差异 (p < 0.01) **")
        elif p_value < 0.05:
            print(f"  * 聚类间存在一定差异 (p < 0.05) *")
        else:
            print(f"  聚类间差异不显著 (p >= 0.05)")

    # 2.2 可视化置信区间
    plt.figure(figsize=(15, 5))

    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 3, i)

        # 计算每个聚类的均值和95%置信区间
        ci_data = []
        for cluster in sorted(rfm_df['Cluster'].unique()):
            cluster_vals = rfm_df[rfm_df['Cluster'] == cluster][metric].values
            mean = np.mean(cluster_vals)
            std = np.std(cluster_vals)
            n = len(cluster_vals)
            ci = 1.96 * std / np.sqrt(n)  # 95%置信区间

            ci_data.append({
                'cluster': cluster,
                'mean': mean,
                'ci_lower': mean - ci,
                'ci_upper': mean + ci
            })

        ci_df = pd.DataFrame(ci_data)

        # 绘制带置信区间的条形图
        x = ci_df['cluster'].astype(str)
        y = ci_df['mean']
        yerr = [ci_df['mean'] - ci_df['ci_lower'], ci_df['ci_upper'] - ci_df['mean']]

        plt.bar(x, y, yerr=yerr, capsize=5, color='skyblue', edgecolor='black')
        plt.title(f'{metric} by Cluster\n(95% Confidence Interval)')
        plt.xlabel('Cluster')
        plt.ylabel(metric)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

    return results


# 执行统计显著性检验
significance_results = cluster_significance_test(rfm.copy())