import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 1. 数据读取
CSV_FILENAME = "data.csv"
df = pd.read_csv(CSV_FILENAME, encoding='ISO-8859-1', low_memory=False)

# 2. 数据清洗
# 清理列名
df.columns = [c.strip() for c in df.columns]
# 转换日期格式
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], dayfirst=True, errors='coerce')
# 删除缺失值
df = df.dropna(subset=['CustomerID', 'InvoiceDate']).copy()
# 确保数值类型并过滤无效数据
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)].copy()
# 计算总价格
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# 3. RFM特征计算
reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)

# 计算RFM指标
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
    'InvoiceNo': 'nunique',  # Frequency
    'TotalPrice': 'sum'      # Monetary
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
rfm['Monetary'] = rfm['Monetary'].round(2)

# 4. 特征转换与标准化
# 对RFM进行对数变换处理偏态分布
rfm['Recency_log'] = np.log1p(rfm['Recency'])
rfm['Frequency_log'] = np.log1p(rfm['Frequency'])
rfm['Monetary_log'] = np.log1p(rfm['Monetary'])

# 标准化特征
features = ['Recency_log', 'Frequency_log', 'Monetary_log']
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[features])

import numpy as np  # 需导入numpy用于计算
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

# 5. 确定最佳聚类数 (Elbow & Silhouette)
sse = []
sil_scores = []
ks = range(2, 11)  # 测试2-10个聚类

for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(rfm_scaled)
    sse.append(km.inertia_)
    sil = silhouette_score(rfm_scaled, labels)
    sil_scores.append(sil)
    print(f"k={k} SSE={km.inertia_:.1f} Silhouette={sil:.4f}")

# 可视化Elbow和Silhouette结果
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(ks, sse, 'bo-')
plt.title('Elbow Method (SSE)')
plt.xlabel('cluster number (k)')
plt.ylabel('SSE')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(ks, sil_scores, 'go-')
plt.title('Silhouette Score vs k')
plt.xlabel('cluster number (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)

plt.tight_layout()
plt.show()

# 6. 聚类
K_FINAL = 4

km_final = KMeans(n_clusters=K_FINAL, random_state=42, n_init=20)
km_final =  AgglomerativeClustering(n_clusters=K_FINAL, linkage='ward')

rfm['Cluster'] = km_final.fit_predict(rfm_scaled)

# 7. 聚类结果可视化
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.boxplot([rfm[rfm['Cluster'] == i]['Recency'] for i in range(K_FINAL)],
            labels=[f"cluster {i}" for i in range(K_FINAL)])
plt.title('Recency')
plt.ylabel('Day')
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 3, 2)
plt.boxplot([rfm[rfm['Cluster'] == i]['Frequency'] for i in range(K_FINAL)],
            labels=[f"cluster {i}" for i in range(K_FINAL)])
plt.title('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)

plt.subplot(1, 3, 3)
plt.boxplot([rfm[rfm['Cluster'] == i]['Monetary'] for i in range(K_FINAL)],
            labels=[f"cluster {i}" for i in range(K_FINAL)])
plt.title('Monetary')
plt.grid(True, linestyle='--', alpha=0.7)

plt.suptitle('Distribution of RFM Features Across Clusters', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# 8. 聚类特征分析
profile = rfm.groupby('Cluster').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': ['mean', 'count']
}).round(2)

# 重命名列
profile.columns = ['Recency_mean', 'Frequency_mean', 'Monetary_mean', 'CustomerCount']
profile = profile.reset_index().sort_values('Cluster')
print("聚类特征分析:")
print(profile)

# 9. 为聚类分配业务含义
cluster_names = {
    0: "高价值忠诚客户",
    1: "潜力客户",
    2: "一次性买家",
    3: "不活跃客户"
}
rfm['SegmentName'] = rfm['Cluster'].map(cluster_names)

# 显示各聚类的代表性客户
print("\n各聚类的代表性客户 (按消费金额排序):")
for cluster_id in range(K_FINAL):
    cluster_data = rfm[rfm['Cluster'] == cluster_id].sort_values('Monetary', ascending=False).head(3)
    print(f"\n{cluster_names[cluster_id]} (簇 {cluster_id}):")
    print(cluster_data[['CustomerID', 'Recency', 'Frequency', 'Monetary']])

# 保存到新文件（文件名可自定义）
output_filename = "rfm_cluster_labels.csv"
rfm.to_csv(output_filename, index=False)

# ============== 修复后的高级可视化图表 (所有图表使用英文) ==============
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from math import pi
from sklearn.metrics import silhouette_samples

# 设置专业美观的绘图风格
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams['font.family'] = 'DejaVu Sans'  # 确保英文显示正常

# 1. 3D聚类分布图 (使用对数转换后的特征)
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# 为每个聚类分配独特颜色
colors = cm.tab20(np.linspace(0, 1, K_FINAL))

for i in range(K_FINAL):
    cluster_data = rfm[rfm['Cluster'] == i]
    ax.scatter(
        cluster_data['Recency_log'],
        cluster_data['Frequency_log'],
        cluster_data['Monetary_log'],
        s=40, alpha=0.8,
        c=[colors[i]],
        label=f'Cluster {i}',
        edgecolor='w', linewidth=0.5
    )

# 计算聚类中心 (直接使用标准化数据)
cluster_centers = np.zeros((K_FINAL, 3))
for i in range(K_FINAL):
    cluster_samples = rfm_scaled[rfm['Cluster'] == i]
    if len(cluster_samples) > 0:  # 确保簇不为空
        cluster_centers[i] = cluster_samples.mean(axis=0)

# 将中心点转换回原始对数尺度
recency_center = cluster_centers[:, 0]
frequency_center = cluster_centers[:, 1]
monetary_center = cluster_centers[:, 2]

ax.scatter(
    recency_center,
    frequency_center,
    monetary_center,
    s=300, c='gold', marker='*',
    edgecolor='k', linewidth=1.5,
    label='Cluster Centers'
)

# 设置3D图表属性
ax.set_xlabel('Recency (log)', labelpad=10, fontsize=12)
ax.set_ylabel('Frequency (log)', labelpad=10, fontsize=12)
ax.set_zlabel('Monetary (log)', labelpad=10, fontsize=12)
ax.set_title('3D Cluster Distribution in RFM Space', fontsize=16, pad=20)
ax.legend(loc='upper left', bbox_to_anchor=(0.05, 0.95), frameon=True, shadow=True)
ax.grid(True, linestyle='--', alpha=0.7)
ax.view_init(elev=20, azim=35)
plt.tight_layout()
plt.show()

# 2. 聚类特征雷达图 (标准化后特征) - 修复版本
# 仅选择数值列进行分组聚合
cluster_means = rfm.groupby('Cluster')[features].mean().reset_index()

# 设置雷达图参数
categories = features
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]  # 闭合图形

fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))

for i, row in cluster_means.iterrows():
    values = row[categories].values.flatten().tolist()
    values += values[:1]  # 闭合数据
    ax.plot(angles, values, 'o-', linewidth=2, label=f'Cluster {int(row["Cluster"])}')
    ax.fill(angles, values, alpha=0.1)

# 添加雷达图装饰元素
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)
plt.xticks(angles[:-1], [cat.replace("_log", "") for cat in categories], size=12)
ax.tick_params(axis='x', pad=20)

# 设置y轴范围
all_values = cluster_means[categories].values.flatten()
y_min, y_max = all_values.min(), all_values.max()
y_range = y_max - y_min
ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

plt.title('Cluster Profile Comparison (Standardized Features)', size=16, pad=30)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), frameon=True, shadow=True, title="Clusters")
plt.tight_layout()
plt.show()

# 3. 轮廓系数分布图
cluster_labels = rfm['Cluster'].values
sample_silhouette_values = silhouette_samples(rfm_scaled, cluster_labels)

fig, ax = plt.subplots(figsize=(12, 8))
y_lower = 10

for i in range(K_FINAL):
    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i

    color = cm.tab20(float(i) / K_FINAL)
    ax.fill_betweenx(np.arange(y_lower, y_upper),
                     0, ith_cluster_silhouette_values,
                     facecolor=color, edgecolor=color, alpha=0.8, label=f'Cluster {i}')

    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, f'Cluster {i}', fontsize=12, fontweight='bold')
    y_lower = y_upper + 10

# 标记平均轮廓系数
avg_silhouette = silhouette_score(rfm_scaled, cluster_labels)
ax.axvline(x=avg_silhouette, color="red", linestyle="--", linewidth=2.5,
           label=f'Average Silhouette Score: {avg_silhouette:.3f}')

ax.set_title('Silhouette Plot for Clusters', fontsize=16)
ax.set_xlabel('Silhouette Coefficient Values', fontsize=12)
ax.set_ylabel('Cluster Labels', fontsize=12)
ax.set_yticks([])
ax.set_xlim([-0.1, 1])
ax.set_facecolor('#f8f9fa')
ax.grid(color='white', linestyle='-', linewidth=1.5)
ax.legend(loc='lower right', frameon=True, shadow=True, fontsize=11)
plt.tight_layout()
plt.show()

# 4. 散点图矩阵 (Pair Plot)
# 仅使用原始RFM特征，避免包含SegmentName列
plot_df = rfm[['Recency', 'Frequency', 'Monetary', 'Cluster']].copy()
plot_df['Cluster'] = plot_df['Cluster'].astype(str)

# 创建自定义调色板
cluster_palette = sns.color_palette("tab20", K_FINAL)
cluster_colors = dict(zip(plot_df['Cluster'].unique(), cluster_palette))

g = sns.PairGrid(plot_df, hue='Cluster', palette=cluster_colors, diag_sharey=False,
                 height=3.5, aspect=1.1)

g.map_diag(sns.kdeplot, fill=True, alpha=0.6, linewidth=1.5, common_norm=False)
g.map_offdiag(sns.scatterplot, alpha=0.6, s=40, edgecolor='w', linewidth=0.5)

plt.subplots_adjust(top=0.95)
g.fig.suptitle('Pairwise Relationships Between RFM Features by Cluster', fontsize=18)

handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cluster_colors[label],
                      markersize=10, label=f'Cluster {label}') for label in cluster_colors]
g.fig.legend(handles=handles, loc='upper right', bbox_to_anchor=(0.98, 0.98),
             title='Clusters', frameon=True, shadow=True, title_fontsize=14)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# 5. 聚类中心热力图 (原始RFM值) - 修复版本
# 仅对数值列进行分组聚合
cluster_means_original = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(2)

# 创建热力图数据
heatmap_data = cluster_means_original.copy()

plt.figure(figsize=(14, 8))
ax = sns.heatmap(
    heatmap_data,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    linewidths=0.5,
    annot_kws={'size': 14, 'weight': 'bold'},
    cbar_kws={'label': 'Value', 'shrink': 0.6},
    square=True
)

# 标记最大值和最小值
for i, cluster in enumerate(heatmap_data.index):
    row = heatmap_data.loc[cluster]
    max_col = row.idxmax()
    min_col = row.idxmin()

    # 标记最大值
    max_idx = list(heatmap_data.columns).index(max_col)
    ax.add_patch(plt.Rectangle((max_idx, i), 1, 1, fill=False, edgecolor='gold', lw=3, clip_on=False))

    # 标记最小值
    min_idx = list(heatmap_data.columns).index(min_col)
    ax.add_patch(plt.Circle((min_idx + 0.5, i + 0.5), 0.35, fill=False, edgecolor='lime', lw=3, clip_on=False))

plt.title('Cluster Centroids in Original RFM Scale', fontsize=18, pad=20)
plt.xlabel('RFM Features', fontsize=14, labelpad=15)
plt.ylabel('Cluster ID', fontsize=14, labelpad=15)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)

from matplotlib.lines import Line2D

legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='none', markeredgecolor='gold',
           markersize=15, markeredgewidth=3, label='Maximum in Row'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none', markeredgecolor='lime',
           markersize=15, markeredgewidth=3, label='Minimum in Row')
]
ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1.05),
          frameon=True, shadow=True, title='Value Significance', title_fontsize=12)

plt.tight_layout()
plt.savefig('cluster_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# [原有代码结束]