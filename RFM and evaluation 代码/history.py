import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
global df


rfm = pd.read_csv('optimized_rfm_cluster_results.csv')

df = pd.read_csv('data.csv', encoding='ISO-8859-1', low_memory=False)

df.columns = [c.strip() for c in df.columns]    # 去除列名空格
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], dayfirst=True, errors='coerce')

# 删除缺失关键字段的记录
df = df.dropna(subset=['CustomerID', 'InvoiceDate']).copy()

# 数值列转换
df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
df['UnitPrice'] = pd.to_numeric(df['UnitPrice'], errors='coerce')

# 过滤掉异常（非正数量或价格）
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)].copy()

# 总金额列
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
# 设置随机种子确保可重现性
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# 3.1 计算各聚类的历史留存率
def calculate_retention_rates(rfm_df, transaction_df, months_back=6):
    """
    计算各聚类在历史数据中的留存率
    """
    # 准备交易数据
    transaction_df = transaction_df.copy()
    transaction_df['Month'] = transaction_df['InvoiceDate'].dt.to_period('M')

    # 获取最近的月份
    latest_month = transaction_df['Month'].max()
    analysis_months = [latest_month - i for i in range(months_back, 0, -1)]
    # analysis_months = [m.to_period('M') for m in analysis_months]

    retention_results = []

    for cluster in sorted(rfm_df['Cluster'].unique()):
        # 获取该聚类的客户
        cluster_customers = rfm_df[rfm_df['Cluster'] == cluster]['CustomerID'].values

        # 计算连续月份的留存率
        print(f"\n聚类 {cluster} (客户数: {len(cluster_customers)}):")

        # 创建仅包含这些客户的交易数据
        cluster_transactions = transaction_df[transaction_df['CustomerID'].isin(cluster_customers)]

        # 计算每个客户首次购买月份
        first_purchase = cluster_transactions.groupby('CustomerID')['Month'].min().reset_index()
        first_purchase.columns = ['CustomerID', 'FirstPurchaseMonth']

        # 合并回交易数据
        cluster_transactions = cluster_transactions.merge(first_purchase, on='CustomerID')

        retention_rates = []
        for i, month in enumerate(analysis_months):
            if i == 0:
                continue  # 跳过第一个月

            prev_month = analysis_months[i - 1]

            # 上个月活跃的客户
            prev_active = cluster_transactions[cluster_transactions['Month'] == prev_month]['CustomerID'].unique()

            # 本月活跃的客户
            current_active = cluster_transactions[cluster_transactions['Month'] == month]['CustomerID'].unique()

            # 计算留存率
            if len(prev_active) > 0:
                retained = len(set(prev_active) & set(current_active))
                retention_rate = retained / len(prev_active)
                retention_rates.append({
                    'PreviousMonth': str(prev_month),
                    'CurrentMonth': str(month),
                    'RetentionRate': retention_rate,
                    'PrevActiveCount': len(prev_active),
                    'CurrentActiveCount': len(current_active)
                })

        # 转换为DataFrame
        retention_df = pd.DataFrame(retention_rates)
        if not retention_df.empty:
            avg_retention = retention_df['RetentionRate'].mean()
            print(f"  平均月留存率: {avg_retention:.2%}")
            retention_results.append({
                'Cluster': cluster,
                'AvgRetention': avg_retention,
                'Data': retention_df
            })

    # 3.2 可视化各聚类的留存率
    if retention_results:
        plt.figure(figsize=(10, 6))
        clusters = [r['Cluster'] for r in retention_results]
        avg_retentions = [r['AvgRetention'] for r in retention_results]

        bars = plt.bar(clusters, avg_retentions, color='green', edgecolor='black', alpha=0.7)
        plt.title('Comparison of Average Monthly Retention Rates Across Clusters')
        plt.xlabel('Cluster')
        plt.ylabel('Average Monthly Retention Rate')
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # 在柱子上显示数值
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                     f'{height:.1%}', ha='center', va='bottom')

        plt.show()

    return retention_results


# 执行留存率分析 (需要原始交易数据df)
retention_analysis = calculate_retention_rates(rfm.copy(), df.copy())