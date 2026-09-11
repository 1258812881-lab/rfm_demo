import pandas as pd

from BDA.cluster_stability import cluster_stability_test
from BDA.history import retention_analysis
from BDA.quality_assessment import quality_metrics
from BDA.static_evident import significance_results

file_name = 'optimized_rfm_cluster_results.csv'

rfm = pd.read_csv(file_name)

# 5.1 生成验证报告
def generate_validation_report(rfm_df, significance_results, retention_results, quality_metrics):
    """
    生成聚类标签合理性验证的综合报告
    """
    report = f"""
    # RFM聚类标签合理性验证报告
    生成日期: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

    ## 1. 聚类稳定性分析
    - 平均稳定性得分: {rfm_df['StabilityScore'].mean():.3f}
    - 稳定性得分>0.8的客户比例: {(rfm_df['StabilityScore'] > 0.8).mean():.1%}
    - 稳定性得分<0.5的客户比例: {(rfm_df['StabilityScore'] < 0.5).mean():.1%}

    ## 2. 聚类间差异显著性
    """

    for metric, result in significance_results.items():
        p_value = result['p_value']
        significance = ""
        if p_value < 0.001:
            significance = "极显著 (***p < 0.001)"
        elif p_value < 0.01:
            significance = "显著 (**p < 0.01)"
        elif p_value < 0.05:
            significance = "有一定显著性 (*p < 0.05)"
        else:
            significance = "不显著 (p >= 0.05)"

        report += f"- {metric}: p值 = {p_value:.6f}, {significance}\n"

    report += "\n## 3. 业务指标关联性\n"

    if retention_results:
        for result in retention_results:
            report += f"- 聚类 {result['Cluster']}: 平均月留存率 = {result['AvgRetention']:.1%}\n"
    else:
        report += "- 未计算留存率分析\n"

    report += "\n## 4. 聚类质量指标\n"
    report += f"- Silhouette Score: {quality_metrics['silhouette_score']:.4f}\n"
    report += f"- Calinski-Harabasz Score: {quality_metrics['calinski_harabasz_score']:.2f}\n"
    report += f"- Davies-Bouldin Index: {quality_metrics['davies_bouldin_index']:.4f}\n"
    report += f"- 簇内/簇间距离比例: {quality_metrics['distance_ratio']:.2f}\n"

    report += "\n## 5. 验证结论\n"

    # 根据指标给出结论
    stability_ok = rfm_df['StabilityScore'].mean() > 0.7
    significant_ok = all(r['p_value'] < 0.01 for r in significance_results.values())
    quality_ok = quality_metrics['silhouette_score'] > 0.35

    if stability_ok and significant_ok and quality_ok:
        conclusion = "✅ 聚类标签验证通过。聚类结果稳定，聚类间差异显著，且聚类质量良好。这些标签适合作为监督学习的训练目标。"
    elif stability_ok and significant_ok:
        conclusion = "⚠️ 聚类标签基本可靠。聚类结果稳定且聚类间差异显著，但聚类质量中等。建议在监督学习中使用对噪声鲁棒的算法。"
    else:
        conclusion = "❌ 聚类标签可靠性不足。建议重新进行聚类分析或考虑其他标签构建方法。"

    report += conclusion

    print(report)
    return report

rfm_df = cluster_stability_test(rfm)

# 生成验证报告
validation_report = generate_validation_report(
    rfm_df,
    significance_results,
    retention_analysis if 'retention_analysis' in locals() else None,
    quality_metrics
)