"""
结果可视化模块：生成分析结果的图表
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import config

# 设置中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except:
    print("警告: 无法设置中文字体，图表中的中文可能无法正确显示")

def plot_common_targets_network(common_targets, baicalin_relations, tetrandrine_relations):
    """
    绘制共同靶点的网络图
    
    参数:
    common_targets - 共同靶点数据
    baicalin_relations - 黄芩苷关系数据
    tetrandrine_relations - 汉防己甲素关系数据
    """
    # 创建一个有向图
    G = nx.DiGraph()
    
    # 添加药物节点
    G.add_node("黄芩苷", type="drug")
    G.add_node("汉防己甲素", type="drug")
    
    # 添加共同靶点节点和边
    for _, target in common_targets.iterrows():
        target_id = target['Target_ID']
        target_name = target['Target_Name']
        
        # 添加靶点节点
        G.add_node(target_name, type="target")
        
        # 添加黄芩苷到靶点的边
        baicalin_to_target = baicalin_relations[
            (baicalin_relations['Source_Name'] == '黄芩苷') & 
            (baicalin_relations['Target_ID'] == target_id)
        ]
        
        if not baicalin_to_target.empty:
            relation_type = baicalin_to_target['Type'].iloc[0]
            probability = baicalin_to_target['Probability'].iloc[0]
            
            # 根据关系类型确定边的属性
            if relation_type == 'Positive_Correlation':
                edge_color = 'green'
                edge_style = 'solid'
            elif relation_type == 'Negative_Correlation':
                edge_color = 'red'
                edge_style = 'solid'
            else:
                edge_color = 'gray'
                edge_style = 'dashed'
            
            G.add_edge("黄芩苷", target_name, color=edge_color, style=edge_style, weight=probability*3)
        
        # 添加汉防己甲素到靶点的边
        tetrandrine_to_target = tetrandrine_relations[
            (tetrandrine_relations['Source_Name'] == '汉防己甲素') & 
            (tetrandrine_relations['Target_ID'] == target_id)
        ]
        
        if not tetrandrine_to_target.empty:
            relation_type = tetrandrine_to_target['Type'].iloc[0]
            probability = tetrandrine_to_target['Probability'].iloc[0]
            
            # 根据关系类型确定边的属性
            if relation_type == 'Positive_Correlation':
                edge_color = 'green'
                edge_style = 'solid'
            elif relation_type == 'Negative_Correlation':
                edge_color = 'red'
                edge_style = 'solid'
            else:
                edge_color = 'gray'
                edge_style = 'dashed'
            
            G.add_edge("汉防己甲素", target_name, color=edge_color, style=edge_style, weight=probability*3)
    
    # 创建图形
    plt.figure(figsize=(16, 12))
    
    # 设置布局
    pos = nx.spring_layout(G, seed=42)
    
    # 绘制节点
    drug_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'drug']
    target_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'target']
    
    nx.draw_networkx_nodes(G, pos, nodelist=drug_nodes, node_color='lightblue', node_size=800, alpha=0.8)
    nx.draw_networkx_nodes(G, pos, nodelist=target_nodes, node_color='lightgreen', node_size=500, alpha=0.8)
    
    # 绘制边
    edges = G.edges(data=True)
    edge_colors = [attr['color'] for _, _, attr in edges]
    edge_styles = [attr['style'] for _, _, attr in edges]
    edge_widths = [attr['weight'] for _, _, attr in edges]
    
    # 分别绘制实线和虚线边
    solid_edges = [(u, v) for u, v, attr in edges if attr['style'] == 'solid']
    dashed_edges = [(u, v) for u, v, attr in edges if attr['style'] == 'dashed']
    
    solid_colors = [attr['color'] for u, v, attr in edges if attr['style'] == 'solid']
    dashed_colors = [attr['color'] for u, v, attr in edges if attr['style'] == 'dashed']
    
    solid_widths = [attr['weight'] for u, v, attr in edges if attr['style'] == 'solid']
    dashed_widths = [attr['weight'] for u, v, attr in edges if attr['style'] == 'dashed']
    
    nx.draw_networkx_edges(G, pos, edgelist=solid_edges, edge_color=solid_colors, width=solid_widths, alpha=0.7)
    nx.draw_networkx_edges(G, pos, edgelist=dashed_edges, edge_color=dashed_colors, width=dashed_widths, style='dashed', alpha=0.5)
    
    # 绘制标签
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    # 添加图例
    plt.title('黄芩苷与汉防己甲素的共同靶点网络图', fontsize=15)
    plt.tight_layout()
    
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', markersize=15, label='药物'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', markersize=15, label='靶点'),
        Line2D([0], [0], color='green', lw=2, label='正相关'),
        Line2D([0], [0], color='red', lw=2, label='负相关'),
        Line2D([0], [0], color='gray', lw=2, linestyle='dashed', label='关联关系')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # 保存图像
    os.makedirs(os.path.join(config.RESULTS_DIR, 'figures'), exist_ok=True)
    plt.savefig(os.path.join(config.RESULTS_DIR, 'figures', 'common_targets_network.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_pathway_enrichment(pathway_results):
    """
    绘制通路富集分析热图
    
    参数:
    pathway_results - 通路富集分析结果
    """
    # 提取通路参与度数据
    pathway_involvement = pathway_results['pathway_involvement']
    pathway_enrichment = pathway_results['pathway_enrichment']
    
    # 1. 绘制靶点-通路热图
    plt.figure(figsize=(14, 10))
    
    # 准备热图数据
    heatmap_data = pathway_involvement[['Target_Name', 'Inflammatory_Score', 'Oxidative_Stress_Score', 'Apoptosis_Score', 
                                        'Fibrosis_Score', 'Liver_Metabolism_Score', 'Kidney_Function_Score']]
    
    # 重命名列以便显示
    heatmap_data = heatmap_data.rename(columns={
        'Inflammatory_Score': '炎症通路',
        'Oxidative_Stress_Score': '氧化应激通路',
        'Apoptosis_Score': '细胞凋亡通路',
        'Fibrosis_Score': '纤维化通路',
        'Liver_Metabolism_Score': '肝脏代谢通路',
        'Kidney_Function_Score': '肾脏功能通路'
    })
    
    # 设置靶点名称为索引
    heatmap_data = heatmap_data.set_index('Target_Name')
    
    # 创建热图
    sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='.2f', linewidths=.5)
    plt.title('共同靶点的通路参与度热图', fontsize=15)
    plt.ylabel('靶点')
    plt.xlabel('生物学通路')
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(config.RESULTS_DIR, 'figures', 'pathway_involvement_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 绘制通路富集柱状图
    plt.figure(figsize=(12, 8))
    
    # 准备数据
    pathways = ['炎症通路', '氧化应激通路', '细胞凋亡通路', '纤维化通路', '肝脏代谢通路', '肾脏功能通路']
    enrichment_values = [
        pathway_enrichment['Inflammatory_Pathway'],
        pathway_enrichment['Oxidative_Stress_Pathway'],
        pathway_enrichment['Apoptosis_Pathway'],
        pathway_enrichment['Fibrosis_Pathway'],
        pathway_enrichment['Liver_Metabolism_Pathway'],
        pathway_enrichment['Kidney_Function_Pathway']
    ]
    
    # 绘制柱状图
    bars = plt.bar(pathways, enrichment_values, color=sns.color_palette("YlGnBu", 6))
    
    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=10)
    
    plt.title('生物学通路富集程度', fontsize=15)
    plt.ylabel('富集分数')
    plt.ylim(0, max(enrichment_values) * 1.2)
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(os.path.join(config.RESULTS_DIR, 'figures', 'pathway_enrichment_barplot.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_synergy_mechanisms(synergy_results):
    """
    绘制协同机制分析结果
    修复了维度不匹配问题，确保类别和值数组长度相同
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # 提取数据
    mechanism_types = synergy_results['mechanism_types']
    
    # 提取共同靶点数量（如果可用）
    common_targets_count = 0
    if 'therapeutic_targets' in synergy_results:
        common_targets_count = len(synergy_results.get('therapeutic_targets', []))
    
    # 创建协同系数雷达图
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
    
    # 修改：确保类别与值数组长度相同
    categories = [
        'Therapeutic Effect', 
        'Hepatoprotective Effect', 
        'Nephroprotective Effect', 
        'Target Count'  # 第四个类别
    ]
    
    # 确保值数组与类别数组长度匹配
    values = [
        mechanism_types['silicosis_treatment']['synergy_coefficient'],
        mechanism_types['hepatotoxicity_reduction']['synergy_coefficient'],
        mechanism_types['nephrotoxicity_reduction']['synergy_coefficient'],
        min(1.0, common_targets_count / 100)  # 归一化靶点数量
    ]
    
    # 调试输出
    print(f"雷达图类别: {categories}")
    print(f"雷达图值: {values}")
    
    # 确保类别和值的长度相同
    assert len(categories) == len(values), "类别和值必须有相同的长度"
    
    # 类别数量
    N = len(categories)
    
    # 计算每个类别的角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    
    # 闭合角度列表
    angles += angles[:1]
    
    # 同样闭合值列表
    values_closed = values + values[:1]
    
    # 绘制数据
    ax.fill(angles, values_closed, color='skyblue', alpha=0.5)
    ax.plot(angles, values_closed, color='blue', linewidth=2)
    
    # 添加类别标签
    plt.xticks(angles[:-1], categories)
    
    # 添加径向网格线
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
    ax.set_rlabel_position(0)
    
    # 添加标题
    plt.title('Synergy Mechanism Analysis', size=14, fontweight='bold')
    
    # 保存图形
    save_dir = os.path.join('results', 'figures')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'synergy_mechanism_radar.png'), dpi=300, bbox_inches='tight')
    
    # 创建靶点计数条形图
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 靶点计数
    target_types = ['Therapeutic', 'Hepatoprotective', 'Nephroprotective']
    target_counts = [
        len(synergy_results.get('therapeutic_targets', [])),
        len(synergy_results.get('hepatoprotective_targets', [])),
        len(synergy_results.get('nephroprotective_targets', []))
    ]
    
    # 创建条形
    bars = ax.bar(target_types, target_counts, color=['green', 'blue', 'purple'])
    
    # 在条形上方添加计数标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(height)}', ha='center', va='bottom')
    
    # 添加标签和标题
    ax.set_xlabel('Target Type', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Common Target Counts by Function', fontsize=14, fontweight='bold')
    
    # 添加网格以提高可读性
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 保存图形
    plt.savefig(os.path.join(save_dir, 'target_counts.png'), dpi=300, bbox_inches='tight')
    
    # 机制类型
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 机制数据
    mechanism_names = ['Silicosis Treatment', 'Hepatotoxicity Reduction', 'Nephrotoxicity Reduction']
    mechanism_coefficients = [
        mechanism_types['silicosis_treatment']['synergy_coefficient'],
        mechanism_types['hepatotoxicity_reduction']['synergy_coefficient'],
        mechanism_types['nephrotoxicity_reduction']['synergy_coefficient']
    ]
    
    # 根据系数值设置颜色
    colors = []
    for coef in mechanism_coefficients:
        if coef > 0.6:
            colors.append('darkgreen')
        elif coef > 0.3:
            colors.append('green')
        elif coef > 0.05:
            colors.append('lightgreen')
        else:
            colors.append('gray')
    
    # 创建条形
    bars = ax.bar(mechanism_names, mechanism_coefficients, color=colors)
    
    # 在条形上方添加系数标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 添加标签和标题
    ax.set_xlabel('Mechanism Type', fontsize=12)
    ax.set_ylabel('Synergy Coefficient', fontsize=12)
    ax.set_title('Synergy Coefficients by Mechanism Type', fontsize=14, fontweight='bold')
    
    # 添加网格以提高可读性
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 保存图形
    plt.savefig(os.path.join(save_dir, 'synergy_coefficients.png'), dpi=300, bbox_inches='tight')
    
    plt.close('all')
def plot_key_synergy_genes(synergy_results):
    """
    绘制关键协同增效减毒基因可视化图
    
    参数:
    synergy_results - 协同机制分析结果
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    # 提取数据
    therapeutic_genes = synergy_results.get('therapeutic_genes', [])
    hepatoprotective_genes = synergy_results.get('hepatoprotective_genes', [])
    nephroprotective_genes = synergy_results.get('nephroprotective_genes', [])
    
    # 创建保存目录
    save_dir = os.path.join('results', 'figures')
    os.makedirs(save_dir, exist_ok=True)
    
    # 检查是否有数据可绘制
    if not therapeutic_genes and not hepatoprotective_genes and not nephroprotective_genes:
        print("没有足够的基因数据用于可视化")
        return
    
    # 绘制治疗效应关键基因
    if therapeutic_genes:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 提取数据
        genes = [gene['gene_name'] for gene in therapeutic_genes]
        scores = [gene['importance_score'] for gene in therapeutic_genes]
        
        # 创建条形图
        bars = ax.barh(genes, scores, color='green', alpha=0.7)
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', va='center')
        
        # 添加标签和标题
        ax.set_xlabel('重要性评分', fontsize=12)
        ax.set_ylabel('基因名称', fontsize=12)
        ax.set_title('硅肺病治疗关键协同基因', fontsize=14, fontweight='bold')
        
        # 添加网格
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        # 保存图形
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'therapeutic_genes.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 绘制肝保护关键基因
    if hepatoprotective_genes:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 提取数据
        genes = [gene['gene_name'] for gene in hepatoprotective_genes]
        scores = [gene['importance_score'] for gene in hepatoprotective_genes]
        
        # 创建条形图
        bars = ax.barh(genes, scores, color='blue', alpha=0.7)
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', va='center')
        
        # 添加标签和标题
        ax.set_xlabel('重要性评分', fontsize=12)
        ax.set_ylabel('基因名称', fontsize=12)
        ax.set_title('肝保护关键协同基因', fontsize=14, fontweight='bold')
        
        # 添加网格
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        # 保存图形
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'hepatoprotective_genes.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 绘制肾保护关键基因
    if nephroprotective_genes:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 提取数据
        genes = [gene['gene_name'] for gene in nephroprotective_genes]
        scores = [gene['importance_score'] for gene in nephroprotective_genes]
        
        # 创建条形图
        bars = ax.barh(genes, scores, color='purple', alpha=0.7)
        
        # 添加数值标签
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', va='center')
        
        # 添加标签和标题
        ax.set_xlabel('重要性评分', fontsize=12)
        ax.set_ylabel('基因名称', fontsize=12)
        ax.set_title('肾保护关键协同基因', fontsize=14, fontweight='bold')
        
        # 添加网格
        ax.grid(axis='x', linestyle='--', alpha=0.6)
        
        # 保存图形
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'nephroprotective_genes.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 绘制三种类型基因的对比图
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # 数据准备
    gene_types = ['治疗效应', '肝保护效应', '肾保护效应']
    gene_counts = [
        len(therapeutic_genes),
        len(hepatoprotective_genes),
        len(nephroprotective_genes)
    ]
    
    # 创建条形图
    bars = ax.bar(gene_types, gene_counts, color=['green', 'blue', 'purple'], alpha=0.7)
    
    # 添加数量标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                f'{int(height)}', ha='center', va='bottom')
    
    # 添加标签和标题
    ax.set_xlabel('作用类型', fontsize=14)
    ax.set_ylabel('关键基因数量', fontsize=14)
    ax.set_title('不同作用类型的关键协同基因数量', fontsize=16, fontweight='bold')
    
    # 添加网格
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    # 保存图形
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'gene_type_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()