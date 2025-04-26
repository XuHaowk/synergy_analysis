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

def plot_pathway_enrichment(enrichment_results, output_dir='./results/figures'):
    """
    可视化通路富集分析结果
    
    参数:
    enrichment_results - 富集分析结果
    output_dir - 输出目录
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 从结果中获取数据框
    if isinstance(enrichment_results, dict):
        enrichment_df = enrichment_results.get('enrichment_results', None)
        gene_pathway_df = enrichment_results.get('gene_pathway_mapping', None)
    else:
        enrichment_df = enrichment_results
        gene_pathway_df = None
    
    if enrichment_df is None or enrichment_df.empty:
        print("错误: 没有富集结果可供可视化")
        return
    
    print(f"正在可视化 {len(enrichment_df)} 个富集通路的结果...")
    
    # 1. 绘制富集气泡图
    plt.figure(figsize=(12, 8))
    
    # 只显示前10个通路
    if len(enrichment_df) > 10:
        plot_df = enrichment_df.head(10)
    else:
        plot_df = enrichment_df
    
    # 转换P值为负对数
    plot_df['NegLogP'] = -np.log10(plot_df['P_Value'])
    
    # 绘制气泡图
    sns.scatterplot(
        x='Fold_Enrichment',
        y='Pathway',
        size='Hits',
        hue='NegLogP',
        sizes=(50, 400),
        palette='viridis',
        data=plot_df
    )
    
    plt.title('Pathway Enrichment Analysis', fontsize=16)
    plt.xlabel('Fold Enrichment', fontsize=14)
    plt.ylabel('Pathway', fontsize=14)
    plt.legend(title='-log10(P-value)', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pathway_enrichment_bubble.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. 绘制热图 (如果有基因-通路映射)
    if gene_pathway_df is not None and not gene_pathway_df.empty:
        # 创建一个透视表：基因 × 通路
        try:
            pivot_df = gene_pathway_df.pivot_table(
                index='Gene', 
                columns='Pathway', 
                values='Score',
                fill_value=0
            )
            
            # 如果透视表太大，只选择前20个基因和前8个通路
            if pivot_df.shape[0] > 20:
                # 选择出现在最多通路中的前20个基因
                gene_counts = gene_pathway_df['Gene'].value_counts().head(20)
                top_genes = gene_counts.index.tolist()
                pivot_df = pivot_df.loc[top_genes]
            
            if pivot_df.shape[1] > 8:
                # 选择最富集的前8个通路
                top_pathways = enrichment_df['Pathway'].head(8).tolist()
                pivot_df = pivot_df[top_pathways]
            
            # 设置图形大小
            plt.figure(figsize=(12, 10))
            
            # 创建自定义颜色映射
            colors = ['#ffffff', '#ffffcc', '#ffeda0', '#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026', '#800026']
            cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
            
            # 绘制热图
            sns.heatmap(
                pivot_df,
                cmap=cmap,
                linewidths=0.5,
                linecolor='gray',
                square=True,
                cbar_kws={'label': 'Association Score'}
            )
            
            plt.title('Gene-Pathway Association Heatmap', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'gene_pathway_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"绘制热图时出错: {e}")
    
    # 3. 绘制条形图
    plt.figure(figsize=(10, 8))
    
    # 按富集倍数排序
    plot_df = plot_df.sort_values('Fold_Enrichment')
    
    # 绘制条形图
    bars = plt.barh(plot_df['Pathway'], plot_df['Fold_Enrichment'], color='skyblue')
    
    # 在条形上标注P值
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height()/2,
            f"p={plot_df.iloc[i]['P_Value']:.1e}",
            va='center',
            fontsize=9
        )
    
    plt.title('Pathway Enrichment Analysis', fontsize=16)
    plt.xlabel('Fold Enrichment', fontsize=14)
    plt.ylabel('Pathway', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pathway_enrichment_bars.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"通路富集可视化完成，图像已保存至: {output_dir}")

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
    import traceback
    
    print("\n===== 开始绘制关键协同基因图表 =====")
    
    # 解决字体问题
    try:
        # 尝试多种可能的中文字体
        font_found = False
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'AR PL UMing CN', 'NSimSun', 'STSong']
        
        for font in chinese_fonts:
            try:
                plt.rcParams['font.sans-serif'] = [font]  # 用来正常显示中文标签
                plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
                # 测试是否成功设置字体
                fig, ax = plt.subplots(figsize=(1, 1))
                ax.set_title('测试')
                plt.close(fig)
                print(f"成功设置中文字体: {font}")
                font_found = True
                break
            except Exception as e:
                print(f"尝试设置字体{font}失败: {e}")
                continue
        
        if not font_found:
            # 如果所有中文字体都失败，使用系统默认字体
            print("无法找到中文字体，使用系统默认字体")
            plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    except Exception as e:
        print(f"设置字体时出错: {e}")
        # 使用安全的默认设置
        plt.rcParams.update({'font.sans-serif': ['DejaVu Sans', 'Arial']})
    
    # 提取数据
    try:
        print("提取协同分析结果数据...")
        therapeutic_genes = synergy_results.get('therapeutic_genes', [])
        hepatoprotective_genes = synergy_results.get('hepatoprotective_genes', [])
        nephroprotective_genes = synergy_results.get('nephroprotective_genes', [])
        
        print(f"- 治疗基因: {len(therapeutic_genes)}个")
        print(f"- 肝保护基因: {len(hepatoprotective_genes)}个")
        print(f"- 肾保护基因: {len(nephroprotective_genes)}个")
    except Exception as e:
        print(f"提取数据时出错: {e}")
        traceback.print_exc()
        return
    
    # 创建保存目录
    save_dir = os.path.join('results', 'figures')
    os.makedirs(save_dir, exist_ok=True)
    print(f"图表将保存到: {save_dir}")
    
    # 检查是否有数据可绘制
    if not therapeutic_genes and not hepatoprotective_genes and not nephroprotective_genes:
        print("没有足够的基因数据用于可视化")
        return
    
    try:
        # 绘制治疗效应关键基因
        if therapeutic_genes:
            print("绘制治疗效应关键基因图...")
            
            try:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # 提取数据
                genes = [gene.get('gene_name', f"未知基因_{i}") for i, gene in enumerate(therapeutic_genes)]
                scores = [gene.get('importance_score', 0) for gene in therapeutic_genes]
                
                print(f"- 基因名称: {genes}")
                print(f"- 重要性评分: {scores}")
                
                # 如果没有有效数据，跳过绘图
                if not genes or not scores or len(genes) != len(scores):
                    print(f"警告: 治疗基因数据不完整，跳过绘图")
                    plt.close(fig)
                else:
                    # 创建条形图
                    bars = ax.barh(genes, scores, color='green', alpha=0.7)
                    
                    # 添加数值标签
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                                f'{width:.3f}', va='center')
                    
                    # 添加标签和标题
                    ax.set_xlabel('Importance Score', fontsize=12)
                    ax.set_ylabel('Gene Name', fontsize=12)
                    ax.set_title('Key Genes for Silicosis Treatment', fontsize=14, fontweight='bold')
                    
                    # 添加网格
                    ax.grid(axis='x', linestyle='--', alpha=0.6)
                    
                    # 保存图形
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, 'therapeutic_genes.png'), dpi=300, bbox_inches='tight')
                    print(f"治疗效应关键基因图保存到: {os.path.join(save_dir, 'therapeutic_genes.png')}")
                    plt.close()
            except Exception as e:
                print(f"绘制治疗效应图时出错: {e}")
                traceback.print_exc()
                try:
                    plt.close()
                except:
                    pass
        
        # 绘制肝保护关键基因
        if hepatoprotective_genes:
            print("绘制肝保护关键基因图...")
            
            try:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # 提取数据
                genes = [gene.get('gene_name', f"未知基因_{i}") for i, gene in enumerate(hepatoprotective_genes)]
                scores = [gene.get('importance_score', 0) for gene in hepatoprotective_genes]
                
                # 如果没有有效数据，跳过绘图
                if not genes or not scores or len(genes) != len(scores):
                    print(f"警告: 肝保护基因数据不完整，跳过绘图")
                    plt.close(fig)
                else:
                    # 创建条形图
                    bars = ax.barh(genes, scores, color='blue', alpha=0.7)
                    
                    # 添加数值标签
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                                f'{width:.3f}', va='center')
                    
                    # 添加标签和标题
                    ax.set_xlabel('Importance Score', fontsize=12)
                    ax.set_ylabel('Gene Name', fontsize=12)
                    ax.set_title('Key Genes for Hepatoprotection', fontsize=14, fontweight='bold')
                    
                    # 添加网格
                    ax.grid(axis='x', linestyle='--', alpha=0.6)
                    
                    # 保存图形
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, 'hepatoprotective_genes.png'), dpi=300, bbox_inches='tight')
                    print(f"肝保护关键基因图保存到: {os.path.join(save_dir, 'hepatoprotective_genes.png')}")
                    plt.close()
            except Exception as e:
                print(f"绘制肝保护图时出错: {e}")
                traceback.print_exc()
                try:
                    plt.close()
                except:
                    pass
        
        # 绘制肾保护关键基因
        if nephroprotective_genes:
            print("绘制肾保护关键基因图...")
            
            try:
                fig, ax = plt.subplots(figsize=(12, 8))
                
                # 提取数据
                genes = [gene.get('gene_name', f"未知基因_{i}") for i, gene in enumerate(nephroprotective_genes)]
                scores = [gene.get('importance_score', 0) for gene in nephroprotective_genes]
                
                # 如果没有有效数据，跳过绘图
                if not genes or not scores or len(genes) != len(scores):
                    print(f"警告: 肾保护基因数据不完整，跳过绘图")
                    plt.close(fig)
                else:
                    # 创建条形图
                    bars = ax.barh(genes, scores, color='purple', alpha=0.7)
                    
                    # 添加数值标签
                    for bar in bars:
                        width = bar.get_width()
                        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                                f'{width:.3f}', va='center')
                    
                    # 添加标签和标题
                    ax.set_xlabel('Importance Score', fontsize=12)
                    ax.set_ylabel('Gene Name', fontsize=12)
                    ax.set_title('Key Genes for Nephroprotection', fontsize=14, fontweight='bold')
                    
                    # 添加网格
                    ax.grid(axis='x', linestyle='--', alpha=0.6)
                    
                    # 保存图形
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, 'nephroprotective_genes.png'), dpi=300, bbox_inches='tight')
                    print(f"肾保护关键基因图保存到: {os.path.join(save_dir, 'nephroprotective_genes.png')}")
                    plt.close()
            except Exception as e:
                print(f"绘制肾保护图时出错: {e}")
                traceback.print_exc()
                try:
                    plt.close()
                except:
                    pass
        
        # 绘制三种类型基因的对比图
        print("绘制基因类型对比图...")
        
        try:
            fig, ax = plt.subplots(figsize=(15, 10))
            
            # 数据准备
            gene_types = ['Therapeutic', 'Hepatoprotective', 'Nephroprotective']
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
            ax.set_xlabel('Effect Type', fontsize=14)
            ax.set_ylabel('Key Gene Count', fontsize=14)
            ax.set_title('Key Synergistic Gene Counts by Effect Type', fontsize=16, fontweight='bold')
            
            # 添加网格
            ax.grid(axis='y', linestyle='--', alpha=0.6)
            
            # 保存图形
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'gene_type_comparison.png'), dpi=300, bbox_inches='tight')
            print(f"基因类型对比图保存到: {os.path.join(save_dir, 'gene_type_comparison.png')}")
            plt.close()
        except Exception as e:
            print(f"绘制类型对比图时出错: {e}")
            traceback.print_exc()
            try:
                plt.close()
            except:
                pass
        
        print("===== 关键协同基因图表绘制完成 =====\n")
    except Exception as e:
        print(f"绘制图表时出现未预期的错误: {e}")
        traceback.print_exc()

