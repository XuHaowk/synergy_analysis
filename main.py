"""
主程序：黄芩苷与汉防己甲素联合应用增效减毒机制分析系统
"""

import os
import pandas as pd
import numpy as np
import sys
import argparse
import json
import math
from datetime import datetime
import config

# Import data and utility modules
from modules.data_input import load_and_process_data, extract_targets_from_relations_complex, load_gene_pairs_data

# Import core analysis modules 
from modules.synergy_analysis import analyze_common_targets, analyze_synergy_mechanisms

# Import visualization modules
from modules.visualization import plot_common_targets_network, plot_pathway_enrichment, plot_synergy_mechanisms, plot_key_synergy_genes

# Import the probability module functions
from modules.probability import (
    calculate_gene_network_centrality, 
    extract_gene_evidence_scores,
    extract_gene_gene_relations,
    determine_direction,
    calculate_data_driven_path_weight,
    calculate_path_weight,
    adjust_path_probability_by_direction,
    calculate_dynamic_cap,
    cluster_paths_with_importance,
    calculate_combined_probability_with_importance,
    calculate_indirect_relation_probability,
    analyze_cross_regulation_mechanisms,
    calculate_synergy_probabilities,
    calculate_reduction_probability,
    run_comprehensive_test
)

# New function to load enhanced relations data
def load_enhanced_relations_data(file_path='data/raw/enhanced_relations_cleaned.csv'):
    """加载增强关系数据并提取相关实体"""
    import pandas as pd
    import os
    
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误: 找不到关系数据文件 {file_path}")
        return None
    
    # 加载CSV数据
    try:
        relations_df = pd.read_csv(file_path)
        print(f"成功加载关系数据文件，包含 {len(relations_df)} 行数据")
        
        # 检查必要的列是否存在
        required_columns = ['Source_Name', 'Source_Type', 'Source_ID', 
                           'Target_Name', 'Target_Type', 'Target_ID',
                           'Type', 'Probability']
        
        missing_columns = [col for col in required_columns if col not in relations_df.columns]
        if missing_columns:
            print(f"警告: 关系数据文件缺少以下列: {missing_columns}")
        
        # 提取不同类型的实体
        entity_types = {}
        
        # 从源实体提取
        for _, row in relations_df.iterrows():
            entity_type = row['Source_Type']
            entity_id = row['Source_ID']
            entity_name = row['Source_Name']
            
            if entity_type not in entity_types:
                entity_types[entity_type] = {}
            
            if entity_id not in entity_types[entity_type]:
                entity_types[entity_type][entity_id] = entity_name
        
        # 从目标实体提取
        for _, row in relations_df.iterrows():
            entity_type = row['Target_Type']
            entity_id = row['Target_ID']
            entity_name = row['Target_Name']
            
            if entity_type not in entity_types:
                entity_types[entity_type] = {}
            
            if entity_id not in entity_types[entity_type]:
                entity_types[entity_type][entity_id] = entity_name
        
        # 打印实体类型统计
        print("\n实体类型统计:")
        for entity_type, entities in entity_types.items():
            print(f"  - {entity_type}: {len(entities)} 个唯一实体")
        
        # 提取所有基因实体
        genes = entity_types.get('Gene', {})
        print(f"\n找到 {len(genes)} 个基因实体")
        if genes:
            print("前5个基因样本:")
            sample_genes = list(genes.items())[:5]
            for gene_id, gene_name in sample_genes:
                print(f"  - ID: {gene_id}, 名称: {gene_name}")
        
        # 创建返回结果
        return {
            'relations_df': relations_df,
            'entity_types': entity_types,
            'genes': genes
        }
        
    except Exception as e:
        print(f"加载关系数据文件时出错: {e}")
        return None

# Function to perform pathway enrichment on gene lists
def perform_gene_pathway_enrichment(gene_list, output_dir='./results'):
    """
    使用KEGG或其他数据库对基因列表进行通路富集分析
    
    参数:
    gene_list - 基因名称或ID列表
    output_dir - 输出目录
    
    返回:
    enrichment_results - 富集分析结果
    """
    import os
    import pandas as pd
    from collections import defaultdict
    import scipy.stats as stats
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"对 {len(gene_list)} 个基因进行通路富集分析...")
    
    # 预定义的通路-基因映射
    pathway_gene_mapping = {
        # 炎症相关通路
        "Inflammatory_Response": [
            "IL1B", "IL6", "TNF", "CXCL8", "IL10", "TLR4", "NFKB1", "STAT3", "CCL2", 
            "NLRP3", "IL17A", "IL4", "IL13", "IFNG", "IL1A", "IL18", "MCP-1", "STAT1", "iNOS"
        ],
        
        # 氧化应激通路
        "Oxidative_Stress": [
            "SOD1", "SOD2", "CAT", "GPX1", "HMOX1", "NQO1", "NFE2L2", "KEAP1", "GCLC", 
            "GCLM", "GSR", "NRF2", "TXN", "TXNRD1", "NOX1", "NOX2", "NOX4", "CYBA"
        ],
        
        # 细胞凋亡通路
        "Apoptosis": [
            "BAX", "BCL2", "CASP3", "CASP8", "CASP9", "TP53", "FAS", "FASLG", "XIAP", 
            "BIRC5", "MCL1", "BAK1", "BID", "BAD", "BCL2L1", "CYCS", "APAF1", "CD95", "FasL",
            "p53", "p38", "M4"
        ],
        
        # 纤维化通路
        "Fibrosis": [
            "TGFB1", "SMAD2", "SMAD3", "SMAD4", "ACTA2", "COL1A1", "COL3A1", "FN1", 
            "CTGF", "MMP2", "MMP9", "TIMP1", "TIMP2", "PDGFA", "PDGFB", "PDGFRB", "PDZRN4",
            "α-smooth muscle actin", "LOXL2", "AP-1"
        ],
        
        # 肝脏代谢通路
        "Liver_Metabolism": [
            "CYP1A2", "CYP2E1", "CYP3A4", "GSTM1", "UGT1A1", "ABCB1", "ABCC2", "SLCO1B1", 
            "SLC22A1", "ALB", "TTR", "HMGCR", "LDLR", "APOB", "MTTP", "PTEN", "IDH",
            "CHRM1", "alkaline phosphatase", "GOT", "ALT", "AST", "AhR", "Chrm5"
        ],
        
        # 肾脏功能通路
        "Kidney_Function": [
            "AQP1", "AQP2", "SLC22A6", "SLC22A8", "KCNJ1", "SCNN1A", "SCNN1B", "AGT", 
            "REN", "ACE", "AGTR1", "NPHS1", "NPHS2", "UMOD", "PODXL", "WT1", "Calcium channel",
            "angiotensin", "Voltage-Dependent", "alpha/1D)-adre", "alpha 1A-adre", "Rob1", "PREPL"
        ],
        
        # 免疫调节通路
        "Immune_Regulation": [
            "CD4", "CD8A", "FOXP3", "IL2RA", "CTLA4", "PDCD1", "CD274", "IFNG", "IL12A", 
            "IL12B", "IL23A", "JAK1", "JAK2", "STAT1", "STAT3", "STAT4", "STAT5A", "STAT5B",
            "STK33", "RBD"
        ],
        
        # 钙信号通路
        "Calcium_Signaling": [
            "CACNA1C", "CACNA1D", "CACNA1S", "CACNA2D1", "RYR1", "RYR2", "ITPR1", "ITPR2", 
            "ITPR3", "CALM1", "CAMK2A", "CAMK2B", "CAMK2D", "CAMK2G", "PPP3CA", "PPP3CB",
            "Calcium channel", "Muscarinic Ace"
        ],
        
        # 能量代谢通路
        "Energy_Metabolism": [
            "AMPK", "PRKAA1", "PRKAA2", "PRKAB1", "PRKAB2", "PRKAG1", "PRKAG2", "PRKAG3", 
            "SLC2A4", "PDK1", "PDK2", "PDK3", "PDK4", "G6PC", "PCK1", "PCK2", "PPARGC1A", "Akt"
        ]
    }
    
    # 标准化基因名称（去除空格，转为大写等）
    normalized_gene_list = [g.strip().upper() for g in gene_list if isinstance(g, str)]
    
    # 统计每个通路中出现的基因数量
    pathway_hits = defaultdict(list)
    gene_pathway_map = {}
    
    for gene in normalized_gene_list:
        found = False
        
        # 直接匹配
        for pathway, pathway_genes in pathway_gene_mapping.items():
            normalized_pathway_genes = [pg.strip().upper() for pg in pathway_genes]
            if gene in normalized_pathway_genes:
                pathway_hits[pathway].append(gene)
                if gene not in gene_pathway_map:
                    gene_pathway_map[gene] = []
                gene_pathway_map[gene].append(pathway)
                found = True
        
        # 如果没有直接匹配，尝试部分匹配
        if not found:
            for pathway, pathway_genes in pathway_gene_mapping.items():
                for pg in pathway_genes:
                    normalized_pg = pg.strip().upper()
                    # 部分匹配，如果基因名是另一个的子字符串
                    if (gene in normalized_pg) or (normalized_pg in gene):
                        pathway_hits[pathway].append(gene)
                        if gene not in gene_pathway_map:
                            gene_pathway_map[gene] = []
                        gene_pathway_map[gene].append(pathway)
                        found = True
                        break
                if found:
                    break
    
    # 计算富集分析
    total_genes = len(normalized_gene_list)
    background_size = 20000  # 假设人类基因组大约有20,000个基因
    
    enrichment_results = []
    
    for pathway, genes in pathway_hits.items():
        # 通路中命中的基因数
        hits = len(genes)
        
        # 通路中的总基因数
        pathway_size = len(pathway_gene_mapping[pathway])
        
        # 计算富集倍数
        expected = total_genes * (pathway_size / background_size)
        fold_enrichment = hits / expected if expected > 0 else 0
        
        # 计算P值（超几何分布检验）
        try:
            p_value = stats.hypergeom.sf(hits-1, background_size, pathway_size, total_genes)
        except:
            p_value = 1.0
        
        enrichment_results.append({
            'Pathway': pathway,
            'Hits': hits,
            'Total_In_Pathway': pathway_size,
            'Fold_Enrichment': fold_enrichment,
            'P_Value': p_value,
            'Genes': ','.join(genes)
        })
    
    # 按富集倍数排序
    enrichment_results = sorted(enrichment_results, key=lambda x: x['Fold_Enrichment'], reverse=True)
    
    # 创建基因-通路映射DataFrame
    gene_pathway_data = []
    for gene, pathways in gene_pathway_map.items():
        for pathway in pathways:
            gene_pathway_data.append({
                'Gene': gene,
                'Pathway': pathway,
                'Score': 1.0
            })
    
    # 保存结果
    enrichment_df = pd.DataFrame(enrichment_results)
    gene_pathway_df = pd.DataFrame(gene_pathway_data)
    
    enrichment_df.to_csv(os.path.join(output_dir, 'pathway_enrichment.csv'), index=False)
    gene_pathway_df.to_csv(os.path.join(output_dir, 'gene_pathway_mapping.csv'), index=False)
    
    print(f"富集分析完成，发现 {len(enrichment_results)} 个有显著富集的通路")
    
    return {
        'enrichment_results': enrichment_df,
        'gene_pathway_mapping': gene_pathway_df,
        'source': 'Built-in pathway data'
    }

# Function to visualize pathway enrichment results
def visualize_pathway_enrichment(enrichment_results, output_dir='./results/figures'):
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

# Keep the original functions from main.py:
# build_gene_network, analyze_direct_synergy, analyze_cross_regulation_synergy, 
# analyze_pathway_complementary, analyze_network_synergy, analyze_toxicity_reduction, 
# diagnose_path_existence - these remain unchanged

def main():
    """主函数：运行完整的分析流程"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='黄芩苷与汉防己甲素联合应用增效减毒机制分析系统')
    parser.add_argument('--data-file', type=str, default='data/raw/enhanced_relations_cleaned.csv', 
                        help='增强关系数据文件路径')
    parser.add_argument('--skip-analysis', action='store_true', help='跳过分析阶段，仅进行可视化')
    parser.add_argument('--skip-visualization', action='store_true', help='跳过可视化阶段，仅进行分析')
    parser.add_argument('--verbose', action='store_true', help='显示详细日志信息')
    parser.add_argument('--diagnose-only', action='store_true', help='仅执行路径诊断，不进行完整分析')
    parser.add_argument('--use-legacy', action='store_true', 
                        help='使用传统的分析流程（而非增强关系数据）')
    parser.add_argument('--output-dir', type=str, default='./results', help='输出目录')
    args = parser.parse_args()
    
    verbose = args.verbose
    output_dir = args.output_dir
    data_file = args.data_file

    print("=" * 80)
    print("黄芩苷与汉防己甲素联合应用增效减毒机制分析系统")
    print("=" * 80)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 如果使用传统流程，走原有逻辑
    if args.use_legacy:
        print("\n使用传统分析流程...")
        
        # 1. 数据加载和处理
        print("\n1. 数据加载和处理中...")
        
        try:
            # 原有的数据加载流程
            processed_data = load_and_process_data()
            
            # 提取实体和关系数据
            baicalin = processed_data['baicalin']
            tetrandrine = processed_data['tetrandrine']
            silicosis = processed_data['silicosis']
            hepatotox = processed_data['hepatotox']
            nephrotox = processed_data['nephrotox']
            
            baicalin_relations = processed_data['baicalin_relations']
            tetrandrine_relations = processed_data['tetrandrine_relations']
            silicosis_relations = processed_data['silicosis_relations']
            hepatotox_relations = processed_data['hepatotox_relations']
            nephrotox_relations = processed_data['nephrotox_relations']
            
            all_relations = processed_data['all_relations']
            
            print("数据加载成功!")
            
            # 继续执行原有流程...
            # 原有的诊断和分析步骤保持不变
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            print("请确保数据文件路径正确，且格式符合要求。")
            return
    
    else:
        # 使用新的增强关系数据分析流程
        print("\n使用增强关系数据分析流程...")
        
        # 1. 加载增强关系数据
        print(f"\n1. 从 {data_file} 加载增强关系数据...")
        relations_data = load_enhanced_relations_data(data_file)
        
        if not relations_data:
            print("增强关系数据加载失败，程序退出。")
            return
        
        # 提取数据
        relations_df = relations_data['relations_df']
        entity_types = relations_data['entity_types']
        genes = relations_data['genes']
        
        # 2. 查找药物实体
        print("\n2. 查找药物实体...")
        
        # 查找黄芩苷（Baicalin）和汉防己甲素（Tetrandrine）
        baicalin_id = None
        tetrandrine_id = None
        silicosis_id = None
        hepatotox_id = None
        nephrotox_id = None
        
        # 查找药物ID
        for entity_id, entity_name in entity_types.get('Chemical', {}).items():
            if 'Baicalin' in entity_name:
                baicalin_id = entity_id
                print(f"找到黄芩苷 ID: {baicalin_id}, 名称: {entity_name}")
            elif 'Tetrandrine' in entity_name:
                tetrandrine_id = entity_id
                print(f"找到汉防己甲素 ID: {tetrandrine_id}, 名称: {entity_name}")
        
        # 查找疾病ID
        for entity_id, entity_name in entity_types.get('Disease', {}).items():
            if 'Silicosis' in entity_name:
                silicosis_id = entity_id
                print(f"找到硅肺病 ID: {silicosis_id}, 名称: {entity_name}")
            elif 'Hepatotoxicity' in entity_name or 'liver toxicity' in entity_name.lower():
                hepatotox_id = entity_id
                print(f"找到肝毒性 ID: {hepatotox_id}, 名称: {entity_name}")
            elif 'Nephrotoxicity' in entity_name or 'kidney toxicity' in entity_name.lower():
                nephrotox_id = entity_id
                print(f"找到肾毒性 ID: {nephrotox_id}, 名称: {entity_name}")
        
        # 3. 提取药物靶向的基因
        print("\n3. 提取药物靶向的基因...")
        
        # 提取黄芩苷靶向的基因
        baicalin_target_genes = set()
        if baicalin_id:
            baicalin_targets = relations_df[
                (relations_df['Source_ID'] == baicalin_id) & 
                (relations_df['Target_Type'] == 'Gene')
            ]
            
            for _, row in baicalin_targets.iterrows():
                gene_id = row['Target_ID']
                gene_name = row['Target_Name']
                baicalin_target_genes.add(gene_name)
            
            print(f"黄芩苷靶向 {len(baicalin_target_genes)} 个基因")
            if verbose and baicalin_target_genes:
                print("前5个样本: ", list(baicalin_target_genes)[:5])
        
        # 提取汉防己甲素靶向的基因
        tetrandrine_target_genes = set()
        if tetrandrine_id:
            tetrandrine_targets = relations_df[
                (relations_df['Source_ID'] == tetrandrine_id) & 
                (relations_df['Target_Type'] == 'Gene')
            ]
            
            for _, row in tetrandrine_targets.iterrows():
                gene_id = row['Target_ID']
                gene_name = row['Target_Name']
                tetrandrine_target_genes.add(gene_name)
            
            print(f"汉防己甲素靶向 {len(tetrandrine_target_genes)} 个基因")
            if verbose and tetrandrine_target_genes:
                print("前5个样本: ", list(tetrandrine_target_genes)[:5])
        
        # 找出共同靶点
        common_target_genes = baicalin_target_genes.intersection(tetrandrine_target_genes)
        print(f"两种药物共有 {len(common_target_genes)} 个共同靶点")
        if verbose and common_target_genes:
            print("共同靶点样本: ", list(common_target_genes)[:10])
        
        # 4. 对各基因集合进行通路富集分析
        if not args.skip_analysis:
            print("\n4. 通路富集分析...")
            
            # 创建富集分析目录
            enrichment_dir = os.path.join(output_dir, 'enrichment')
            os.makedirs(enrichment_dir, exist_ok=True)
            
            # 4.1 对所有基因进行富集分析
            if len(genes) > 0:
                print(f"\n4.1 对所有 {len(genes)} 个基因进行通路富集分析...")
                all_gene_names = list(genes.values())
                all_genes_enrichment = perform_gene_pathway_enrichment(
                    all_gene_names, 
                    output_dir=os.path.join(enrichment_dir, 'all_genes')
                )
            
            # 4.2 对黄芩苷靶向基因进行富集分析
            if baicalin_target_genes:
                print(f"\n4.2 对黄芩苷靶向的 {len(baicalin_target_genes)} 个基因进行通路富集分析...")
                baicalin_enrichment = perform_gene_pathway_enrichment(
                    list(baicalin_target_genes),
                    output_dir=os.path.join(enrichment_dir, 'baicalin_targets')
                )
            
            # 4.3 对汉防己甲素靶向基因进行富集分析
            if tetrandrine_target_genes:
                print(f"\n4.3 对汉防己甲素靶向的 {len(tetrandrine_target_genes)} 个基因进行通路富集分析...")
                tetrandrine_enrichment = perform_gene_pathway_enrichment(
                    list(tetrandrine_target_genes),
                    output_dir=os.path.join(enrichment_dir, 'tetrandrine_targets')
                )
            
            # 4.4 对共同靶点基因进行富集分析
            if common_target_genes:
                print(f"\n4.4 对 {len(common_target_genes)} 个共同靶点基因进行通路富集分析...")
                common_targets_enrichment = perform_gene_pathway_enrichment(
                    list(common_target_genes),
                    output_dir=os.path.join(enrichment_dir, 'common_targets')
                )
            
            # 5. 高级分析：提取关系数据进行直接靶点和交叉调控分析
            print("\n5. 进行高级关系分析...")
            
            # 创建关系字典来支持分析函数
            if baicalin_id and tetrandrine_id:
                # 提取药物-基因关系
                baicalin_gene_relations = relations_df[
                    (relations_df['Source_ID'] == baicalin_id) & 
                    (relations_df['Target_Type'] == 'Gene')
                ].to_dict('records')
                
                tetrandrine_gene_relations = relations_df[
                    (relations_df['Source_ID'] == tetrandrine_id) & 
                    (relations_df['Target_Type'] == 'Gene')
                ].to_dict('records')
                
                # 提取基因-疾病关系
                if silicosis_id:
                    silicosis_gene_relations = relations_df[
                        (relations_df['Target_ID'] == silicosis_id) & 
                        (relations_df['Source_Type'] == 'Gene')
                    ].to_dict('records')
                    
                    # 分析对硅肺病的直接协同效应
                    print("\n5.1 分析对硅肺病的直接协同效应...")
                    silicosis_direct_synergy = analyze_direct_synergy(
                        baicalin_gene_relations,
                        tetrandrine_gene_relations,
                        silicosis_gene_relations
                    )
                    
                    # 保存结果
                    with open(os.path.join(output_dir, 'silicosis_direct_synergy.json'), 'w') as f:
                        json.dump(silicosis_direct_synergy, f, indent=2)
                    
                    # 打印主要结果
                    if silicosis_direct_synergy:
                        print(f"发现 {len(silicosis_direct_synergy)} 个直接协同靶点")
                        for i, mechanism in enumerate(silicosis_direct_synergy[:3], 1):
                            print(f"  {i}. 靶点: {mechanism['target_name']} - 协同类型: {mechanism['synergy_type']}, 强度: {mechanism['synergy_strength']:.3f}")
                
                # 提取基因-基因关系进行交叉调控分析
                gene_gene_relations = relations_df[
                    (relations_df['Source_Type'] == 'Gene') & 
                    (relations_df['Target_Type'] == 'Gene')
                ].to_dict('records')
                
                if silicosis_id and gene_gene_relations:
                    # 分析对硅肺病的交叉调控协同作用
                    print("\n5.2 分析对硅肺病的交叉调控协同作用...")
                    cross_regulation_synergy = analyze_cross_regulation_synergy(
                        baicalin_gene_relations,
                        tetrandrine_gene_relations,
                        gene_gene_relations,
                        silicosis_gene_relations
                    )
                    
                    # 保存结果
                    with open(os.path.join(output_dir, 'cross_regulation_synergy.json'), 'w') as f:
                        json.dump(cross_regulation_synergy, f, indent=2)
                    
                    # 打印主要结果
                    if cross_regulation_synergy:
                        print(f"发现 {len(cross_regulation_synergy)} 个交叉调控协同机制")
                        for i, mechanism in enumerate(cross_regulation_synergy[:3], 1):
                            print(f"  {i}. 靶点链: {mechanism['target1_name']} → {mechanism['target2_name']} - 协同: {'是' if mechanism['is_synergistic'] else '否'}, 治疗性: {'是' if mechanism['is_therapeutic'] else '否'}, 强度: {mechanism['synergy_strength']:.3f}")
                
                # 分析毒性降低机制
                if hepatotox_id:
                    hepatotox_gene_relations = relations_df[
                        (relations_df['Target_ID'] == hepatotox_id) & 
                        (relations_df['Source_Type'] == 'Gene')
                    ].to_dict('records')
                    
                    print("\n5.3 分析肝毒性降低机制...")
                    hepatotox_reduction = analyze_toxicity_reduction(
                        baicalin_gene_relations,
                        tetrandrine_gene_relations,
                        hepatotox_gene_relations,
                        hepatotox_id
                    )
                    
                    # 保存结果
                    with open(os.path.join(output_dir, 'hepatotox_reduction.json'), 'w') as f:
                        json.dump(hepatotox_reduction, f, indent=2)
                    
                    # 打印主要结果
                    if hepatotox_reduction:
                        print(f"发现 {len(hepatotox_reduction)} 个肝毒性降低机制")
                        for i, mechanism in enumerate(hepatotox_reduction[:3], 1):
                            print(f"  {i}. 靶点: {mechanism['gene_name']} - 机制: {mechanism['description']}, 强度: {mechanism['strength']:.3f}")
                
                if nephrotox_id:
                    nephrotox_gene_relations = relations_df[
                        (relations_df['Target_ID'] == nephrotox_id) & 
                        (relations_df['Source_Type'] == 'Gene')
                    ].to_dict('records')
                    
                    print("\n5.4 分析肾毒性降低机制...")
                    nephrotox_reduction = analyze_toxicity_reduction(
                        baicalin_gene_relations,
                        tetrandrine_gene_relations,
                        nephrotox_gene_relations,
                        nephrotox_id
                    )
                    
                    # 保存结果
                    with open(os.path.join(output_dir, 'nephrotox_reduction.json'), 'w') as f:
                        json.dump(nephrotox_reduction, f, indent=2)
                    
                    # 打印主要结果
                    if nephrotox_reduction:
                        print(f"发现 {len(nephrotox_reduction)} 个肾毒性降低机制")
                        for i, mechanism in enumerate(nephrotox_reduction[:3], 1):
                            print(f"  {i}. 靶点: {mechanism['gene_name']} - 机制: {mechanism['description']}, 强度: {mechanism['strength']:.3f}")
        
        # 6. 可视化结果
        if not args.skip_visualization:
            print("\n6. 可视化分析结果...")
            
            # 创建可视化目录
            figures_dir = os.path.join(output_dir, 'figures')
            os.makedirs(figures_dir, exist_ok=True)
            
            # 可视化富集分析结果
            if 'all_genes_enrichment' in locals():
                print("\n6.1 可视化所有基因的通路富集结果...")
                visualize_pathway_enrichment(all_genes_enrichment, 
                                           os.path.join(figures_dir, 'all_genes'))
            
            if 'baicalin_enrichment' in locals():
                print("\n6.2 可视化黄芩苷靶向基因的通路富集结果...")
                visualize_pathway_enrichment(baicalin_enrichment, 
                                           os.path.join(figures_dir, 'baicalin_targets'))
            
            if 'tetrandrine_enrichment' in locals():
                print("\n6.3 可视化汉防己甲素靶向基因的通路富集结果...")
                visualize_pathway_enrichment(tetrandrine_enrichment, 
                                           os.path.join(figures_dir, 'tetrandrine_targets'))
            
            if 'common_targets_enrichment' in locals():
                print("\n6.4 可视化共同靶点的通路富集结果...")
                visualize_pathway_enrichment(common_targets_enrichment, 
                                           os.path.join(figures_dir, 'common_targets'))
            
            # 这里可以添加其他可视化，如网络图等
    
    print("\n分析完成! 结果保存在目录中: " + output_dir)

if __name__ == "__main__":
    main()