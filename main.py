"""
主程序：黄芩苷与汉防己甲素联合应用增效减毒机制分析系统
"""

import os
import pandas as pd
import numpy as np
import sys
import argparse
import config

# 导入自定义模块
from modules.data_input import load_and_process_data, extract_targets_from_relations_complex, load_gene_pairs_data
from modules.synergy_analysis import analyze_common_targets, analyze_pathway_enrichment, analyze_synergy_mechanisms
from modules.visualization import plot_common_targets_network, plot_pathway_enrichment, plot_synergy_mechanisms

def diagnose_path_existence(all_relations, drug_id, disease_id):
    """
    诊断函数：检查是否存在从药物到疾病的完整路径
    
    参数:
    all_relations - 所有关系数据
    drug_id - 药物ID
    disease_id - 疾病ID
    
    返回:
    共同靶点集合
    """
    # 确保ID为字符串类型
    drug_id = str(drug_id)
    disease_id = str(disease_id)
    all_relations['Source_ID'] = all_relations['Source_ID'].astype(str)
    all_relations['Target_ID'] = all_relations['Target_ID'].astype(str)
    
    # 1. 找出药物直接关联的所有靶点
    drug_targets = set()
    
    # 1.1 药物→基因
    for _, rel in all_relations.iterrows():
        if rel['Source_ID'] == drug_id and rel['Source_Type'] == 'Chemical' and rel['Target_Type'] == 'Gene':
            drug_targets.add(rel['Target_ID'])
    
    # 1.2 基因→药物
    for _, rel in all_relations.iterrows():
        if rel['Target_ID'] == drug_id and rel['Target_Type'] == 'Chemical' and rel['Source_Type'] == 'Gene':
            drug_targets.add(rel['Source_ID'])
    
    print(f"药物(ID:{drug_id})直接关联的靶点数量: {len(drug_targets)}")
    
    # 2. 找出疾病直接关联的所有靶点
    disease_targets = set()
    
    # 2.1 基因→疾病
    for _, rel in all_relations.iterrows():
        if rel['Target_ID'] == disease_id and rel['Target_Type'] == 'Disease' and rel['Source_Type'] == 'Gene':
            disease_targets.add(rel['Source_ID'])
    
    # 2.2 疾病→基因
    for _, rel in all_relations.iterrows():
        if rel['Source_ID'] == disease_id and rel['Source_Type'] == 'Disease' and rel['Target_Type'] == 'Gene':
            disease_targets.add(rel['Target_ID'])
    
    print(f"疾病(ID:{disease_id})直接关联的靶点数量: {len(disease_targets)}")
    
    # 3. 找出共同靶点
    common_targets = drug_targets.intersection(disease_targets)
    
    print(f"药物(ID:{drug_id})和疾病(ID:{disease_id})的共同靶点数量: {len(common_targets)}")
    
    # 4. 如果有共同靶点，输出详细信息
    if common_targets:
        print("\n找到以下形成完整路径的靶点:")
        
        for target_id in common_targets:
            # 查找靶点名称
            target_name = "未知"
            
            # 尝试从关系中找出靶点名称
            for _, rel in all_relations.iterrows():
                if (rel['Source_ID'] == target_id and rel['Source_Type'] == 'Gene') or \
                   (rel['Target_ID'] == target_id and rel['Target_Type'] == 'Gene'):
                    if rel['Source_ID'] == target_id:
                        target_name = rel['Source_Name']
                    else:
                        target_name = rel['Target_Name']
                    break
            
            # 找出药物→靶点的关系
            drug_to_target_rel = all_relations[
                ((all_relations['Source_ID'] == drug_id) & (all_relations['Target_ID'] == target_id)) |
                ((all_relations['Source_ID'] == target_id) & (all_relations['Target_ID'] == drug_id))
            ]
            
            # 找出靶点→疾病的关系
            target_to_disease_rel = all_relations[
                ((all_relations['Source_ID'] == target_id) & (all_relations['Target_ID'] == disease_id)) |
                ((all_relations['Source_ID'] == disease_id) & (all_relations['Target_ID'] == target_id))
            ]
            
            # 输出详情
            print(f"  - 靶点ID: {target_id}, 名称: {target_name}")
            
            if not drug_to_target_rel.empty:
                rel = drug_to_target_rel.iloc[0]
                if rel['Source_ID'] == drug_id:
                    print(f"    * 药物→靶点关系: {rel['Type']}, 概率: {rel['Probability']}")
                else:
                    print(f"    * 靶点→药物关系: {rel['Type']}, 概率: {rel['Probability']}")
            
            if not target_to_disease_rel.empty:
                rel = target_to_disease_rel.iloc[0]
                if rel['Source_ID'] == target_id:
                    print(f"    * 靶点→疾病关系: {rel['Type']}, 概率: {rel['Probability']}")
                else:
                    print(f"    * 疾病→靶点关系: {rel['Type']}, 概率: {rel['Probability']}")
    
    return common_targets

def main():
    """主函数：运行完整的分析流程"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='黄芩苷与汉防己甲素联合应用增效减毒机制分析系统')
    parser.add_argument('--skip-analysis', action='store_true', help='跳过分析阶段，仅进行可视化')
    parser.add_argument('--skip-visualization', action='store_true', help='跳过可视化阶段，仅进行分析')
    parser.add_argument('--verbose', action='store_true', help='显示详细日志信息')
    parser.add_argument('--diagnose-only', action='store_true', help='仅执行路径诊断，不进行完整分析')
    args = parser.parse_args()
    
    verbose = args.verbose

    print("=" * 80)
    print("黄芩苷与汉防己甲素联合应用增效减毒机制分析系统")
    print("=" * 80)
    
    # 1. 数据加载和处理
    print("\n1. 数据加载和处理中...")
    
    try:
        # 加载数据
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
        
        if verbose:
            print(f"黄芩苷数据: {len(baicalin_relations)}行")
            print(f"汉防己甲素数据: {len(tetrandrine_relations)}行")
            print(f"硅肺病数据: {len(silicosis_relations)}行")
            print(f"肝毒性数据: {len(hepatotox_relations)}行")
            print(f"肾毒性数据: {len(nephrotox_relations)}行")
    
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("请确保数据文件路径正确，且格式符合要求。")
        return
    
    # 执行路径诊断
    print("\n执行路径诊断...")
    print("\n检查黄芩苷→靶点→硅肺病路径:")
    baicalin_silicosis_paths = diagnose_path_existence(
        all_relations, 
        baicalin['Source_ID'].values[0], 
        silicosis['Source_ID'].values[0]
    )
    
    print("\n检查汉防己甲素→靶点→硅肺病路径:")
    tetrandrine_silicosis_paths = diagnose_path_existence(
        all_relations, 
        tetrandrine['Source_ID'].values[0], 
        silicosis['Source_ID'].values[0]
    )
    
    # 检查共同路径
    if isinstance(baicalin_silicosis_paths, set) and isinstance(tetrandrine_silicosis_paths, set):
        common_path_targets = baicalin_silicosis_paths.intersection(tetrandrine_silicosis_paths)
        print(f"\n两种药物共同影响硅肺病的靶点数量: {len(common_path_targets)}")
    
    print("\n检查黄芩苷→靶点→肝毒性路径:")
    baicalin_hepatotox_paths = diagnose_path_existence(
        all_relations, 
        baicalin['Source_ID'].values[0], 
        hepatotox['Source_ID'].values[0]
    )
    
    print("\n检查汉防己甲素→靶点→肝毒性路径:")
    tetrandrine_hepatotox_paths = diagnose_path_existence(
        all_relations, 
        tetrandrine['Source_ID'].values[0], 
        hepatotox['Source_ID'].values[0]
    )
    
    print("\n检查黄芩苷→靶点→肾毒性路径:")
    baicalin_nephrotox_paths = diagnose_path_existence(
        all_relations, 
        baicalin['Source_ID'].values[0], 
        nephrotox['Source_ID'].values[0]
    )
    
    print("\n检查汉防己甲素→靶点→肾毒性路径:")
    tetrandrine_nephrotox_paths = diagnose_path_existence(
        all_relations, 
        tetrandrine['Source_ID'].values[0], 
        nephrotox['Source_ID'].values[0]
    )
    
    # 如果只需要诊断，则退出
    if args.diagnose_only:
        print("\n诊断完成!")
        return
    
    # 2. 提取靶点
    print("\n2. 提取药物和疾病靶点中...")
    
    # 提取黄芩苷靶点
    baicalin_targets = extract_targets_from_relations_complex(
        all_relations, 'Drug', baicalin['Source_ID'].values[0]
    )
    
    # 提取汉防己甲素靶点
    tetrandrine_targets = extract_targets_from_relations_complex(
        all_relations, 'Drug', tetrandrine['Source_ID'].values[0]
    )
    
    # 提取硅肺病相关靶点
    silicosis_targets = extract_targets_from_relations_complex(
        all_relations, 'Disease', silicosis['Source_ID'].values[0]
    )
    
    # 提取肝毒性相关靶点
    hepatotox_targets = extract_targets_from_relations_complex(
        all_relations, 'Hepatotoxicity', hepatotox['Source_ID'].values[0]
    )
    
    # 提取肾毒性相关靶点
    nephrotox_targets = extract_targets_from_relations_complex(
        all_relations, 'Nephrotoxicity', nephrotox['Source_ID'].values[0]
    )
    
    # 转换靶点列表为DataFrame
    baicalin_targets_df = pd.DataFrame({'Target_ID': baicalin_targets})
    tetrandrine_targets_df = pd.DataFrame({'Target_ID': tetrandrine_targets})
    silicosis_targets_df = pd.DataFrame({'Target_ID': silicosis_targets})
    hepatotox_targets_df = pd.DataFrame({'Target_ID': hepatotox_targets})
    nephrotox_targets_df = pd.DataFrame({'Target_ID': nephrotox_targets})
    
    # 3. 加载基因对关系数据
    print("\n3. 加载基因对关系数据中...")
    gene_pairs_data = load_gene_pairs_data()
    
    if not args.skip_analysis:
        # 4. 分析共同靶点
        print("\n4. 分析共同靶点中...")
        common_targets = analyze_common_targets(
            baicalin_targets_df, tetrandrine_targets_df, all_relations
        )
        
        # 5. 分析通路富集
        print("\n5. 分析通路富集中...")
        pathway_results = analyze_pathway_enrichment(common_targets, gene_pairs_data)
        
        # 6. 分析协同机制
        print("\n6. 分析协同增效减毒机制中...")
        synergy_results = analyze_synergy_mechanisms(
            baicalin_relations, tetrandrine_relations, common_targets,
            silicosis_relations, hepatotox_relations, nephrotox_relations
        )
        
        # 打印主要分析结果
        print("\n主要分析结果:")
        print(f"发现 {len(common_targets)} 个共同靶点")
        
        mechanism_types = synergy_results['mechanism_types']
        print(f"治疗效应: {mechanism_types['silicosis_treatment']['synergy_type']} (系数: {mechanism_types['silicosis_treatment']['synergy_coefficient']:.2f})")
        print(f"肝保护效应: {mechanism_types['hepatotoxicity_reduction']['synergy_type']} (系数: {mechanism_types['hepatotoxicity_reduction']['synergy_coefficient']:.2f})")
        print(f"肾保护效应: {mechanism_types['nephrotoxicity_reduction']['synergy_type']} (系数: {mechanism_types['nephrotoxicity_reduction']['synergy_coefficient']:.2f})")
    else:
        # 如果跳过分析阶段，则加载已保存的分析结果
        print("\n跳过分析阶段，加载已保存的分析结果...")
        
        # 加载共同靶点数据
        common_targets_path = os.path.join(config.PROCESSED_DATA_DIR, 'targets', 'common_targets.csv')
        if os.path.exists(common_targets_path):
            common_targets = pd.read_csv(common_targets_path)
        else:
            print(f"错误: 未找到共同靶点数据 {common_targets_path}")
            common_targets = None
        
        # 加载通路富集结果
        pathway_involvement_path = os.path.join(config.RESULTS_DIR, 'tables', 'pathway_involvement.csv')
        pathway_enrichment_path = os.path.join(config.RESULTS_DIR, 'tables', 'pathway_enrichment.csv')
        
        if os.path.exists(pathway_involvement_path) and os.path.exists(pathway_enrichment_path):
            pathway_involvement = pd.read_csv(pathway_involvement_path)
            pathway_enrichment = pd.read_csv(pathway_enrichment_path)
            
            pathway_results = {
                'pathway_involvement': pathway_involvement,
                'pathway_enrichment': pathway_enrichment
            }
        else:
            print(f"错误: 未找到通路富集分析结果")
            pathway_results = None
        
        # 加载协同机制分析结果
        mechanism_types_path = os.path.join(config.RESULTS_DIR, 'tables', 'mechanism_types.csv')
        therapeutic_targets_path = os.path.join(config.RESULTS_DIR, 'tables', 'therapeutic_targets.csv')
        hepatoprotective_targets_path = os.path.join(config.RESULTS_DIR, 'tables', 'hepatoprotective_targets.csv')
        nephroprotective_targets_path = os.path.join(config.RESULTS_DIR, 'tables', 'nephroprotective_targets.csv')
        
        if os.path.exists(mechanism_types_path):
            mechanism_types = pd.read_csv(mechanism_types_path).to_dict('records')[0]
            
            # 重建协同机制分析结果
            synergy_results = {
                'mechanism_types': mechanism_types
            }
            
            # 加载各类靶点数据
            if os.path.exists(therapeutic_targets_path):
                therapeutic_targets = pd.read_csv(therapeutic_targets_path).to_dict('records')
                synergy_results['therapeutic_targets'] = therapeutic_targets
            else:
                synergy_results['therapeutic_targets'] = []
            
            if os.path.exists(hepatoprotective_targets_path):
                hepatoprotective_targets = pd.read_csv(hepatoprotective_targets_path).to_dict('records')
                synergy_results['hepatoprotective_targets'] = hepatoprotective_targets
            else:
                synergy_results['hepatoprotective_targets'] = []
            
            if os.path.exists(nephroprotective_targets_path):
                nephroprotective_targets = pd.read_csv(nephroprotective_targets_path).to_dict('records')
                synergy_results['nephroprotective_targets'] = nephroprotective_targets
            else:
                synergy_results['nephroprotective_targets'] = []
        else:
            print(f"错误: 未找到协同机制分析结果")
            synergy_results = None
    
    if not args.skip_visualization:
        # 7. 可视化分析结果
        print("\n7. 可视化分析结果中...")
        
        if common_targets is not None:
            print("绘制共同靶点网络图...")
            plot_common_targets_network(common_targets, baicalin_relations, tetrandrine_relations)
        
        if pathway_results is not None:
            print("绘制通路富集分析图...")
            plot_pathway_enrichment(pathway_results)
        
        if synergy_results is not None:
            print("绘制协同机制分析图...")
            plot_synergy_mechanisms(synergy_results)
            plot_key_synergy_genes(synergy_results)   # 新添加的函数
    print("\n分析完成! 结果保存在 results 目录中。")
    print(f"- 表格数据: {os.path.join(config.RESULTS_DIR, 'tables')}")
    print(f"- 图形结果: {os.path.join(config.RESULTS_DIR, 'figures')}")

if __name__ == "__main__":
    main()
