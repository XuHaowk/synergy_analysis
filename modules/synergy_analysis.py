"""
协同作用分析模块：分析增效减毒机制
"""

import pandas as pd
import numpy as np
import os
from collections import Counter
import config
from modules.probability import calculate_direct_relation_probability, calculate_indirect_relation_probability, calculate_synergistic_effect

def analyze_common_targets(baicalin_targets, tetrandrine_targets, all_relations):
    """
    分析两种药物的共同靶点
    
    参数:
    baicalin_targets - 黄芩苷靶点
    tetrandrine_targets - 汉防己甲素靶点
    all_relations - 所有关系数据
    
    返回:
    共同靶点数据和分析结果
    """
    # 提取靶点ID列表
    baicalin_target_ids = set(baicalin_targets['Target_ID'])
    tetrandrine_target_ids = set(tetrandrine_targets['Target_ID'])
    
    # 识别共同靶点
    common_target_ids = baicalin_target_ids.intersection(tetrandrine_target_ids)
    
    print(f"黄芩苷靶点: {len(baicalin_target_ids)}")
    print(f"汉防己甲素靶点: {len(tetrandrine_target_ids)}")
    print(f"共同靶点: {len(common_target_ids)}")
    
    # 创建共同靶点DataFrame
    common_targets = pd.DataFrame({
        'Target_ID': list(common_target_ids)
    })
    
    # 添加靶点名称和其他信息
    target_info = all_relations[all_relations['Target_ID'].isin(common_target_ids)]
    target_names = {}
    for _, row in target_info.iterrows():
        if row['Target_ID'] not in target_names:
            target_names[row['Target_ID']] = row['Target_Name']
    
    common_targets['Target_Name'] = common_targets['Target_ID'].map(target_names)
    
    # 计算每个靶点的重要性 (出现频率)
    target_counts = Counter(target_info['Target_ID'])
    common_targets['Importance'] = common_targets['Target_ID'].map(lambda x: target_counts.get(x, 0))
    
    # 按重要性排序
    common_targets = common_targets.sort_values('Importance', ascending=False).reset_index(drop=True)
    
    # 添加靶点与功能相关性分析
    # 这需要额外的生物信息数据，可以后续扩展
    
    # 保存共同靶点数据
    os.makedirs(os.path.join(config.PROCESSED_DATA_DIR, 'targets'), exist_ok=True)
    common_targets.to_csv(os.path.join(config.PROCESSED_DATA_DIR, 'targets', 'common_targets.csv'), index=False)
    
    return common_targets

def analyze_pathway_enrichment(common_targets, gene_pairs_data):
    """
    分析共同靶点的通路富集情况
    
    参数:
    common_targets - 共同靶点数据
    gene_pairs_data - 基因对关系数据
    
    返回:
    通路富集分析结果
    """
    # 获取基因对关系数据
    gene_pairs = gene_pairs_data['gene_pairs']
    gene_pairs_db = gene_pairs_data['gene_pairs_db']
    gene_pairs_pubmed = gene_pairs_data['gene_pairs_pubmed']
    
    # 初始化通路结果
    pathway_results = pd.DataFrame()
    
    # 根据基因对关系，检查靶点是否参与关键通路
    target_ids = common_targets['Target_ID'].values
    
    # 定义关键通路和对应的标志基因
    key_pathways = {
        'inflammatory': ['IL6', 'TNF', 'IL1B', 'CXCL8', 'NFKB1'],
        'oxidative_stress': ['SOD1', 'CAT', 'GPX1', 'NFE2L2', 'HMOX1'],
        'apoptosis': ['BAX', 'BCL2', 'CASP3', 'TP53', 'PARP1'],
        'fibrosis': ['TGFB1', 'COL1A1', 'MMP9', 'TIMP1', 'ACTA2'],
        'liver_metabolism': ['CYP3A4', 'CYP2E1', 'CYP1A2', 'UGT1A1', 'ABCB1'],
        'kidney_function': ['AQP1', 'SLC22A6', 'SLC22A8', 'HAVCR1', 'LCN2']
    }
    
    # 计算每个靶点对各通路的参与度
    pathway_involvement = []
    
    for target_id in target_ids:
        # 在基因对中寻找该靶点
        target_pairs_db = gene_pairs_db[(gene_pairs_db['Source_ID'] == target_id) | (gene_pairs_db['Target_ID'] == target_id)]
        target_pairs_pubmed = gene_pairs_pubmed[(gene_pairs_pubmed['Source_ID'] == target_id) | (gene_pairs_pubmed['Target_ID'] == target_id)]
        
        # 提取关联基因
        related_genes_db = set(target_pairs_db['Source_ID']).union(set(target_pairs_db['Target_ID']))
        related_genes_pubmed = set(target_pairs_pubmed['Source_ID']).union(set(target_pairs_pubmed['Target_ID']))
        related_genes = related_genes_db.union(related_genes_pubmed)
        
        # 从关联基因中移除自身
        related_genes.discard(target_id)
        
        # 计算各通路的参与度
        pathway_scores = {}
        
        for pathway, marker_genes in key_pathways.items():
            # 检查有多少标志基因与该靶点相关
            overlap = len([gene for gene in related_genes if str(gene) in marker_genes or 
                           any(marker in str(gene_name).upper() for gene_name in common_targets[common_targets['Target_ID'] == target_id]['Target_Name'] 
                               for marker in marker_genes)])
            
            pathway_scores[pathway] = overlap / len(marker_genes) if overlap > 0 else 0
        
        # 获取靶点名称
        target_name = common_targets[common_targets['Target_ID'] == target_id]['Target_Name'].values[0]
        
        # 添加到结果
        pathway_involvement.append({
            'Target_ID': target_id,
            'Target_Name': target_name,
            'Inflammatory_Score': pathway_scores['inflammatory'],
            'Oxidative_Stress_Score': pathway_scores['oxidative_stress'],
            'Apoptosis_Score': pathway_scores['apoptosis'],
            'Fibrosis_Score': pathway_scores['fibrosis'],
            'Liver_Metabolism_Score': pathway_scores['liver_metabolism'],
            'Kidney_Function_Score': pathway_scores['kidney_function'],
            'Related_Genes_Count': len(related_genes)
        })
    
    # 转换为DataFrame
    pathway_results = pd.DataFrame(pathway_involvement)
    
    # 计算每个通路的总体富集程度
    pathway_enrichment = {
        'Inflammatory_Pathway': pathway_results['Inflammatory_Score'].mean(),
        'Oxidative_Stress_Pathway': pathway_results['Oxidative_Stress_Score'].mean(),
        'Apoptosis_Pathway': pathway_results['Apoptosis_Score'].mean(),
        'Fibrosis_Pathway': pathway_results['Fibrosis_Score'].mean(),
        'Liver_Metabolism_Pathway': pathway_results['Liver_Metabolism_Score'].mean(),
        'Kidney_Function_Pathway': pathway_results['Kidney_Function_Score'].mean()
    }
    
    # 保存通路分析结果
    os.makedirs(os.path.join(config.RESULTS_DIR, 'tables'), exist_ok=True)
    pathway_results.to_csv(os.path.join(config.RESULTS_DIR, 'tables', 'pathway_involvement.csv'), index=False)
    
    # 保存通路富集结果
    pd.DataFrame([pathway_enrichment]).to_csv(os.path.join(config.RESULTS_DIR, 'tables', 'pathway_enrichment.csv'), index=False)
    
    return {
        'pathway_involvement': pathway_results,
        'pathway_enrichment': pathway_enrichment
    }

def get_significant_targets(drug1_relations, drug2_relations, common_targets):
    """
    获取两种药物对某一疾病或毒性的显著共同靶点
    
    参数:
    drug1_relations - 药物1与疾病/毒性的间接关系数据
    drug2_relations - 药物2与疾病/毒性的间接关系数据
    common_targets - 两种药物的共同靶点数据
    
    返回:
    显著共同靶点列表
    """
    # 如果任一关系数据为空，返回空列表
    if drug1_relations.empty or drug2_relations.empty or common_targets.empty:
        return []
    
    significant_targets = []
    
    # 获取药物1中所有中间靶点的ID和名称
    drug1_targets = {}
    if 'Path_Details' in drug1_relations.columns:
        for _, row in drug1_relations.iterrows():
            if isinstance(row['Path_Details'], list):
                for path in row['Path_Details']:
                    if isinstance(path, dict) and 'intermediate_target_id' in path:
                        target_id = path['intermediate_target_id']
                        target_name = path.get('intermediate_target_name', "Unknown")
                        path_prob = path.get('path_probability', 0)
                        
                        if target_id not in drug1_targets or path_prob > drug1_targets[target_id]['probability']:
                            drug1_targets[target_id] = {
                                'name': target_name,
                                'probability': path_prob,
                                'direction': path.get('source_to_inter_direction', "unknown")
                            }
    
    # 获取药物2中所有中间靶点的ID和名称
    drug2_targets = {}
    if 'Path_Details' in drug2_relations.columns:
        for _, row in drug2_relations.iterrows():
            if isinstance(row['Path_Details'], list):
                for path in row['Path_Details']:
                    if isinstance(path, dict) and 'intermediate_target_id' in path:
                        target_id = path['intermediate_target_id']
                        target_name = path.get('intermediate_target_name', "Unknown")
                        path_prob = path.get('path_probability', 0)
                        
                        if target_id not in drug2_targets or path_prob > drug2_targets[target_id]['probability']:
                            drug2_targets[target_id] = {
                                'name': target_name,
                                'probability': path_prob,
                                'direction': path.get('source_to_inter_direction', "unknown")
                            }
    
    # 找出共同的显著靶点
    common_target_ids = set(drug1_targets.keys()).intersection(set(drug2_targets.keys()))
    
    # 将Target_ID列转换为字符串，以便进行匹配
    if 'Target_ID' in common_targets.columns:
        common_targets['Target_ID'] = common_targets['Target_ID'].astype(str)
    
    # 检查共同靶点是否也在两种药物的共同靶点列表中
    for target_id in common_target_ids:
        # 检查是否在共同靶点列表中
        is_common = False
        target_row = None
        
        if 'Target_ID' in common_targets.columns:
            matches = common_targets[common_targets['Target_ID'] == target_id]
            if not matches.empty:
                is_common = True
                target_row = matches.iloc[0]
        
        # 如果是共同靶点，则添加到显著靶点列表
        if is_common:
            drug1_info = drug1_targets[target_id]
            drug2_info = drug2_targets[target_id]
            
            # 判断协同类型
            if drug1_info['direction'] == drug2_info['direction']:
                synergy_type = "协同型" if drug1_info['direction'] == "positive" else "拮抗型"
            else:
                synergy_type = "互补型"
            
            # 计算综合概率
            combined_probability = drug1_info['probability'] * drug2_info['probability']
            
            # 只添加概率大于阈值的靶点
            if combined_probability > 0.2:  # 可以根据需要调整阈值
                target_info = {
                    'Target_ID': target_id,
                    'Target_Name': drug1_info['name'],
                    'Drug1_Probability': drug1_info['probability'],
                    'Drug1_Direction': drug1_info['direction'],
                    'Drug2_Probability': drug2_info['probability'],
                    'Drug2_Direction': drug2_info['direction'],
                    'Combined_Probability': combined_probability,
                    'Synergy_Type': synergy_type
                }
                
                # 添加其他可能的靶点信息
                if target_row is not None:
                    for col in target_row.index:
                        if col not in target_info and col != 'Target_ID':
                            target_info[col] = target_row[col]
                
                significant_targets.append(target_info)
    
    # 按照综合概率降序排序
    significant_targets.sort(key=lambda x: x['Combined_Probability'], reverse=True)
    
    return significant_targets

def analyze_synergy_mechanisms(
        baicalin_relations, tetrandrine_relations, common_targets,
        silicosis_relations, hepatotox_relations, nephrotox_relations):
    """
    分析黄芩苷和汉防己甲素的协同增效和减毒机制，基于PSR算法
    
    参数:
    baicalin_relations - 黄芩苷的关系数据
    tetrandrine_relations - 汉防己甲素的关系数据
    common_targets - 共同靶点数据
    silicosis_relations - 硅肺病的关系数据
    hepatotox_relations - 肝毒性的关系数据
    nephrotox_relations - 肾毒性的关系数据
    
    返回:
    协同机制分析结果
    """
    import math
    import pandas as pd
    
    print("计算直接关系概率...")
    
    # 确保关系数据中包含必要的列
    def ensure_relation_columns(relation_df, entity_type):
        """确保关系数据中包含必要的列"""
        relation_df = relation_df.copy()
        
        # 添加缺失的列
        if 'Source_Type' not in relation_df.columns:
            if entity_type in ['Drug', 'Chemical']:
                relation_df['Source_Type'] = 'Chemical'
            else:
                relation_df['Source_Type'] = 'Disease'
                
        if 'Target_Type' not in relation_df.columns:
            relation_df['Target_Type'] = 'Gene'
            
        return relation_df
    
    # 处理所有关系数据
    baicalin_relations = ensure_relation_columns(baicalin_relations, 'Drug')
    tetrandrine_relations = ensure_relation_columns(tetrandrine_relations, 'Drug')
    silicosis_relations = ensure_relation_columns(silicosis_relations, 'Disease')
    hepatotox_relations = ensure_relation_columns(hepatotox_relations, 'Disease')
    nephrotox_relations = ensure_relation_columns(nephrotox_relations, 'Disease')
    
    # 计算对硅肺病的间接关系概率（使用PSR算法）
    print("计算对硅肺病的间接关系概率...")
    baicalin_silicosis = calculate_indirect_relation_probability(
        baicalin_relations, silicosis_relations
    )
    tetrandrine_silicosis = calculate_indirect_relation_probability(
        tetrandrine_relations, silicosis_relations
    )
    
    # 增强调试输出
    print("硅肺病协同作用分析:")
    if not baicalin_silicosis.empty and 'Indirect_Probability' in baicalin_silicosis.columns:
        probs = baicalin_silicosis['Indirect_Probability'].tolist()
        print(f"黄芩苷-硅肺病路径数: {len(baicalin_silicosis)}")
        print(f"最大概率值: {max(probs) if probs else 0}")
    
    if not tetrandrine_silicosis.empty and 'Indirect_Probability' in tetrandrine_silicosis.columns:
        probs = tetrandrine_silicosis['Indirect_Probability'].tolist()
        print(f"汉防己甲素-硅肺病路径数: {len(tetrandrine_silicosis)}")
        print(f"最大概率值: {max(probs) if probs else 0}")
    
    # 计算对肝毒性的间接关系概率
    print("计算对肝毒性的间接关系概率...")
    baicalin_hepatotox = calculate_indirect_relation_probability(
        baicalin_relations, hepatotox_relations
    )
    tetrandrine_hepatotox = calculate_indirect_relation_probability(
        tetrandrine_relations, hepatotox_relations
    )
    
    # 调试输出
    print("肝毒性协同作用分析:")
    if not baicalin_hepatotox.empty and 'Indirect_Probability' in baicalin_hepatotox.columns:
        probs = baicalin_hepatotox['Indirect_Probability'].tolist()
        print(f"黄芩苷-肝毒性路径数: {len(baicalin_hepatotox)}")
        print(f"最大概率值: {max(probs) if probs else 0}")
    
    if not tetrandrine_hepatotox.empty and 'Indirect_Probability' in tetrandrine_hepatotox.columns:
        probs = tetrandrine_hepatotox['Indirect_Probability'].tolist()
        print(f"汉防己甲素-肝毒性路径数: {len(tetrandrine_hepatotox)}")
        print(f"最大概率值: {max(probs) if probs else 0}")
    
    # 计算对肾毒性的间接关系概率
    print("计算对肾毒性的间接关系概率...")
    baicalin_nephrotox = calculate_indirect_relation_probability(
        baicalin_relations, nephrotox_relations
    )
    tetrandrine_nephrotox = calculate_indirect_relation_probability(
        tetrandrine_relations, nephrotox_relations
    )
    
    # 调试输出
    print("肾毒性协同作用分析:")
    if not baicalin_nephrotox.empty and 'Indirect_Probability' in baicalin_nephrotox.columns:
        probs = baicalin_nephrotox['Indirect_Probability'].tolist()
        print(f"黄芩苷-肾毒性路径数: {len(baicalin_nephrotox)}")
        print(f"最大概率值: {max(probs) if probs else 0}")
    
    if not tetrandrine_nephrotox.empty and 'Indirect_Probability' in tetrandrine_nephrotox.columns:
        probs = tetrandrine_nephrotox['Indirect_Probability'].tolist()
        print(f"汉防己甲素-肾毒性路径数: {len(tetrandrine_nephrotox)}")
        print(f"最大概率值: {max(probs) if probs else 0}")
    
    # 分析增效和减毒机制
    print("分析协同增效和减毒机制...")
    
    # 初始化结果
    mechanism_types = {
        'silicosis_treatment': {
            'synergy_coefficient': 0.0,
            'synergy_type': '无明显作用'
        },
        'hepatotoxicity_reduction': {
            'synergy_coefficient': 0.0,
            'synergy_type': '无明显作用'
        },
        'nephrotoxicity_reduction': {
            'synergy_coefficient': 0.0,
            'synergy_type': '无明显作用'
        }
    }
    
    # 计算硅肺病治疗协同系数（基于PSR算法）
    if not baicalin_silicosis.empty and not tetrandrine_silicosis.empty:
        # 使用PSR方法计算协同系数
        baicalin_prob = max(baicalin_silicosis['Indirect_Probability']) if not baicalin_silicosis.empty else 0
        tetrandrine_prob = max(tetrandrine_silicosis['Indirect_Probability']) if not tetrandrine_silicosis.empty else 0
        
        # 确保非零概率值
        if len(baicalin_silicosis) > 0 and baicalin_prob < 0.01:
            baicalin_prob = 0.01 + (len(baicalin_silicosis) * 0.001)
        if len(tetrandrine_silicosis) > 0 and tetrandrine_prob < 0.01:
            tetrandrine_prob = 0.01 + (len(tetrandrine_silicosis) * 0.001)
        
        # PSR协同系数计算：几何平均值
        synergy_coefficient = math.sqrt(baicalin_prob * tetrandrine_prob)
        
        # 路径数量调整
        silicosis_path_count = len(baicalin_silicosis) + len(tetrandrine_silicosis)
        path_count_factor = min(0.3, silicosis_path_count / 200)
        synergy_coefficient = synergy_coefficient + path_count_factor
        
        print(f"硅肺病治疗协同系数计算: baicalin_prob={baicalin_prob}, tetrandrine_prob={tetrandrine_prob}, result={synergy_coefficient}")
            
        # 确定协同类型
        if synergy_coefficient > 0.5:
            synergy_type = "强协同作用"
        elif synergy_coefficient > 0.3:
            synergy_type = "协同作用"
        elif synergy_coefficient > 0.1:
            synergy_type = "弱协同作用"
        else:
            synergy_type = "无明显作用"
            
        mechanism_types['silicosis_treatment']['synergy_coefficient'] = synergy_coefficient
        mechanism_types['silicosis_treatment']['synergy_type'] = synergy_type
    
    # 计算肝毒性减轻协同系数
    if not baicalin_hepatotox.empty and not tetrandrine_hepatotox.empty:
        # 使用PSR方法计算减毒协同系数
        baicalin_prob = max(baicalin_hepatotox['Indirect_Probability']) if not baicalin_hepatotox.empty else 0
        tetrandrine_prob = max(tetrandrine_hepatotox['Indirect_Probability']) if not tetrandrine_hepatotox.empty else 0
        
        # 确保非零概率值
        if len(baicalin_hepatotox) > 0 and baicalin_prob < 0.01:
            baicalin_prob = 0.01 + (len(baicalin_hepatotox) * 0.001)
        if len(tetrandrine_hepatotox) > 0 and tetrandrine_prob < 0.01:
            tetrandrine_prob = 0.01 + (len(tetrandrine_hepatotox) * 0.001)
        
        # PSR协同系数计算
        synergy_coefficient = math.sqrt(baicalin_prob * tetrandrine_prob)
        
        # 路径数量调整
        hepatotox_path_count = len(baicalin_hepatotox) + len(tetrandrine_hepatotox)
        path_count_factor = min(0.3, hepatotox_path_count / 200)
        synergy_coefficient = synergy_coefficient + path_count_factor
        
        print(f"肝保护协同系数计算: baicalin_prob={baicalin_prob}, tetrandrine_prob={tetrandrine_prob}, result={synergy_coefficient}")
        
        # 确定协同类型
        if synergy_coefficient > 0.5:
            synergy_type = "强保护作用"
        elif synergy_coefficient > 0.3:
            synergy_type = "保护作用"
        elif synergy_coefficient > 0.1:
            synergy_type = "弱保护作用"
        else:
            synergy_type = "无明显作用"
            
        mechanism_types['hepatotoxicity_reduction']['synergy_coefficient'] = synergy_coefficient
        mechanism_types['hepatotoxicity_reduction']['synergy_type'] = synergy_type
    
    # 计算肾毒性减轻协同系数
    if not baicalin_nephrotox.empty and not tetrandrine_nephrotox.empty:
        # 使用PSR方法计算减毒协同系数
        baicalin_prob = max(baicalin_nephrotox['Indirect_Probability']) if not baicalin_nephrotox.empty else 0
        tetrandrine_prob = max(tetrandrine_nephrotox['Indirect_Probability']) if not tetrandrine_nephrotox.empty else 0
        
        # 确保非零概率值
        if len(baicalin_nephrotox) > 0 and baicalin_prob < 0.01:
            baicalin_prob = 0.01 + (len(baicalin_nephrotox) * 0.001)
        if len(tetrandrine_nephrotox) > 0 and tetrandrine_prob < 0.01:
            tetrandrine_prob = 0.01 + (len(tetrandrine_nephrotox) * 0.001)
        
        # PSR协同系数计算
        synergy_coefficient = math.sqrt(baicalin_prob * tetrandrine_prob)
        
        # 路径数量调整
        nephrotox_path_count = len(baicalin_nephrotox) + len(tetrandrine_nephrotox)
        path_count_factor = min(0.3, nephrotox_path_count / 200)
        synergy_coefficient = synergy_coefficient + path_count_factor
        
        print(f"肾保护协同系数计算: baicalin_prob={baicalin_prob}, tetrandrine_prob={tetrandrine_prob}, result={synergy_coefficient}")
        
        # 确定协同类型
        if synergy_coefficient > 0.5:
            synergy_type = "强保护作用"
        elif synergy_coefficient > 0.3:
            synergy_type = "保护作用"
        elif synergy_coefficient > 0.1:
            synergy_type = "弱保护作用"
        else:
            synergy_type = "无明显作用"
            
        mechanism_types['nephrotoxicity_reduction']['synergy_coefficient'] = synergy_coefficient
        mechanism_types['nephrotoxicity_reduction']['synergy_type'] = synergy_type
    
    # 获取关键协同基因
    therapeutic_genes = get_key_synergy_genes(baicalin_silicosis, tetrandrine_silicosis, common_targets)
    hepatoprotective_genes = get_key_synergy_genes(baicalin_hepatotox, tetrandrine_hepatotox, common_targets)
    nephroprotective_genes = get_key_synergy_genes(baicalin_nephrotox, tetrandrine_nephrotox, common_targets)
    
    # 整合结果
    return {
        'mechanism_types': mechanism_types,
        'therapeutic_genes': therapeutic_genes,
        'hepatoprotective_genes': hepatoprotective_genes,
        'nephroprotective_genes': nephroprotective_genes
    }

def get_key_synergy_genes(drug1_relations, drug2_relations, common_targets, top_n=10):
    """
    获取两种药物协同作用的关键基因（按名称而非ID）
    
    参数:
    drug1_relations - 药物1的间接关系
    drug2_relations - 药物2的间接关系
    common_targets - 共同靶点数据
    top_n - 返回的关键基因数量
    
    返回:
    关键基因列表，包含基因名称和重要性评分
    """
    import pandas as pd
    
    # 初始化结果
    key_genes = {}
    
    # 如果任一关系为空，返回空列表
    if drug1_relations.empty or drug2_relations.empty:
        return []
    
    # 确保Path_Details列存在
    if 'Path_Details' not in drug1_relations.columns or 'Path_Details' not in drug2_relations.columns:
        return []
    
    # 处理第一种药物的路径
    for _, row in drug1_relations.iterrows():
        if 'Path_Details' in row and row['Path_Details']:
            for path in row['Path_Details']:
                if 'intermediate_target_id' in path and 'intermediate_target_name' in path:
                    target_id = path['intermediate_target_id']
                    target_name = path['intermediate_target_name']
                    
                    # 如果靶点名称是Unknown，尝试从common_targets获取
                    if target_name == "Unknown":
                        target_name_rows = common_targets[common_targets['Target_ID'] == target_id]
                        if not target_name_rows.empty and 'Target_Name' in target_name_rows.columns:
                            target_name = target_name_rows['Target_Name'].iloc[0]
                    
                    # 初始化或更新基因信息
                    if target_id not in key_genes:
                        key_genes[target_id] = {
                            'name': target_name,
                            'importance_score': 0,
                            'drug1_effect': 0,
                            'drug2_effect': 0,
                            'path_count': 0
                        }
                    
                    # 更新重要性得分和药物1效应
                    path_prob = path.get('path_probability', 0)
                    key_genes[target_id]['importance_score'] += path_prob
                    key_genes[target_id]['drug1_effect'] = 1  # 药物1对该靶点有作用
                    key_genes[target_id]['path_count'] += 1
    
    # 处理第二种药物的路径
    for _, row in drug2_relations.iterrows():
        if 'Path_Details' in row and row['Path_Details']:
            for path in row['Path_Details']:
                if 'intermediate_target_id' in path and 'intermediate_target_name' in path:
                    target_id = path['intermediate_target_id']
                    target_name = path['intermediate_target_name']
                    
                    # 如果靶点名称是Unknown，尝试从common_targets获取
                    if target_name == "Unknown":
                        target_name_rows = common_targets[common_targets['Target_ID'] == target_id]
                        if not target_name_rows.empty and 'Target_Name' in target_name_rows.columns:
                            target_name = target_name_rows['Target_Name'].iloc[0]
                    
                    # 初始化或更新基因信息
                    if target_id not in key_genes:
                        key_genes[target_id] = {
                            'name': target_name,
                            'importance_score': 0,
                            'drug1_effect': 0,
                            'drug2_effect': 0,
                            'path_count': 0
                        }
                    
                    # 更新重要性得分和药物2效应
                    path_prob = path.get('path_probability', 0)
                    key_genes[target_id]['importance_score'] += path_prob
                    key_genes[target_id]['drug2_effect'] = 1  # 药物2对该靶点有作用
                    key_genes[target_id]['path_count'] += 1
    
    # 转换为DataFrame并排序
    genes_df = pd.DataFrame([
        {
            'target_id': target_id,
            'gene_name': info['name'],
            'importance_score': info['importance_score'],
            'is_synergy_target': info['drug1_effect'] > 0 and info['drug2_effect'] > 0,
            'path_count': info['path_count']
        }
        for target_id, info in key_genes.items()
    ])
    
    # 筛选两种药物都作用的靶点，并按重要性排序
    synergy_genes = genes_df[genes_df['is_synergy_target'] == True].sort_values(
        by=['importance_score', 'path_count'], 
        ascending=False
    )
    
    # 获取前N个关键基因
    top_genes = synergy_genes.head(top_n)
    
    # 转换为列表格式
    result = []
    for _, row in top_genes.iterrows():
        result.append({
            'gene_id': row['target_id'],
            'gene_name': row['gene_name'],
            'importance_score': row['importance_score'],
            'path_count': row['path_count']
        })
    
    return result



def get_significant_targets(source_indirect_relations, target_indirect_relations, common_targets, top_n=10):
    """
    提取最重要的共同靶点基因（按名称而非ID）
    
    Args:
        source_indirect_relations: 药物1的间接关系
        target_indirect_relations: 药物2的间接关系
        common_targets: 共同靶点数据
        top_n: 返回的重要靶点数量
        
    Returns:
        重要靶点列表，包含基因名称和重要性评分
    """
    import pandas as pd
    
    # 初始化结果
    significant_targets = []
    
    # 如果任一关系为空，返回空列表
    if source_indirect_relations.empty or target_indirect_relations.empty:
        return significant_targets
    
    # 确保Path_Details列存在
    if 'Path_Details' not in source_indirect_relations.columns or 'Path_Details' not in target_indirect_relations.columns:
        return significant_targets
    
    # 创建目标基因重要性字典
    target_importance = {}
    
    # 处理第一种药物的中间靶点
    for _, row in source_indirect_relations.iterrows():
        if 'Path_Details' in row and row['Path_Details']:
            for path in row['Path_Details']:
                if 'intermediate_target_id' in path and 'intermediate_target_name' in path:
                    target_id = path['intermediate_target_id']
                    target_name = path['intermediate_target_name']
                    
                    # 如果靶点名称是Unknown，尝试从common_targets获取
                    if target_name == "Unknown" and target_id in common_targets:
                        # 假设common_targets包含ID到名称的映射
                        if 'Target_Name' in common_targets[common_targets['Target_ID'] == target_id].columns:
                            target_name = common_targets[common_targets['Target_ID'] == target_id]['Target_Name'].iloc[0]
                    
                    # 计算该靶点的重要性
                    path_prob = path.get('path_probability', 0)
                    
                    # 更新靶点重要性
                    if target_id not in target_importance:
                        target_importance[target_id] = {
                            'name': target_name,
                            'importance': 0,
                            'paths': 0
                        }
                    
                    target_importance[target_id]['importance'] += path_prob
                    target_importance[target_id]['paths'] += 1
    
    # 处理第二种药物的中间靶点
    for _, row in target_indirect_relations.iterrows():
        if 'Path_Details' in row and row['Path_Details']:
            for path in row['Path_Details']:
                if 'intermediate_target_id' in path and 'intermediate_target_name' in path:
                    target_id = path['intermediate_target_id']
                    target_name = path['intermediate_target_name']
                    
                    # 如果靶点名称是Unknown，尝试从common_targets获取
                    if target_name == "Unknown" and target_id in common_targets:
                        if 'Target_Name' in common_targets[common_targets['Target_ID'] == target_id].columns:
                            target_name = common_targets[common_targets['Target_ID'] == target_id]['Target_Name'].iloc[0]
                    
                    # 只处理两种药物的共同靶点
                    if target_id in target_importance:
                        path_prob = path.get('path_probability', 0)
                        target_importance[target_id]['importance'] += path_prob
                        target_importance[target_id]['paths'] += 1
                        
                        # 如果有更好的名称，更新它
                        if target_name != "Unknown" and target_importance[target_id]['name'] == "Unknown":
                            target_importance[target_id]['name'] = target_name
    
    # 转换为DataFrame并排序
    targets_df = pd.DataFrame([
        {
            'target_id': target_id,
            'target_name': info['name'],
            'importance': info['importance'],
            'path_count': info['paths']
        }
        for target_id, info in target_importance.items()
    ])
    
    if not targets_df.empty:
        # 按重要性排序
        targets_df = targets_df.sort_values(by='importance', ascending=False)
        
        # 获取前N个靶点
        top_targets = targets_df.head(top_n)
        
        # 转换为结果格式
        for _, row in top_targets.iterrows():
            significant_targets.append({
                'target_id': row['target_id'],
                'target_name': row['target_name'],
                'importance': row['importance'],
                'path_count': row['path_count']
            })
    
    return significant_targets
