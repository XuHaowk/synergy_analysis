"""
概率推理模块：实现PSR算法
"""

import pandas as pd
import numpy as np
import os
import config

def calculate_direct_relation_probability(entity_relations):
    """
    计算直接关系概率:  PA→B = 1 - ∏(1 - piA→B)
    
    参数:
    entity_relations - 包含实体关系的DataFrame
    
    返回:
    包含计算概率的DataFrame
    """
    # 按来源和目标ID组合分组，计算总体概率
    relation_probs = entity_relations.copy()
    
    # 确保概率列是数值型
    relation_probs['Probability'] = pd.to_numeric(relation_probs['Probability'], errors='coerce')
    
    # 创建实体对标识符
    relation_probs['entity_pair'] = relation_probs['Source_ID'].astype(str) + '_' + relation_probs['Target_ID'].astype(str)
    
    # 计算每个实体对的综合概率
    result = pd.DataFrame()
    for pair, group in relation_probs.groupby('entity_pair'):
        # 提取源和目标信息
        source_id = group['Source_ID'].iloc[0]
        source_name = group['Source_Name'].iloc[0]
        target_id = group['Target_ID'].iloc[0]
        target_name = group['Target_Name'].iloc[0]
        
        # 获取所有概率值
        probabilities = group['Probability'].values
        
        # 应用PSR公式: P = 1 - ∏(1 - pi)
        composite_prob = 1 - np.prod([1 - p for p in probabilities])
        
        # 关系类型
        relation_types = group['Type'].unique()
        primary_relation = group['Type'].iloc[0]  # 使用第一个关系类型作为主要类型
        
        # 确定关系方向 (正相关或负相关)
        direction = 'positive'
        for rel_type in relation_types:
            if rel_type in ['Negative_Correlation']:
                direction = 'negative'
                break
        
        # 添加到结果
        result = pd.concat([result, pd.DataFrame({
            'Source_ID': [source_id],
            'Source_Name': [source_name],
            'Target_ID': [target_id],
            'Target_Name': [target_name],
            'Composite_Probability': [composite_prob],
            'Primary_Relation': [primary_relation],
            'Direction': [direction],
            'Relation_Count': [len(probabilities)]
        })], ignore_index=True)
    
    return result

def calculate_indirect_relation_probability(source_target_relations, target_target_relations):
    """
    计算间接关系概率，基于PSR (Probabilistic Semantic Relationships)算法
    
    参数:
    source_target_relations - 药物→基因的关系数据
    target_target_relations - 基因→疾病的关系数据
    
    返回:
    药物→疾病的间接关系DataFrame
    """
    import pandas as pd
    import numpy as np
    
    # 检查必要的列是否存在，如果不存在则添加
    source_target_relations = source_target_relations.copy()
    if 'Source_Type' not in source_target_relations.columns:
        source_target_relations['Source_Type'] = 'Chemical'
    if 'Target_Type' not in source_target_relations.columns:
        source_target_relations['Target_Type'] = 'Gene'
    
    target_target_relations = target_target_relations.copy()
    if 'Source_Type' not in target_target_relations.columns:
        target_target_relations['Source_Type'] = 'Gene'
    if 'Target_Type' not in target_target_relations.columns:
        target_target_relations['Target_Type'] = 'Disease'
    
    # 确保概率列存在
    if 'Probability' not in source_target_relations.columns:
        if 'Score' in source_target_relations.columns:
            source_target_relations['Probability'] = source_target_relations['Score']
        else:
            source_target_relations['Probability'] = 0.5  # 默认值
    
    if 'Probability' not in target_target_relations.columns:
        if 'Score' in target_target_relations.columns:
            target_target_relations['Probability'] = target_target_relations['Score']
        else:
            target_target_relations['Probability'] = 0.5  # 默认值
    
    # 将所有ID转换为字符串类型
    source_target_relations['Source_ID'] = source_target_relations['Source_ID'].astype(str)
    source_target_relations['Target_ID'] = source_target_relations['Target_ID'].astype(str)
    target_target_relations['Source_ID'] = target_target_relations['Source_ID'].astype(str)
    target_target_relations['Target_ID'] = target_target_relations['Target_ID'].astype(str)
    
    # 初始化结果
    indirect_relations = []
    
    # 获取所有药物靶点（考虑双向关系）
    drug_targets = set()
    for _, row in source_target_relations.iterrows():
        if row['Source_Type'] == 'Chemical' and row['Target_Type'] == 'Gene':
            drug_targets.add(row['Target_ID'])
        elif row['Source_Type'] == 'Gene' and row['Target_Type'] == 'Chemical':
            drug_targets.add(row['Source_ID'])
    
    # 获取所有疾病靶点（考虑双向关系）
    disease_targets = set()
    for _, row in target_target_relations.iterrows():
        if row['Source_Type'] == 'Gene' and row['Target_Type'] == 'Disease':
            disease_targets.add(row['Source_ID'])
        elif row['Source_Type'] == 'Disease' and row['Target_Type'] == 'Gene':
            disease_targets.add(row['Target_ID'])
    
    # 寻找药物和疾病的共同靶点
    intermediate_targets = drug_targets.intersection(disease_targets)
    print(f"找到{len(intermediate_targets)}个共同的中间靶点")
    
    # 如果没有共同靶点，返回空DataFrame
    if len(intermediate_targets) == 0:
        return pd.DataFrame()
    
    # 对每个源和目标实体对计算间接关系
    source_ids = source_target_relations['Source_ID'].unique()
    target_ids = target_target_relations['Target_ID'].unique()
    
    for source_id in source_ids:
        # 只处理类型为Chemical的源实体
        source_chemical_rows = source_target_relations[
            (source_target_relations['Source_ID'] == source_id) & 
            (source_target_relations['Source_Type'] == 'Chemical')
        ]
        if len(source_chemical_rows) == 0:
            continue
            
        for target_id in target_ids:
            # 只处理类型为Disease的目标实体
            target_disease_rows = target_target_relations[
                (target_target_relations['Target_ID'] == target_id) & 
                (target_target_relations['Target_Type'] == 'Disease')
            ]
            if len(target_disease_rows) == 0:
                continue
                
            # 跳过自反关系
            if source_id == target_id:
                continue
                
            # 初始化路径概率计算
            path_probabilities = []
            path_details = []
            
            # 获取源实体和目标实体名称
            source_name = "Unknown"
            if 'Source_Name' in source_chemical_rows.columns and not source_chemical_rows.empty:
                source_name = source_chemical_rows['Source_Name'].iloc[0]
            
            target_name = "Unknown"
            if 'Target_Name' in target_disease_rows.columns and not target_disease_rows.empty:
                target_name = target_disease_rows['Target_Name'].iloc[0]
            
            # 对每个中间靶点计算路径概率（实现PSR公式2）
            for inter_target in intermediate_targets:
                # 获取药物→基因关系
                drug_to_gene = source_target_relations[
                    (source_target_relations['Source_ID'] == source_id) & 
                    (source_target_relations['Target_ID'] == inter_target) &
                    (source_target_relations['Source_Type'] == 'Chemical') &
                    (source_target_relations['Target_Type'] == 'Gene')
                ]
                
                # 获取基因→疾病关系
                gene_to_disease = target_target_relations[
                    (target_target_relations['Source_ID'] == inter_target) & 
                    (target_target_relations['Target_ID'] == target_id) &
                    (target_target_relations['Source_Type'] == 'Gene') &
                    (target_target_relations['Target_Type'] == 'Disease')
                ]
                
                # 如果两个关系都存在，计算路径概率
                if not drug_to_gene.empty and not gene_to_disease.empty:
                    # 获取概率
                    drug_gene_prob = float(drug_to_gene['Probability'].iloc[0])
                    gene_disease_prob = float(gene_to_disease['Probability'].iloc[0])
                    
                    # 确定关系方向（正/负相关）
                    drug_gene_dir = "positive"
                    if 'Type' in drug_to_gene.columns and drug_to_gene['Type'].iloc[0] == "Negative_Correlation":
                        drug_gene_dir = "negative"
                    
                    gene_disease_dir = "positive"
                    if 'Type' in gene_to_disease.columns and gene_to_disease['Type'].iloc[0] == "Negative_Correlation":
                        gene_disease_dir = "negative"
                    
                    # PSR算法：计算路径概率（公式2）
                    path_prob = drug_gene_prob * gene_disease_prob
                    
                    # 根据关系方向调整概率
                    if drug_gene_dir == gene_disease_dir:
                        # 同向效应权重提高
                        path_prob = 0.8 * path_prob
                    else:
                        # 异向效应权重降低
                        path_prob = 0.4 * path_prob
                    
                    # 应用最小阈值
                    if path_prob < 0.001:
                        path_prob = 0.001
                    
                    path_probabilities.append(path_prob)
                    
                    # 获取中间靶点名称
                    gene_name = "Unknown"
                    if 'Target_Name' in drug_to_gene.columns:
                        gene_name = drug_to_gene['Target_Name'].iloc[0]
                    
                    # 记录路径详情
                    path_details.append({
                        'intermediate_target_id': inter_target,
                        'intermediate_target_name': gene_name,
                        'drug_gene_prob': drug_gene_prob,
                        'drug_gene_dir': drug_gene_dir,
                        'gene_disease_prob': gene_disease_prob,
                        'gene_disease_dir': gene_disease_dir,
                        'path_probability': path_prob
                    })
            
            # 如果存在有效路径，使用PSR公式3计算总体概率
            if path_probabilities:
                # PSR公式3: P(A→C) = 1 - ∏[1 - P(A→B→C)]
                indirect_prob = 1 - np.prod([1 - p for p in path_probabilities])
                
                # 路径数量补偿：考虑路径数量对总体概率的贡献
                path_count_factor = min(0.5, len(path_probabilities) / 100)
                indirect_prob = indirect_prob + (path_count_factor * 0.1)
                
                # 确保概率有意义
                if indirect_prob < 0.01:
                    indirect_prob = 0.01 + (len(path_probabilities) * 0.001)
                
                # 添加到结果
                indirect_relations.append({
                    'Source_ID': source_id,
                    'Source_Name': source_name,
                    'Target_ID': target_id,
                    'Target_Name': target_name,
                    'Indirect_Probability': indirect_prob,
                    'Path_Count': len(path_probabilities),
                    'Path_Details': path_details
                })
    
    return pd.DataFrame(indirect_relations) if indirect_relations else pd.DataFrame()

def calculate_synergistic_effect(drug1_disease_relation, drug2_disease_relation, common_targets):
    """
    计算两种药物对疾病的协同增效或减毒效果
    """
    # 初始化结果
    synergy_results = {}
    
    # 提取关键概率
    drug1_prob = drug1_disease_relation.get('Indirect_Probability', 0) if isinstance(drug1_disease_relation, dict) else 0
    drug2_prob = drug2_disease_relation.get('Indirect_Probability', 0) if isinstance(drug2_disease_relation, dict) else 0
    
    # 计算理论上的独立作用概率
    independent_effect = 1 - (1 - drug1_prob) * (1 - drug2_prob)
    
    # 计算协同指数
    common_targets_effect = 0
    
    # 检查是否有共同靶点和路径详情
    has_path_details = (isinstance(drug1_disease_relation, dict) and 'Path_Details' in drug1_disease_relation and
                        isinstance(drug2_disease_relation, dict) and 'Path_Details' in drug2_disease_relation)
    
    if common_targets is not None and not common_targets.empty and has_path_details:
        # 获取每个共同靶点的作用强度
        drug1_targets = {r['intermediate_target_id']: r['path_probability'] 
                         for r in drug1_disease_relation['Path_Details']
                         if r['intermediate_target_id'] in common_targets['Target_ID'].values}
        
        drug2_targets = {r['intermediate_target_id']: r['path_probability'] 
                         for r in drug2_disease_relation['Path_Details']
                         if r['intermediate_target_id'] in common_targets['Target_ID'].values}
        
        # 计算共同靶点的协同效应
        for target_id in set(drug1_targets.keys()) & set(drug2_targets.keys()):
            common_targets_effect += drug1_targets[target_id] * drug2_targets[target_id]
    
    # 计算协同系数 (避免除以零)
    synergy_coefficient = common_targets_effect / independent_effect if independent_effect > 0 else 0
    
    # 确定协同类型
    if synergy_coefficient > 1.2:
        synergy_type = "强协同作用"
    elif synergy_coefficient > 1.0:
        synergy_type = "协同作用"
    elif synergy_coefficient >= 0.8:
        synergy_type = "加性作用"
    else:
        synergy_type = "拮抗作用" if synergy_coefficient > 0 else "无明显作用"
    
    # 记录结果
    synergy_results = {
        'Drug1_Effect': drug1_prob,
        'Drug2_Effect': drug2_prob,
        'Independent_Effect': independent_effect,
        'Common_Targets_Effect': common_targets_effect,
        'Synergy_Coefficient': synergy_coefficient,
        'Synergy_Type': synergy_type
    }
    
    return synergy_results



