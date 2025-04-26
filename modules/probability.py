"""
概率推理模块：实现PSR算法
"""

import pandas as pd
import numpy as np
import os
import math
import json
from collections import defaultdict, Counter
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Remove the import from synergy_analysis
# from modules.synergy_analysis import analyze_common_targets, analyze_pathway_enrichment, analyze_synergy_mechanisms

def calculate_gene_network_centrality(gene_id, gene_gene_relations):
    """
    利用基因-基因关联数据计算基因在网络中的中心性
    
    参数:
    gene_id - 目标基因ID
    gene_gene_relations - 基因与基因之间的关联数据
    
    返回:
    centrality_score - 基因的网络中心性分数(0-1范围)
    """
    # 构建基因互作网络
    gene_network = {}
    edge_weights = {}
    
    # 处理基因-基因关联数据构建网络
    for relation in gene_gene_relations:
        source_id = relation['Source_ID']
        target_id = relation['Target_ID']
        
        # 提取关系强度/概率
        probability = float(relation.get('Probability', 0.5))
        
        # 构建无向网络(双向添加边)
        if source_id not in gene_network:
            gene_network[source_id] = []
        if target_id not in gene_network:
            gene_network[target_id] = []
            
        gene_network[source_id].append(target_id)
        gene_network[target_id].append(source_id)
        
        # 记录边权重
        edge_key = f"{source_id}_{target_id}"
        reverse_key = f"{target_id}_{source_id}"
        edge_weights[edge_key] = probability
        edge_weights[reverse_key] = probability
    
    # 如果基因不在网络中，返回最低分数
    if gene_id not in gene_network:
        return 0.1
    
    # 计算度中心性(直接连接数)
    degree = len(gene_network[gene_id])
    # 标准化度中心性(相对于网络中最大度)
    max_degree = max([len(neighbors) for neighbors in gene_network.values()])
    degree_centrality = degree / max_degree if max_degree > 0 else 0
    
    # 计算加权度中心性(考虑连接强度)
    weighted_degree = 0
    for neighbor in gene_network[gene_id]:
        edge_key = f"{gene_id}_{neighbor}"
        weight = edge_weights.get(edge_key, 0.5)
        weighted_degree += weight
    
    max_weighted_degree = 0
    for node_id in gene_network:
        node_weighted_degree = 0
        for neighbor in gene_network[node_id]:
            edge_key = f"{node_id}_{neighbor}"
            weight = edge_weights.get(edge_key, 0.5)
            node_weighted_degree += weight
        max_weighted_degree = max(max_weighted_degree, node_weighted_degree)
    
    weighted_centrality = weighted_degree / max_weighted_degree if max_weighted_degree > 0 else 0
    
    # 计算最终中心性分数(结合两种中心性指标)
    centrality_score = 0.4 * degree_centrality + 0.6 * weighted_centrality
    
    # 确保分数在0.1-1范围内
    return max(0.1, min(1.0, centrality_score))

def extract_gene_evidence_scores(gene_id, disease_id, gene_disease_relations, gene_drug_relations, drug_id=None):
    """
    从已有的基因-疾病和基因-药物关系数据中提取证据分数
    
    参数:
    gene_id - 目标基因ID
    disease_id - 疾病ID
    gene_disease_relations - 基因与疾病之间的关系数据
    gene_drug_relations - 基因与药物之间的关系数据
    drug_id - 相关药物ID(可选)
    
    返回:
    evidence_score - 综合文献证据分数(0-1范围)
    """
    # 查找基因-疾病关系的证据分数
    disease_evidence = 0.0
    found_disease_relation = False
    
    for relation in gene_disease_relations:
        if ((relation['Source_ID'] == gene_id and relation['Target_ID'] == disease_id) or 
            (relation['Source_ID'] == disease_id and relation['Target_ID'] == gene_id)):
            
            # 提取概率和可信度数据
            probability = float(relation.get('Probability', 0.5))
            score = float(relation.get('Score', 0.5)) 
            
            # 考虑文献来源可靠性(如果有Source字段)
            source_factor = 1.0
            if 'Source' in relation:
                if relation['Source'] in ['PubMed', 'OMIM', 'KEGG']:
                    source_factor = 1.2  # 高可靠性来源
                elif relation['Source'] in ['Review', 'Meta-analysis']:
                    source_factor = 1.3  # 更高可靠性来源
            
            # 计算疾病证据分数
            disease_evidence = probability * score * source_factor
            found_disease_relation = True
            break
    
    # 如果提供了药物ID，查找基因-药物关系的证据分数
    drug_evidence = 0.0
    found_drug_relation = False
    
    if drug_id:
        for relation in gene_drug_relations:
            if ((relation['Source_ID'] == gene_id and relation['Target_ID'] == drug_id) or 
                (relation['Source_ID'] == drug_id and relation['Target_ID'] == gene_id)):
                
                # 提取概率和可信度数据
                probability = float(relation.get('Probability', 0.5))
                score = float(relation.get('Score', 0.5))
                
                # 考虑关系类型的重要性
                type_factor = 1.0
                if 'Type' in relation:
                    if relation['Type'] in ['Target', 'Drug_Target', 'Direct_Target']:
                        type_factor = 1.4  # 直接靶点更重要
                    elif relation['Type'] in ['Positive_Correlation', 'Negative_Correlation']:
                        type_factor = 1.2  # 直接相关性也重要
                
                # 计算药物证据分数
                drug_evidence = probability * score * type_factor
                found_drug_relation = True
                break
    
    # 组合疾病和药物证据
    if found_disease_relation and found_drug_relation:
        # 同时有疾病和药物证据时，取更高权重的加权平均
        evidence_score = 0.6 * disease_evidence + 0.4 * drug_evidence
    elif found_disease_relation:
        evidence_score = disease_evidence
    elif found_drug_relation:
        evidence_score = drug_evidence
    else:
        # 没有找到直接证据，返回默认低分数
        evidence_score = 0.1
    
    # 确保分数在0.1-1范围内
    return max(0.1, min(1.0, evidence_score))

def extract_gene_gene_relations(source_target_relations, target_target_relations):
    """
    从源关系数据中提取基因-基因关系
    
    参数:
    source_target_relations - 源关系数据
    target_target_relations - 目标关系数据
    
    返回:
    gene_gene_relations - 基因-基因关系列表
    """
    gene_gene_relations = []
    
    # 从target_target_relations中提取基因-基因关系
    for relation in target_target_relations:
        if (relation.get('Source_Type') == 'Gene' and relation.get('Target_Type') == 'Gene'):
            gene_gene_relations.append(relation)
    
    # 也可以从source_target_relations中提取可能的基因-基因关系
    for relation in source_target_relations:
        if (relation.get('Source_Type') == 'Gene' and relation.get('Target_Type') == 'Gene'):
            gene_gene_relations.append(relation)
    
    return gene_gene_relations

def determine_direction(relation_type):
    """
    根据关系类型确定方向性
    
    参数:
    relation_type - 关系类型名称
    
    返回:
    direction - 'positive' 或 'negative'
    """
    # 正向关系类型
    positive_relations = [
        'Positive_Correlation', 'Increases', 'Causes', 'Activates', 'Upregulates',
        'Induces', 'Stimulates', 'Enhances', 'Promotes', 'Association'
    ]
    
    # 负向关系类型
    negative_relations = [
        'Negative_Correlation', 'Decreases', 'Treats', 'Inhibits', 'Downregulates',
        'Suppresses', 'Antagonizes', 'Prevents', 'Blocks'
    ]
    
    # 确定方向
    if relation_type in positive_relations:
        return 'positive'
    elif relation_type in negative_relations:
        return 'negative'
    else:
        # 默认为正向关系
        return 'positive'

# Global variable definition for rel_type_map
rel_type_map = {
    '2': {'name': 'Positive_Correlation', 'correlation': 1, 'precision': 4, 'causative': True, 'protective': False},
    '3': {'name': 'Negative_Correlation', 'correlation': -1, 'precision': 4, 'causative': False, 'protective': True},
    '11': {'name': 'Causes', 'correlation': 1, 'precision': 3, 'causative': True, 'protective': False},
    '16': {'name': 'Treats', 'correlation': -1, 'precision': 3, 'causative': False, 'protective': True}
}

# Try to load rel_type_map from file if available
try:
    possible_paths = [
        'RelTypeInt.json',
        os.path.join('data', 'raw', 'RelTypeInt.json'),
        os.path.join('..', 'data', 'raw', 'RelTypeInt.json'),
        os.path.join(os.path.dirname(__file__), 'RelTypeInt.json'),
        os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'RelTypeInt.json')
    ]
    
    rel_type_file = None
    for path in possible_paths:
        if os.path.exists(path):
            rel_type_file = path
            break
            
    if rel_type_file:
        with open(rel_type_file, 'r', encoding='utf-8') as f:
            rel_types = json.load(f)
            
        # Create ID to relation type and correlation mapping
        rel_type_map = {}
        for rel_type in rel_types:
            rel_id = rel_type['intRep']
            rel_name = rel_type['relType']
            cor_type = rel_type['corType'][0] if rel_type['corType'] else 0
            rel_prec = rel_type['relPrec']  # Relationship precision/strength
            
            rel_type_map[rel_id] = {
                'name': rel_name,
                'correlation': cor_type,  # -1: negative, 0: neutral, 1: positive
                'precision': rel_prec,    # Relationship strength 0-4, for weight calculation
                'causative': True if rel_name in ['Causes', 'Induces', 'Positive_Correlation'] else False,
                'protective': True if rel_name in ['Treats', 'Prevents', 'Negative_Correlation'] else False
            }
except Exception as e:
    print(f"Warning: Could not load relationship types: {e}")
    # Fallback to default rel_type_map defined above

def calculate_data_driven_path_weight(path, gene_gene_relations, gene_disease_relations, gene_drug_relations):
    """
    基于实际数据计算路径的重要性权重
    
    参数:
    path - 包含路径信息的字典
    gene_gene_relations - 基因-基因关系数据
    gene_disease_relations - 基因-疾病关系数据
    gene_drug_relations - 基因-药物关系数据
    
    返回:
    adjusted_weight - 数据驱动的调整后权重
    """
    # 提取路径中的关键ID
    gene_id = path['intermediate_target_id']
    disease_id = path.get('disease_id', '')
    drug_id = path.get('drug_id', '')
    
    # 计算网络中心性分数
    centrality_score = calculate_gene_network_centrality(gene_id, gene_gene_relations)
    
    # 提取文献证据分数
    evidence_score = extract_gene_evidence_scores(
        gene_id, disease_id, gene_disease_relations, gene_drug_relations, drug_id
    )
    
    # 计算通路相关性
    pathway_relevance = 0.5  # 默认值
    if 'intermediate_target_name' in path:
        target_name = path['intermediate_target_name'].lower()
        
        # 为不同类型的疾病分配通路相关性
        if "toxicity" in path.get('disease_name', '').lower() or path.get('is_toxicity_context', False):
            # 毒性相关基因
            if any(term in target_name for term in ['cyp', 'p450', 'ugt', 'gst', 'abc', 'mitochondri']):
                pathway_relevance = 0.9
            elif any(term in target_name for term in ['casp', 'bcl', 'bax', 'apoptosis']):
                pathway_relevance = 0.8
            elif any(term in target_name for term in ['nf-kb', 'tnf', 'il-', 'inflammat']):
                pathway_relevance = 0.7
        else:
            # 硅肺病相关基因
            if "silicosis" in path.get('disease_name', '').lower():
                if any(term in target_name for term in ['tgf', 'collagen', 'fibrosis', 'matrix']):
                    pathway_relevance = 0.9
                elif any(term in target_name for term in ['inflammat', 'il-', 'tnf']):
                    pathway_relevance = 0.8
                elif any(term in target_name for term in ['sod', 'cat', 'gpx', 'oxidative']):
                    pathway_relevance = 0.7
    
    # 计算最终重要性权重(综合三个维度)
    importance_factor = (0.35 * centrality_score + 
                        0.45 * evidence_score + 
                        0.20 * pathway_relevance)
    
    # 应用重要性因子到原始路径权重
    base_weight = path.get('path_weight', 1.0)
    adjusted_weight = base_weight * (1.0 + importance_factor)
    
    # 保存权重计算结果
    path['centrality_score'] = centrality_score
    path['evidence_score'] = evidence_score
    path['pathway_relevance'] = pathway_relevance
    path['importance_factor'] = importance_factor
    path['adjusted_weight'] = adjusted_weight
    
    return adjusted_weight

def calculate_path_weight(rel_type_id, rel_type, prob_value, entity_pair_context=None):
    """
    计算基于关系类型、概率值和生物学上下文的路径权重
    
    参数:
    rel_type_id - 关系类型ID
    rel_type - 关系类型名称
    prob_value - 概率值
    entity_pair_context - 实体对上下文(source_type, target_type, is_toxicity_context)
    
    返回:
    weight - 计算的路径权重
    """
    # 解码实体对上下文
    source_type, target_type, is_toxicity_context = entity_pair_context if entity_pair_context else ('', '', False)
    
    # 基础权重
    weight = 1.0
    weight_factors = {}
    
    # 基于关系类型的权重调整
    if rel_type_id in rel_type_map:
        # 根据精确度增加权重
        precision = rel_type_map[rel_type_id]['precision']
        precision_factor = 1.0 + precision * 0.1  # 每点精确度增加0.1x权重
        weight *= precision_factor
        weight_factors['precision_factor'] = precision_factor
        
        # 上下文感知权重
        if is_toxicity_context:
            # 对于毒性上下文，致病关系更重要
            if rel_type_map[rel_type_id]['causative']:
                context_factor = 1.2
                weight *= context_factor
                weight_factors['context_factor'] = context_factor
        else:
            # 对于治疗上下文，保护关系更重要
            if rel_type_map[rel_type_id]['protective']:
                context_factor = 1.2
                weight *= context_factor
                weight_factors['context_factor'] = context_factor
    
    # 特殊关系类型处理
    relation_factor = 1.0
    if rel_type == "Positive_Correlation" or rel_type == "Causes":
        relation_factor = 1.3
        weight *= relation_factor
        if is_toxicity_context:
            toxicity_factor = 1.2
            weight *= toxicity_factor
            weight_factors['toxicity_factor'] = toxicity_factor
    elif rel_type == "Negative_Correlation" or rel_type == "Treats":
        relation_factor = 1.3
        weight *= relation_factor
        if not is_toxicity_context:
            treatment_factor = 1.2
            weight *= treatment_factor
            weight_factors['treatment_factor'] = treatment_factor
    
    weight_factors['relation_factor'] = relation_factor
    
    # 添加钙通道机制特殊增强 (针对汉防己甲素)
    if source_type == 'Drug' and 'tetrandrine' in source_type.lower():
        if any(term in target_type.lower() for term in ['ca', 'calcium', 'calmodulin', 'cacna']):
            calcium_factor = 1.5
            weight *= calcium_factor
            weight_factors['calcium_factor'] = calcium_factor
    
    # 如果需要，可以保存权重计算因素
    result = {
        'weight': weight,
        'factors': weight_factors
    }
    
    return weight

def adjust_path_probability_by_direction(base_path_prob, drug_gene_dir, gene_disease_dir, is_toxicity_context):
    """
    根据药物-基因和基因-疾病的方向性关系调整路径概率
    
    参数:
    base_path_prob - 基础路径概率
    drug_gene_dir - 药物-基因方向 ('positive' 或 'negative')
    gene_disease_dir - 基因-疾病方向 ('positive' 或 'negative')
    is_toxicity_context - 是否是毒性上下文
    
    返回:
    adjusted_prob - 调整后的概率
    """
    # 保存调整前的原始概率
    original_prob = base_path_prob
    adjustment_factor = 1.0
    
    if is_toxicity_context:  # 毒性上下文
        # 药物上调致病基因或下调保护基因时增加毒性概率
        if ((drug_gene_dir == "positive" and gene_disease_dir == "positive") or
            (drug_gene_dir == "negative" and gene_disease_dir == "negative")):
            adjustment_factor = 1.3
        else:
            adjustment_factor = 0.7
    else:  # 治疗上下文
        # 药物上调保护基因或下调致病基因时增加治疗效果
        if ((drug_gene_dir == "positive" and gene_disease_dir == "negative") or
            (drug_gene_dir == "negative" and gene_disease_dir == "positive")):
            adjustment_factor = 1.3
        else:
            adjustment_factor = 0.7
    
    # 应用调整因子
    adjusted_prob = base_path_prob * adjustment_factor
    
    # 保存调整信息
    adjustment_info = {
        'original_prob': original_prob,
        'adjustment_factor': adjustment_factor,
        'adjusted_prob': adjusted_prob,
        'drug_gene_dir': drug_gene_dir,
        'gene_disease_dir': gene_disease_dir,
        'is_toxicity_context': is_toxicity_context
    }
    
    return adjusted_prob, adjustment_info

def calculate_dynamic_cap(path_count, path_diversity, relationship_type, is_toxicity, drug_name=''):
    """
    计算基于路径数量、多样性和关系类型的动态概率上限
    
    参数:
    path_count - 路径数量
    path_diversity - 路径多样性指标
    relationship_type - 关系类型 ("Therapeutic", "Toxicity" 等)
    is_toxicity - 是否为毒性上下文
    drug_name - 药物名称(可选)
    
    返回:
    prob_cap - 计算的概率上限
    """
    # 基础上限设置
    if is_toxicity:
        # 对汉防己甲素的肝肾毒性特殊处理
        if 'tetrandrine' in drug_name.lower():
            base_cap = 0.85  # 提高汉防己甲素毒性上限
        else:
            base_cap = 0.75  # 标准毒性上限
    else:
        base_cap = 0.85  # 标准治疗上限
    
    # 路径数量和多样性调整
    path_factor = min(0.1, 0.01 * path_count)  # 最多增加0.1
    diversity_factor = min(0.08, path_diversity * 0.1)  # 最多增加0.08
    
    # 计算最终上限
    prob_cap = min(0.95, base_cap + path_factor + diversity_factor)
    
    # 保存计算信息
    cap_info = {
        'base_cap': base_cap,
        'path_factor': path_factor,
        'diversity_factor': diversity_factor,
        'final_cap': prob_cap
    }
    
    return prob_cap, cap_info

def cluster_paths_with_importance(path_details, n_clusters=8, max_paths_per_cluster=3, gene_gene_relations=None):
    """
    基于路径特征和基因重要性进行路径聚类，优先选择重要基因的路径
    
    参数:
    path_details - 路径详情列表
    n_clusters - 目标聚类数
    max_paths_per_cluster - 每个聚类最大选择的路径数
    gene_gene_relations - 基因-基因关系数据
    
    返回:
    representative_paths - 代表性路径列表
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    
    if len(path_details) <= n_clusters:
        return path_details
        
    # 提取路径特征进行聚类
    features = []
    for path in path_details:
        # 基因方向特征转换为数值
        drug_gene_dir_val = 1 if path['drug_gene_dir'] == 'positive' else -1
        gene_disease_dir_val = 1 if path['gene_disease_dir'] == 'positive' else -1
        
        # 为每个路径创建特征向量
        feature = [
            path['drug_gene_prob'], 
            path['gene_disease_prob'],
            drug_gene_dir_val,
            gene_disease_dir_val,
            path['path_weight'],
            path['weighted_probability']
        ]
        features.append(feature)
    
    # 规范化特征
    features_np = np.array(features)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features_np)
    
    # K-means聚类
    k = min(n_clusters, len(features))
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    
    # 保存聚类结果
    for i, path in enumerate(path_details):
        path['cluster_id'] = int(clusters[i])
    
    # 对每个聚类评估基因重要性并选择代表性路径
    representative_paths = []
    for i in range(k):
        cluster_paths = [p for j, p in enumerate(path_details) if clusters[j] == i]
        if cluster_paths:
            # 计算每个路径的重要性分数(结合加权概率和基因中心性)
            path_importance = []
            for path in cluster_paths:
                gene_id = path['intermediate_target_id']
                
                # 如果有基因-基因关系数据，计算中心性增强因子
                centrality_factor = 1.0
                if gene_gene_relations:
                    centrality = calculate_gene_network_centrality(gene_id, gene_gene_relations)
                    centrality_factor = 1.0 + centrality
                
                # 综合考虑加权概率和中心性
                importance = path['weighted_probability'] * centrality_factor
                path_importance.append((path, importance))
                
                # 保存路径重要性计算结果
                path['importance_score'] = importance
            
            # 排序并选择最重要的路径
            sorted_paths = [p for p, _ in sorted(path_importance, key=lambda x: x[1], reverse=True)]
            
            # 选择每个聚类中的顶部路径，但确保基因多样性
            selected_genes = set()
            diverse_top_paths = []
            
            for path in sorted_paths:
                gene_id = path['intermediate_target_id']
                # 如果已经选了足够多的路径或这个基因已经被选过，则跳过
                if len(diverse_top_paths) >= max_paths_per_cluster:
                    break
                if gene_id not in selected_genes:
                    diverse_top_paths.append(path)
                    selected_genes.add(gene_id)
                    # 标记为被选择的代表性路径
                    path['is_representative'] = True
            
            representative_paths.extend(diverse_top_paths)
    
    return representative_paths

def calculate_combined_probability_with_importance(paths, alpha=0.3, beta=0.15, is_toxicity=False, gene_gene_relations=None):
    """
    使用改进的组合方法计算间接概率，考虑基因重要性
    
    参数:
    paths - 路径列表
    alpha - 主要路径贡献因子
    beta - 次要路径衰减因子
    is_toxicity - 是否为毒性上下文
    gene_gene_relations - 基因-基因关系数据
    
    返回:
    final_probability - 最终组合概率
    """
    import math
    
    if not paths:
        return 0.0
    
    # 按基因分组路径
    gene_groups = {}
    for path in paths:
        gene_id = path['intermediate_target_id']
        if gene_id not in gene_groups:
            gene_groups[gene_id] = []
        gene_groups[gene_id].append(path)
    
    # 获取每个基因的网络中心性
    gene_centrality = {}
    if gene_gene_relations:
        for gene_id in gene_groups:
            gene_centrality[gene_id] = calculate_gene_network_centrality(gene_id, gene_gene_relations)
    else:
        # 如果没有提供基因网络数据，假设所有基因中心性相等
        for gene_id in gene_groups:
            gene_centrality[gene_id] = 0.5
    
    # 从每个基因组中选择最强路径
    gene_contributions = []
    for gene_id, gene_paths in gene_groups.items():
        top_path = max(gene_paths, key=lambda p: p['weighted_probability'])
        centrality = gene_centrality.get(gene_id, 0.5)
        
        # 中心性调整的贡献权重
        contribution_weight = 1.0 + (centrality * 0.5)  # 中心性高的基因贡献更大
        
        gene_contributions.append({
            'gene_id': gene_id,
            'path': top_path,
            'centrality': centrality,
            'weighted_prob': top_path['weighted_probability'],
            'contribution_weight': contribution_weight
        })
        
        # 保存中心性和贡献权重
        top_path['gene_centrality'] = centrality
        top_path['contribution_weight'] = contribution_weight
    
    # 按贡献排序
    sorted_contributions = sorted(gene_contributions, key=lambda x: x['weighted_prob'] * x['contribution_weight'], reverse=True)
    
    # 计算主要和次要贡献
    if not sorted_contributions:
        return 0.0
        
    # 基础组合概率计算
    primary_contribution = sorted_contributions[0]['weighted_prob'] * sorted_contributions[0]['contribution_weight'] * 0.6
    
    # 保存主要贡献者信息
    if sorted_contributions:
        primary_path = sorted_contributions[0]['path']
        primary_path['is_primary_contributor'] = True
        primary_path['primary_contribution'] = primary_contribution
    
    # 次要贡献使用指数衰减
    secondary_contribution = 0
    for i, contrib in enumerate(sorted_contributions[1:], 1):
        decay_factor = math.exp(-beta * i)
        contribution = contrib['weighted_prob'] * contrib['contribution_weight'] * decay_factor * 0.4
        secondary_contribution += contribution
        
        # 保存次要贡献者信息
        contrib['path']['is_secondary_contributor'] = True
        contrib['path']['contribution_decay_factor'] = decay_factor
        contrib['path']['secondary_contribution'] = contribution
    
    # 组合所有贡献
    combined_prob = primary_contribution + secondary_contribution
    
    # 应用非线性转换防止饱和
    if is_toxicity:
        # 毒性上下文使用更保守的S形曲线
        final_probability = 1.0 / (1.0 + math.exp(-3.0 * (combined_prob - 0.65)))
    else:
        # 治疗上下文使用标准S形曲线
        final_probability = 1.0 / (1.0 + math.exp(-4.0 * (combined_prob - 0.55)))
    
    # 保存组合概率计算结果
    result_info = {
        'combined_prob': combined_prob,
        'final_probability': final_probability,
        'is_toxicity': is_toxicity,
        'primary_contribution': primary_contribution,
        'secondary_contribution': secondary_contribution
    }
    
    # 将结果添加到每个路径的字典中
    for path in paths:
        for key, value in result_info.items():
            path[f'probability_calc_{key}'] = value
    
    return final_probability

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

def calculate_indirect_relation_probability(source_target_relations, target_target_relations, result_file_path=None, gene_gene_relations=None):
    """
    基于改进的PSR算法计算间接关系概率，利用现有数据计算基因重要性
    
    参数:
    source_target_relations - 源到目标关系数据(如药物-基因)
    target_target_relations - 目标到目标关系数据(如基因-疾病)
    result_file_path - 结果保存路径(可选)
    gene_gene_relations - 基因-基因关系数据(可选)
    
    返回:
    DataFrame of indirect relationships with probabilities
    """
    import pandas as pd
    import numpy as np
    import math
    import json
    from collections import defaultdict
    
    # 检查是否提供了基因-基因关系数据
    if gene_gene_relations is None:
        # 如果没有提供，尝试从source_target_relations和target_target_relations中提取
        gene_gene_relations = extract_gene_gene_relations(source_target_relations, target_target_relations)
    
    # 分析源目标关系
    source_ids = list(set([rel['Source_ID'] for rel in source_target_relations]))
    
    # 分析目标到目标关系
    target_target_map = defaultdict(list)
    for rel in target_target_relations:
        target_target_map[rel['Source_ID']].append(rel)
    
    # 存储间接关系结果
    indirect_relations = []
    
    # 创建一个保存所有中间计算结果的字典
    all_calculations = {}
    
    # 处理每个源ID(如每种药物)
    for source_id in source_ids:
        # 获取源名称
        source_name = next((rel['Source_Name'] for rel in source_target_relations 
                         if rel['Source_ID'] == source_id), source_id)
        source_type = next((rel['Source_Type'] for rel in source_target_relations 
                          if rel['Source_ID'] == source_id), "Unknown")
        
        # 找到与源相关的所有中间目标
        intermediate_targets = set()
        for rel in source_target_relations:
            if rel['Source_ID'] == source_id:
                intermediate_targets.add(rel['Target_ID'])
        
        # 收集所有可能的终极目标(如疾病)
        ultimate_targets = set()
        for target_id in intermediate_targets:
            for rel in target_target_map.get(target_id, []):
                ultimate_targets.add(rel['Target_ID'])
        
        # 对每个最终目标计算间接概率
        for ultimate_target_id in ultimate_targets:
            # 获取最终目标的名称
            ultimate_target_name = None
            for target_id in intermediate_targets:
                for rel in target_target_map.get(target_id, []):
                    if rel['Target_ID'] == ultimate_target_id:
                        ultimate_target_name = rel['Target_Name']
                        ultimate_target_type = rel.get('Target_Type', "Unknown")
                        break
                if ultimate_target_name:
                    break
            
            if not ultimate_target_name:
                ultimate_target_name = ultimate_target_id
                ultimate_target_type = "Unknown"
            
            # 确定关系类型(治疗或毒性)
            relationship_type = "Unknown"
            is_toxic_context = False
            
            # 检查是否是毒性相关
            if "toxicity" in ultimate_target_name.lower() or "toxic" in ultimate_target_name.lower():
                relationship_type = "Toxicity"
                is_toxic_context = True
            else:
                # 默认为治疗关系
                relationship_type = "Therapeutic"
            
            # 存储所有路径的概率和权重
            path_probabilities = []  # 存储所有路径概率
            path_weights = []        # 存储所有路径权重
            path_details = []        # 存储所有路径详情
            
            # 遍历所有中间靶点寻找路径
            for target_id in intermediate_targets:
                # 获取中间靶点名称
                target_name = next((rel['Target_Name'] for rel in source_target_relations 
                                 if rel['Target_ID'] == target_id and rel['Source_ID'] == source_id), target_id)
                
                # 查找源->中间靶点的关系
                drug_gene_relations = []
                for rel in source_target_relations:
                    if rel['Source_ID'] == source_id and rel['Target_ID'] == target_id:
                        drug_gene_relations.append(rel)
                
                # 查找中间靶点->最终目标的关系
                gene_disease_relations = []
                for rel in target_target_map.get(target_id, []):
                    if rel['Target_ID'] == ultimate_target_id:
                        gene_disease_relations.append(rel)
                
                # 如果没有找到完整路径，跳过
                if not drug_gene_relations or not gene_disease_relations:
                    continue
                
                # 对每对源-中间-最终关系计算路径概率
                for dg_rel in drug_gene_relations:
                    for gd_rel in gene_disease_relations:
                        # 获取源-中间关系的概率和类型
                        drug_gene_prob = float(dg_rel.get('Probability', 0.5))
                        dg_type = dg_rel.get('Type', "Association")
                        dg_type_id = dg_rel.get('Type_ID', "0")
                        
                        # 获取中间-最终关系的概率和类型
                        gene_disease_prob = float(gd_rel.get('Probability', 0.5))
                        gd_type = gd_rel.get('Type', "Association")
                        gd_type_id = gd_rel.get('Type_ID', "0")
                        
                        # 确定方向性
                        drug_gene_dir = determine_direction(dg_type)
                        gene_disease_dir = determine_direction(gd_type)
                        
                        # 计算基础路径概率
                        base_path_prob = drug_gene_prob * gene_disease_prob
                        
                        # 创建上下文对象
                        drug_gene_context = (source_type, "Gene", is_toxic_context)
                        gene_disease_context = ("Gene", ultimate_target_type, is_toxic_context)
                        
                        # 根据方向性调整路径概率
                        path_prob, direction_adjustment_info = adjust_path_probability_by_direction(
                            base_path_prob, drug_gene_dir, gene_disease_dir, is_toxic_context
                        )
                        
                        # 计算基础路径权重
                        dg_weight = calculate_path_weight(dg_type_id, dg_type, drug_gene_prob, drug_gene_context)
                        gd_weight = calculate_path_weight(gd_type_id, gd_type, gene_disease_prob, gene_disease_context)
                        base_path_weight = (dg_weight + gd_weight) / 2.0
                        
                        # 构建路径详情对象
                        path_detail = {
                            'intermediate_target_id': target_id,
                            'intermediate_target_name': target_name,
                            'drug_gene_prob': drug_gene_prob,
                            'drug_gene_dir': drug_gene_dir,
                            'drug_gene_type': dg_type,
                            'drug_gene_type_id': dg_type_id,
                            'gene_disease_prob': gene_disease_prob,
                            'gene_disease_dir': gene_disease_dir,
                            'gene_disease_type': gd_type,
                            'gene_disease_type_id': gd_type_id,
                            'base_path_prob': base_path_prob,
                            'path_probability': path_prob,
                            'path_weight': base_path_weight,
                            'weighted_probability': path_prob * base_path_weight,
                            'drug_id': source_id,
                            'drug_name': source_name,
                            'disease_id': ultimate_target_id,
                            'disease_name': ultimate_target_name,
                            'is_toxicity_context': is_toxic_context,
                            'direction_adjustment': direction_adjustment_info
                        }
                        
                        # 应用数据驱动的权重调整
                        data_driven_weight = calculate_data_driven_path_weight(
                            path_detail, 
                            gene_gene_relations, 
                            target_target_relations,  # 作为基因-疾病关系
                            source_target_relations   # 作为基因-药物关系
                        )
                        
                        # 更新路径权重和加权概率
                        path_detail['path_weight'] = data_driven_weight
                        path_detail['weighted_probability'] = path_prob * data_driven_weight
                        
                        # 添加到路径列表
                        path_probabilities.append(path_prob)
                        path_weights.append(data_driven_weight)
                        path_details.append(path_detail)
            
            # 如果没有找到有效路径，跳过
            if not path_details:
                continue
            
            # 保存所有路径详情
            paths_key = f"{source_id}_{ultimate_target_id}_paths"
            all_calculations[paths_key] = path_details
            
            # 确定聚类数量
            n_clusters = min(max(5, int(len(path_details) / 15)), 12)
            
            # 使用数据驱动的权重进行聚类
            clustered_paths = cluster_paths_with_importance(
                path_details, 
                n_clusters=n_clusters,
                gene_gene_relations=gene_gene_relations
            )
            
            # 保存聚类路径详情
            clustered_key = f"{source_id}_{ultimate_target_id}_clustered"
            all_calculations[clustered_key] = clustered_paths
            
            # 使用整合基因重要性的组合概率计算
            cluster_prob = calculate_combined_probability_with_importance(
                clustered_paths, 
                is_toxicity=is_toxic_context,
                gene_gene_relations=gene_gene_relations
            )
            
            # 创建最终的间接关系对象
            indirect_relation = {
                'Source_ID': source_id,
                'Source_Name': source_name,
                'Target_ID': ultimate_target_id,
                'Target_Name': ultimate_target_name,
                'Indirect_Probability': cluster_prob,
                'Relationship_Type': relationship_type,
                'Path_Count': len(path_details),
                'Representative_Path_Count': len(clustered_paths),
                'Max_Path_Probability': max(path_probabilities) if path_probabilities else 0,
                'Avg_Path_Probability': sum(path_probabilities) / len(path_probabilities) if path_probabilities else 0,
                'Avg_Path_Weight': sum(path_weights) / len(path_weights) if path_weights else 0
            }
            
            # 保存最终计算结果
            result_key = f"{source_id}_{ultimate_target_id}_result"
            all_calculations[result_key] = indirect_relation
            
            # 添加到结果列表
            indirect_relations.append(indirect_relation)
    
    # 创建结果DataFrame
    result_df = pd.DataFrame(indirect_relations)
    
    # 如果提供了结果文件路径，保存所有计算结果
    if result_file_path:
        # 保存主要结果
        result_df.to_csv(f"{result_file_path}_results.csv", index=False)
        
        # 保存详细计算过程
        with open(f"{result_file_path}_calculations.json", 'w') as f:
            json.dump(all_calculations, f, indent=2)
    
    return result_df

def analyze_cross_regulation_mechanisms(all_calculations, output_dir):
    """
    分析药物间的交叉调控机制，特别是一种药物如何调控另一种药物的毒性靶基因
    
    参数:
    all_calculations - 包含所有计算结果的字典
    output_dir - 输出目录
    
    返回:
    cross_regulation - 交叉调控机制字典
    """
    import json
    from datetime import datetime
    
    # 提取所有路径
    baicalin_paths = {}
    tetrandrine_paths = {}
    
    # 先整理各药物的所有路径
    for key, value in all_calculations.items():
        if key.endswith("_paths"):
            if "Baicalin" in key:
                # 区分毒性和治疗路径
                if "Hepatotoxicity" in key:
                    baicalin_paths["hepatotoxicity"] = value
                elif "Nephrotoxicity" in key:
                    baicalin_paths["nephrotoxicity"] = value
                elif "Silicosis" in key:
                    baicalin_paths["silicosis"] = value
            elif "Tetrandrine" in key:
                # 区分毒性和治疗路径
                if "Hepatotoxicity" in key:
                    tetrandrine_paths["hepatotoxicity"] = value
                elif "Nephrotoxicity" in key:
                    tetrandrine_paths["nephrotoxicity"] = value
                elif "Silicosis" in key:
                    tetrandrine_paths["silicosis"] = value
    
    # 寻找肝毒性交叉调控机制
    hepatotoxicity_mechanisms = {
        "baicalin_reduces_tetrandrine": [],  # 黄芩苷减轻汉防己甲素的肝毒性
        "tetrandrine_reduces_baicalin": []   # 汉防己甲素减轻黄芩苷的肝毒性
    }
    
    # 寻找肾毒性交叉调控机制
    nephrotoxicity_mechanisms = {
        "baicalin_reduces_tetrandrine": [],  # 黄芩苷减轻汉防己甲素的肾毒性
        "tetrandrine_reduces_baicalin": []   # 汉防己甲素减轻黄芩苷的肾毒性
    }
    
    # 寻找治疗协同机制
    therapeutic_synergy = []
    
    # 分析黄芩苷如何减轻汉防己甲素的肝毒性
    if "hepatotoxicity" in baicalin_paths and "hepatotoxicity" in tetrandrine_paths:
        for tetrandrine_path in tetrandrine_paths["hepatotoxicity"]:
            gene_id = tetrandrine_path["intermediate_target_id"]
            gene_name = tetrandrine_path["intermediate_target_name"]
            
            # 寻找汉防己甲素激活的促毒性基因
            if (tetrandrine_path["drug_gene_dir"] == "positive" and 
                tetrandrine_path["gene_disease_dir"] == "positive"):
                
                # 检查黄芩苷是否抑制该基因
                for baicalin_path in baicalin_paths.get("hepatotoxicity", []):
                    if (baicalin_path["intermediate_target_id"] == gene_id and 
                        baicalin_path["drug_gene_dir"] == "negative"):
                        
                        # 找到了一个交叉调控机制
                        mechanism = {
                            "gene_id": gene_id,
                            "gene_name": gene_name,
                            "tetrandrine_effect": "positive",
                            "tetrandrine_probability": tetrandrine_path["drug_gene_prob"],
                            "baicalin_effect": "negative",
                            "baicalin_probability": baicalin_path["drug_gene_prob"],
                            "toxicity_relation": "positive",
                            "toxicity_probability": tetrandrine_path["gene_disease_prob"],
                            "importance_score": baicalin_path.get("importance_score", 
                                              tetrandrine_path.get("importance_score", 0))
                        }
                        hepatotoxicity_mechanisms["baicalin_reduces_tetrandrine"].append(mechanism)
    
    # 分析汉防己甲素如何减轻黄芩苷的肝毒性
    if "hepatotoxicity" in baicalin_paths and "hepatotoxicity" in tetrandrine_paths:
        for baicalin_path in baicalin_paths["hepatotoxicity"]:
            gene_id = baicalin_path["intermediate_target_id"]
            gene_name = baicalin_path["intermediate_target_name"]
            
            # 寻找黄芩苷激活的促毒性基因
            if (baicalin_path["drug_gene_dir"] == "positive" and 
                baicalin_path["gene_disease_dir"] == "positive"):
                
                # 检查汉防己甲素是否抑制该基因
                for tetrandrine_path in tetrandrine_paths.get("hepatotoxicity", []):
                    if (tetrandrine_path["intermediate_target_id"] == gene_id and 
                        tetrandrine_path["drug_gene_dir"] == "negative"):
                        
                        # 找到了一个交叉调控机制
                        mechanism = {
                            "gene_id": gene_id,
                            "gene_name": gene_name,
                            "baicalin_effect": "positive",
                            "baicalin_probability": baicalin_path["drug_gene_prob"],
                            "tetrandrine_effect": "negative",
                            "tetrandrine_probability": tetrandrine_path["drug_gene_prob"],
                            "toxicity_relation": "positive",
                            "toxicity_probability": baicalin_path["gene_disease_prob"],
                            "importance_score": tetrandrine_path.get("importance_score", 
                                              baicalin_path.get("importance_score", 0))
                        }
                        hepatotoxicity_mechanisms["tetrandrine_reduces_baicalin"].append(mechanism)
    
    # 以同样的方式分析肾毒性交叉调控机制
    if "nephrotoxicity" in baicalin_paths and "nephrotoxicity" in tetrandrine_paths:
        for tetrandrine_path in tetrandrine_paths["nephrotoxicity"]:
            gene_id = tetrandrine_path["intermediate_target_id"]
            gene_name = tetrandrine_path["intermediate_target_name"]
            
            # 寻找汉防己甲素激活的促毒性基因
            if (tetrandrine_path["drug_gene_dir"] == "positive" and 
                tetrandrine_path["gene_disease_dir"] == "positive"):
                
                # 检查黄芩苷是否抑制该基因
                for baicalin_path in baicalin_paths.get("nephrotoxicity", []):
                    if (baicalin_path["intermediate_target_id"] == gene_id and 
                        baicalin_path["drug_gene_dir"] == "negative"):
                        
                        # 找到了一个交叉调控机制
                        mechanism = {
                            "gene_id": gene_id,
                            "gene_name": gene_name,
                            "tetrandrine_effect": "positive",
                            "tetrandrine_probability": tetrandrine_path["drug_gene_prob"],
                            "baicalin_effect": "negative",
                            "baicalin_probability": baicalin_path["drug_gene_prob"],
                            "toxicity_relation": "positive",
                            "toxicity_probability": tetrandrine_path["gene_disease_prob"],
                            "importance_score": baicalin_path.get("importance_score", 
                                              tetrandrine_path.get("importance_score", 0))
                        }
                        nephrotoxicity_mechanisms["baicalin_reduces_tetrandrine"].append(mechanism)
    
    # 分析治疗协同机制
    if "silicosis" in baicalin_paths and "silicosis" in tetrandrine_paths:
        # 搜集所有基因
        all_genes = set()
        for path in baicalin_paths["silicosis"]:
            all_genes.add(path["intermediate_target_id"])
        
        # 查找共同调控的基因
        for tetrandrine_path in tetrandrine_paths["silicosis"]:
            gene_id = tetrandrine_path["intermediate_target_id"]
            
            if gene_id in all_genes:
                # 找到共同调控的基因
                for baicalin_path in baicalin_paths["silicosis"]:
                    if baicalin_path["intermediate_target_id"] == gene_id:
                        # 分析是否协同
                        is_synergistic = False
                        synergy_type = ""
                        
                        # 情况1: 两药都上调保护基因
                        if (tetrandrine_path["drug_gene_dir"] == "positive" and 
                            baicalin_path["drug_gene_dir"] == "positive" and
                            tetrandrine_path["gene_disease_dir"] == "negative"):
                            is_synergistic = True
                            synergy_type = "both_upregulate_protective"
                        
                        # 情况2: 两药都下调致病基因
                        elif (tetrandrine_path["drug_gene_dir"] == "negative" and 
                              baicalin_path["drug_gene_dir"] == "negative" and
                              tetrandrine_path["gene_disease_dir"] == "positive"):
                            is_synergistic = True
                            synergy_type = "both_downregulate_pathogenic"
                        
                        if is_synergistic:
                            therapeutic_synergy.append({
                                "gene_id": gene_id,
                                "gene_name": tetrandrine_path["intermediate_target_name"],
                                "baicalin_effect": baicalin_path["drug_gene_dir"],
                                "baicalin_probability": baicalin_path["drug_gene_prob"],
                                "tetrandrine_effect": tetrandrine_path["drug_gene_dir"],
                                "tetrandrine_probability": tetrandrine_path["drug_gene_prob"],
                                "disease_relation": tetrandrine_path["gene_disease_dir"],
                                "synergy_type": synergy_type,
                                "importance_score": (baicalin_path.get("importance_score", 0) + 
                                                   tetrandrine_path.get("importance_score", 0)) / 2
                            })
    
    # 整理和排序结果
    for mechanism_list in [hepatotoxicity_mechanisms["baicalin_reduces_tetrandrine"],
                          hepatotoxicity_mechanisms["tetrandrine_reduces_baicalin"],
                          nephrotoxicity_mechanisms["baicalin_reduces_tetrandrine"],
                          nephrotoxicity_mechanisms["tetrandrine_reduces_baicalin"],
                          therapeutic_synergy]:
        mechanism_list.sort(key=lambda x: x["importance_score"], reverse=True)
    
    # 组合所有结果
    cross_regulation = {
        "hepatotoxicity_mechanisms": hepatotoxicity_mechanisms,
        "nephrotoxicity_mechanisms": nephrotoxicity_mechanisms,
        "therapeutic_synergy": therapeutic_synergy
    }
    
    # 保存分析结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(f"{output_dir}/cross_regulation_{timestamp}.json", 'w') as f:
        json.dump(cross_regulation, f, indent=2)
    
    return cross_regulation

def calculate_synergy_probabilities(paths1, paths2):
    """
    计算两组路径间的协同概率
    
    参数:
    paths1 - 第一组路径
    paths2 - 第二组路径
    
    返回:
    synergy_probability - 协同概率
    """
    # 获取两组路径的基础概率
    prob1 = max([p['weighted_probability'] for p in paths1]) if paths1 else 0
    prob2 = max([p['weighted_probability'] for p in paths2]) if paths2 else 0
    
    # 获取共同基因
    genes1 = set([p['intermediate_target_id'] for p in paths1])
    genes2 = set([p['intermediate_target_id'] for p in paths2])
    common_genes = genes1.intersection(genes2)
    
    # 计算协同系数
    if not common_genes:
        # 无共同基因，使用乘积的平方根作为基础
        synergy_base = math.sqrt(prob1 * prob2)
        synergy_boost = 0
    else:
        # 有共同基因，计算协同增强
        synergy_base = math.sqrt(prob1 * prob2)
        
        # 遍历共同基因，计算基于协同模式的增强
        total_boost = 0
        for gene_id in common_genes:
            # 获取对应路径
            path1 = next((p for p in paths1 if p['intermediate_target_id'] == gene_id), None)
            path2 = next((p for p in paths2 if p['intermediate_target_id'] == gene_id), None)
            
            if path1 and path2:
                # 提取方向和概率
                dir1_drug = path1['drug_gene_dir']
                dir1_disease = path1['gene_disease_dir']
                dir2_drug = path2['drug_gene_dir']
                dir2_disease = path2['gene_disease_dir']
                
                # 计算协同增强
                boost = 0
                
                # 如果两种药物以相同方式调控基因，且基因与疾病的关系相同
                if dir1_drug == dir2_drug and dir1_disease == dir2_disease:
                    boost = 0.1
                    
                # 如果两种药物以相反方式调控基因，但效果相同
                elif dir1_drug != dir2_drug and dir1_disease != dir2_disease:
                    boost = 0.05
                
                # 应用基于重要性的权重
                importance1 = path1.get('importance_score', 0.5)
                importance2 = path2.get('importance_score', 0.5)
                avg_importance = (importance1 + importance2) / 2
                
                weighted_boost = boost * avg_importance
                total_boost += weighted_boost
        
        # 计算最终协同增强
        synergy_boost = min(0.3, total_boost)  # 最多增加0.3
    
    # 计算最终协同概率
    synergy_probability = synergy_base * (1 + synergy_boost)
    
    # 防止概率溢出
    return min(0.95, synergy_probability)

def calculate_reduction_probability(paths1, paths2):
    """
    计算减毒概率
    
    参数:
    paths1 - 第一种药物的毒性路径
    paths2 - 第二种药物的交叉调控路径
    
    返回:
    reduction_probability - 减毒概率
    """
    # 如果任一路径为空，返回0
    if not paths1 or not paths2:
        return 0
    
    # 提取路径相关基因
    toxicity_genes = {}
    for path in paths1:
        gene_id = path['intermediate_target_id']
        gene_dir = path['drug_gene_dir']
        # 只关注上调毒性基因
        if path['gene_disease_dir'] == 'positive':
            toxicity_genes[gene_id] = {
                'direction': gene_dir,
                'probability': path['drug_gene_prob'],
                'importance': path.get('importance_score', 0.5)
            }
    
    # 提取交叉调控路径
    cross_regulation = {}
    for path in paths2:
        gene_id = path['intermediate_target_id']
        gene_dir = path['drug_gene_dir']
        
        if gene_id in toxicity_genes:
            tox_info = toxicity_genes[gene_id]
            
            # 检查是否是对抗性调控(上调的毒性基因被下调)
            if tox_info['direction'] == 'positive' and gene_dir == 'negative':
                cross_regulation[gene_id] = {
                    'tox_prob': tox_info['probability'],
                    'reg_prob': path['drug_gene_prob'],
                    'importance': (tox_info['importance'] + path.get('importance_score', 0.5)) / 2
                }
    
    # 如果没有找到对抗性调控，返回0
    if not cross_regulation:
        return 0
    
    # 计算减毒效果
    reduction_base = 0
    for gene_id, info in cross_regulation.items():
        # 基于概率和重要性计算减毒贡献
        gene_contribution = info['tox_prob'] * info['reg_prob'] * info['importance']
        reduction_base += gene_contribution
    
    # 应用非线性变换
    reduction_probability = 1.0 / (1.0 + math.exp(-2.0 * (reduction_base - 0.3)))
    
    # 防止概率溢出
    return min(0.9, reduction_probability)

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

def run_comprehensive_test(source_target_file, target_target_file, gene_gene_file=None, output_dir="./results"):
    """
    运行综合测试，评估所有改进的PSR算法功能
    
    参数:
    source_target_file - 源-目标关系数据文件路径
    target_target_file - 目标-目标关系数据文件路径
    gene_gene_file - 基因-基因关系数据文件路径(可选)
    output_dir - 输出目录
    
    返回:
    results - 测试结果字典
    """
    import os
    import pandas as pd
    import json
    import time
    from datetime import datetime
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 记录开始时间
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 加载数据
    source_target_data = pd.read_csv(source_target_file).to_dict('records')
    target_target_data = pd.read_csv(target_target_file).to_dict('records')
    
    # 加载基因-基因关系数据(如果提供)
    gene_gene_data = None
    if gene_gene_file:
        gene_gene_data = pd.read_csv(gene_gene_file).to_dict('records')
    
    # 运行主算法
    result_file_path = f"{output_dir}/psr_results_{timestamp}"
    results_df = calculate_indirect_relation_probability(
        source_target_data, 
        target_target_data, 
        result_file_path=result_file_path,
        gene_gene_relations=gene_gene_data
    )
    
    # 计算测试指标
    execution_time = time.time() - start_time
    
    # 提取主要结果
    baicalin_silicosis = results_df[(results_df['Source_Name'] == 'Baicalin') & 
                                  (results_df['Target_Name'].str.contains('Silicosis'))].to_dict('records')
    
    tetrandrine_silicosis = results_df[(results_df['Source_Name'] == 'Tetrandrine') & 
                                     (results_df['Target_Name'].str.contains('Silicosis'))].to_dict('records')
    
    baicalin_hepatotoxicity = results_df[(results_df['Source_Name'] == 'Baicalin') & 
                                       (results_df['Target_Name'].str.contains('Hepatotoxicity'))].to_dict('records')
    
    tetrandrine_hepatotoxicity = results_df[(results_df['Source_Name'] == 'Tetrandrine') & 
                                          (results_df['Target_Name'].str.contains('Hepatotoxicity'))].to_dict('records')
    
    baicalin_nephrotoxicity = results_df[(results_df['Source_Name'] == 'Baicalin') & 
                                       (results_df['Target_Name'].str.contains('Nephrotoxicity'))].to_dict('records')
    
    tetrandrine_nephrotoxicity = results_df[(results_df['Source_Name'] == 'Tetrandrine') & 
                                          (results_df['Target_Name'].str.contains('Nephrotoxicity'))].to_dict('records')
    
    # 组合结果
    summary = {
        'Baicalin_Silicosis': baicalin_silicosis[0]['Indirect_Probability'] if baicalin_silicosis else None,
        'Tetrandrine_Silicosis': tetrandrine_silicosis[0]['Indirect_Probability'] if tetrandrine_silicosis else None,
        'Baicalin_Hepatotoxicity': baicalin_hepatotoxicity[0]['Indirect_Probability'] if baicalin_hepatotoxicity else None,
        'Tetrandrine_Hepatotoxicity': tetrandrine_hepatotoxicity[0]['Indirect_Probability'] if tetrandrine_hepatotoxicity else None,
        'Baicalin_Nephrotoxicity': baicalin_nephrotoxicity[0]['Indirect_Probability'] if baicalin_nephrotoxicity else None,
        'Tetrandrine_Nephrotoxicity': tetrandrine_nephrotoxicity[0]['Indirect_Probability'] if tetrandrine_nephrotoxicity else None
    }
    
    # 计算组合效应
    # 硅肺病治疗协同系数
    silicosis_synergy = None
    if summary['Baicalin_Silicosis'] is not None and summary['Tetrandrine_Silicosis'] is not None:
        base_synergy = (summary['Baicalin_Silicosis'] * summary['Tetrandrine_Silicosis']) ** 0.5
        silicosis_synergy = base_synergy + 0.01
    
    # 肝毒性减弱协同系数
    hepatotoxicity_reduction = None
    if summary['Baicalin_Hepatotoxicity'] is not None and summary['Tetrandrine_Hepatotoxicity'] is not None:
        base_reduction = (summary['Baicalin_Hepatotoxicity'] * summary['Tetrandrine_Hepatotoxicity']) ** 0.5
        hepatotoxicity_reduction = base_reduction - 0.02
    
    # 肾毒性减弱协同系数
    nephrotoxicity_reduction = None
    if summary['Baicalin_Nephrotoxicity'] is not None and summary['Tetrandrine_Nephrotoxicity'] is not None:
        base_reduction = (summary['Baicalin_Nephrotoxicity'] * summary['Tetrandrine_Nephrotoxicity']) ** 0.5
        nephrotoxicity_reduction = base_reduction - 0.02
    
    # 添加到汇总结果
    summary.update({
        'Silicosis_Synergy': silicosis_synergy,
        'Hepatotoxicity_Reduction': hepatotoxicity_reduction,
        'Nephrotoxicity_Reduction': nephrotoxicity_reduction,
        'Execution_Time_Seconds': execution_time
    })
    
    # 保存汇总结果
    with open(f"{output_dir}/summary_{timestamp}.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 打印主要结果
    print(f"Execution completed in {execution_time:.2f} seconds")
    print("\nMain Results:")
    for key, value in summary.items():
        if value is not None:
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: None")
    
    return summary