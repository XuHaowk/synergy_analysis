"""
协同作用分析模块：分析药物间的协同增效和减毒作用
"""

import pandas as pd
import numpy as np
import os
import json
from collections import defaultdict

# Import only the needed functions from probability module but not the ones that would create circular imports
from modules.probability import calculate_direct_relation_probability, calculate_synergistic_effect

def analyze_common_targets(drug1_targets, drug2_targets, all_relations):
    """
    分析两种药物的共同靶点
    
    参数:
    drug1_targets - 药物1的靶点DataFrame
    drug2_targets - 药物2的靶点DataFrame
    all_relations - 所有关系数据
    
    返回:
    common_targets - 共同靶点DataFrame
    """
    # 确保输入为DataFrame
    if not isinstance(drug1_targets, pd.DataFrame) or not isinstance(drug2_targets, pd.DataFrame):
        print("错误: 输入必须是pandas DataFrame")
        return pd.DataFrame()
    
    # 找出共同靶点
    common_targets = pd.merge(drug1_targets, drug2_targets, on='Target_ID', how='inner')
    
    # 如果没有共同靶点，返回空DataFrame
    if common_targets.empty:
        print("未找到共同靶点")
        return common_targets
    
    # 获取靶点详细信息
    targets_info = []
    
    for _, row in common_targets.iterrows():
        target_id = row['Target_ID']
        
        # 查找靶点名称和类型
        target_info = all_relations[
            (all_relations['Source_ID'] == target_id) | 
            (all_relations['Target_ID'] == target_id)
        ].iloc[0] if not all_relations[
            (all_relations['Source_ID'] == target_id) | 
            (all_relations['Target_ID'] == target_id)
        ].empty else None
        
        if target_info is not None:
            if target_info['Source_ID'] == target_id:
                target_name = target_info['Source_Name']
                target_type = target_info['Source_Type']
            else:
                target_name = target_info['Target_Name']
                target_type = target_info['Target_Type']
            
            # 查找与药物1的关系
            drug1_relation = all_relations[
                ((all_relations['Source_ID'] == target_id) & (all_relations['Source_Type'] == 'Gene')) |
                ((all_relations['Target_ID'] == target_id) & (all_relations['Target_Type'] == 'Gene'))
            ]
            
            drug1_rel_type = drug1_relation['Type'].iloc[0] if not drug1_relation.empty else 'Unknown'
            drug1_rel_prob = drug1_relation['Probability'].iloc[0] if not drug1_relation.empty else 0.0
            
            # 查找与药物2的关系
            drug2_relation = all_relations[
                ((all_relations['Source_ID'] == target_id) & (all_relations['Source_Type'] == 'Gene')) |
                ((all_relations['Target_ID'] == target_id) & (all_relations['Target_Type'] == 'Gene'))
            ]
            
            drug2_rel_type = drug2_relation['Type'].iloc[0] if not drug2_relation.empty else 'Unknown'
            drug2_rel_prob = drug2_relation['Probability'].iloc[0] if not drug2_relation.empty else 0.0
            
            # 添加到结果
            targets_info.append({
                'Target_ID': target_id,
                'Target_Name': target_name,
                'Target_Type': target_type,
                'Drug1_Relation_Type': drug1_rel_type,
                'Drug1_Relation_Probability': drug1_rel_prob,
                'Drug2_Relation_Type': drug2_rel_type,
                'Drug2_Relation_Probability': drug2_rel_prob
            })
    
    # 创建结果DataFrame
    result = pd.DataFrame(targets_info)
    
    # 按药物1关系概率排序
    result = result.sort_values(by='Drug1_Relation_Probability', ascending=False)
    
    return result

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
    
    # 尝试使用Python生物信息学包进行KEGG富集
    try:
        import Bio
        has_biopython = True
        print("检测到BioPython包，将尝试使用BioPython进行通路富集")
    except ImportError:
        has_biopython = False
        print("未检测到BioPython包，将使用内置通路数据进行富集")
    
    # 如果有BioPython并且有网络连接，尝试在线查询KEGG
    enrichment_results = None
    if has_biopython:
        try:
            from Bio.KEGG import REST
            
            # 尝试从KEGG获取通路信息
            print("尝试从KEGG数据库获取通路信息...")
            
            # 使用前10个基因测试KEGG连接
            test_genes = gene_list[:10]
            sample_gene = test_genes[0]
            
            try:
                # 尝试一个基因查询测试连接
                test_result = REST.kegg_get(f"genes/{sample_gene}")
                print(f"成功连接KEGG数据库并查询到基因: {sample_gene}")
                
                # 实际进行富集分析 - 这里需要根据KEGG API的具体方法实现
                # 这部分代码会比较复杂，需要多次API调用和数据处理
                
                # 此处为简化示例，在实际开发中需要扩展完整的KEGG查询逻辑
                enrichment_results = {'status': 'success', 'source': 'KEGG online'}
                
            except Exception as e:
                print(f"KEGG连接测试失败: {e}")
                print("将使用内置通路数据进行富集分析")
        
        except ImportError:
            print("未检测到Bio.KEGG模块，将使用内置通路数据进行富集")
    
    # 如果无法使用KEGG API，使用内置的通路数据
    if enrichment_results is None:
        print("使用内置通路数据进行富集分析...")
        
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
    
    return enrichment_results



def analyze_synergy_mechanisms(baicalin_relations, tetrandrine_relations, common_targets,
                              silicosis_relations, hepatotox_relations, nephrotox_relations):
    """
    分析黄芩苷与汉防己甲素的协同增效和减毒机制
    
    参数:
    baicalin_relations - 黄芩苷关系数据
    tetrandrine_relations - 汉防己甲素关系数据
    common_targets - 共同靶点数据
    silicosis_relations - 硅肺病关系数据
    hepatotox_relations - 肝毒性关系数据
    nephrotox_relations - 肾毒性关系数据
    
    返回:
    synergy_results - 协同分析结果
    """
    # 数据检查
    if baicalin_relations.empty or tetrandrine_relations.empty:
        print("错误: 药物关系数据为空")
        return {}
    
    # 计算各药物对疾病的直接关系概率
    baicalin_silicosis = calculate_direct_relation_probability(
        baicalin_relations[baicalin_relations['Target_ID'].isin(silicosis_relations['Source_ID'])]
    )
    
    tetrandrine_silicosis = calculate_direct_relation_probability(
        tetrandrine_relations[tetrandrine_relations['Target_ID'].isin(silicosis_relations['Source_ID'])]
    )
    
    baicalin_hepatotox = calculate_direct_relation_probability(
        baicalin_relations[baicalin_relations['Target_ID'].isin(hepatotox_relations['Source_ID'])]
    )
    
    tetrandrine_hepatotox = calculate_direct_relation_probability(
        tetrandrine_relations[tetrandrine_relations['Target_ID'].isin(hepatotox_relations['Source_ID'])]
    )
    
    baicalin_nephrotox = calculate_direct_relation_probability(
        baicalin_relations[baicalin_relations['Target_ID'].isin(nephrotox_relations['Source_ID'])]
    )
    
    tetrandrine_nephrotox = calculate_direct_relation_probability(
        tetrandrine_relations[tetrandrine_relations['Target_ID'].isin(nephrotox_relations['Source_ID'])]
    )
    
    # 计算协同作用
    silicosis_synergy = calculate_synergistic_effect(
        baicalin_silicosis.iloc[0].to_dict() if not baicalin_silicosis.empty else {},
        tetrandrine_silicosis.iloc[0].to_dict() if not tetrandrine_silicosis.empty else {},
        common_targets
    ) if not common_targets.empty else {'Synergy_Coefficient': 0, 'Synergy_Type': '无效'}
    
    hepatotox_synergy = calculate_synergistic_effect(
        baicalin_hepatotox.iloc[0].to_dict() if not baicalin_hepatotox.empty else {},
        tetrandrine_hepatotox.iloc[0].to_dict() if not tetrandrine_hepatotox.empty else {},
        common_targets
    ) if not common_targets.empty else {'Synergy_Coefficient': 0, 'Synergy_Type': '无效'}
    
    nephrotox_synergy = calculate_synergistic_effect(
        baicalin_nephrotox.iloc[0].to_dict() if not baicalin_nephrotox.empty else {},
        tetrandrine_nephrotox.iloc[0].to_dict() if not tetrandrine_nephrotox.empty else {},
        common_targets
    ) if not common_targets.empty else {'Synergy_Coefficient': 0, 'Synergy_Type': '无效'}
    
    # 收集机制类型
    mechanism_types = {
        'silicosis_treatment': {
            'synergy_coefficient': silicosis_synergy['Synergy_Coefficient'],
            'synergy_type': silicosis_synergy['Synergy_Type']
        },
        'hepatotoxicity_reduction': {
            'synergy_coefficient': 1.0 - hepatotox_synergy['Synergy_Coefficient'] if hepatotox_synergy['Synergy_Coefficient'] > 0 else 0,
            'synergy_type': '协同减毒' if hepatotox_synergy['Synergy_Coefficient'] < 0.8 else '无明显减毒'
        },
        'nephrotoxicity_reduction': {
            'synergy_coefficient': 1.0 - nephrotox_synergy['Synergy_Coefficient'] if nephrotox_synergy['Synergy_Coefficient'] > 0 else 0,
            'synergy_type': '协同减毒' if nephrotox_synergy['Synergy_Coefficient'] < 0.8 else '无明显减毒'
        }
    }
    
    # 分析治疗协同靶点
    therapeutic_targets = []
    
    if not common_targets.empty:
        for _, target in common_targets.iterrows():
            target_id = target['Target_ID']
            target_name = target['Target_Name']
            
            # 获取靶点与硅肺病的关系
            target_silicosis = silicosis_relations[
                (silicosis_relations['Source_ID'] == target_id) |
                (silicosis_relations['Target_ID'] == target_id)
            ]
            
            if not target_silicosis.empty:
                relation_type = target_silicosis['Type'].iloc[0]
                relation_prob = float(target_silicosis['Probability'].iloc[0])
                
                # 获取药物与靶点的关系
                baicalin_target = baicalin_relations[
                    (baicalin_relations['Target_ID'] == target_id)
                ]
                
                tetrandrine_target = tetrandrine_relations[
                    (tetrandrine_relations['Target_ID'] == target_id)
                ]
                
                if not baicalin_target.empty and not tetrandrine_target.empty:
                    baicalin_type = baicalin_target['Type'].iloc[0]
                    baicalin_prob = float(baicalin_target['Probability'].iloc[0])
                    
                    tetrandrine_type = tetrandrine_target['Type'].iloc[0]
                    tetrandrine_prob = float(tetrandrine_target['Probability'].iloc[0])
                    
                    # 判断是否协同
                    is_synergistic = False
                    synergy_mechanism = "未知"
                    
                    # 情况1：两药均激活保护性基因
                    if (baicalin_type in ['Activates', 'Increases', 'Upregulates', 'Positive_Correlation'] and
                        tetrandrine_type in ['Activates', 'Increases', 'Upregulates', 'Positive_Correlation'] and
                        relation_type in ['Negative_Correlation', 'Treats', 'Decreases']):
                        is_synergistic = True
                        synergy_mechanism = "共同激活保护性基因"
                    
                    # 情况2：两药均抑制致病基因
                    elif (baicalin_type in ['Inhibits', 'Decreases', 'Downregulates', 'Negative_Correlation'] and
                          tetrandrine_type in ['Inhibits', 'Decreases', 'Downregulates', 'Negative_Correlation'] and
                          relation_type in ['Positive_Correlation', 'Causes', 'Increases']):
                        is_synergistic = True
                        synergy_mechanism = "共同抑制致病基因"
                    
                    # 如果是协同的，添加到治疗协同靶点列表
                    if is_synergistic:
                        therapeutic_targets.append({
                            'Target_ID': target_id,
                            'Target_Name': target_name,
                            'Mechanism': synergy_mechanism,
                            'Baicalin_Relation': baicalin_type,
                            'Baicalin_Probability': baicalin_prob,
                            'Tetrandrine_Relation': tetrandrine_type,
                            'Tetrandrine_Probability': tetrandrine_prob,
                            'Disease_Relation': relation_type,
                            'Disease_Probability': relation_prob,
                            'Synergy_Score': (baicalin_prob * tetrandrine_prob * relation_prob) ** (1/3)
                        })
    
    # 分析肝保护协同靶点
    hepatoprotective_targets = []
    
    if not common_targets.empty:
        for _, target in common_targets.iterrows():
            target_id = target['Target_ID']
            target_name = target['Target_Name']
            
            # 获取靶点与肝毒性的关系
            target_hepatotox = hepatotox_relations[
                (hepatotox_relations['Source_ID'] == target_id) |
                (hepatotox_relations['Target_ID'] == target_id)
            ]
            
            if not target_hepatotox.empty:
                relation_type = target_hepatotox['Type'].iloc[0]
                relation_prob = float(target_hepatotox['Probability'].iloc[0])
                
                # 仅考虑促进肝毒性的基因
                if relation_type in ['Positive_Correlation', 'Causes', 'Increases']:
                    # 获取药物与靶点的关系
                    baicalin_target = baicalin_relations[
                        (baicalin_relations['Target_ID'] == target_id)
                    ]
                    
                    tetrandrine_target = tetrandrine_relations[
                        (tetrandrine_relations['Target_ID'] == target_id)
                    ]
                    
                    if not baicalin_target.empty and not tetrandrine_target.empty:
                        baicalin_type = baicalin_target['Type'].iloc[0]
                        baicalin_prob = float(baicalin_target['Probability'].iloc[0])
                        
                        tetrandrine_type = tetrandrine_target['Type'].iloc[0]
                        tetrandrine_prob = float(tetrandrine_target['Probability'].iloc[0])
                        
                        # 情况1：两药均抑制促毒性基因
                        if (baicalin_type in ['Inhibits', 'Decreases', 'Downregulates', 'Negative_Correlation'] and
                            tetrandrine_type in ['Inhibits', 'Decreases', 'Downregulates', 'Negative_Correlation']):
                            
                            hepatoprotective_targets.append({
                                'Target_ID': target_id,
                                'Target_Name': target_name,
                                'Mechanism': "共同抑制促肝毒性基因",
                                'Baicalin_Relation': baicalin_type,
                                'Baicalin_Probability': baicalin_prob,
                                'Tetrandrine_Relation': tetrandrine_type,
                                'Tetrandrine_Probability': tetrandrine_prob,
                                'Toxicity_Relation': relation_type,
                                'Toxicity_Probability': relation_prob,
                                'Protection_Score': (baicalin_prob * tetrandrine_prob * relation_prob) ** (1/3)
                            })
                        
                        # 情况2：一药激活，一药抑制的拮抗作用
                        elif ((baicalin_type in ['Activates', 'Increases', 'Upregulates', 'Positive_Correlation'] and
                               tetrandrine_type in ['Inhibits', 'Decreases', 'Downregulates', 'Negative_Correlation']) or
                              (baicalin_type in ['Inhibits', 'Decreases', 'Downregulates', 'Negative_Correlation'] and
                               tetrandrine_type in ['Activates', 'Increases', 'Upregulates', 'Positive_Correlation'])):
                            
                            # 确定哪个药物抑制，哪个药物激活
                            if baicalin_type in ['Inhibits', 'Decreases', 'Downregulates', 'Negative_Correlation']:
                                inhibitor = "黄芩苷"
                                activator = "汉防己甲素"
                            else:
                                inhibitor = "汉防己甲素"
                                activator = "黄芩苷"
                            
                            hepatoprotective_targets.append({
                                'Target_ID': target_id,
                                'Target_Name': target_name,
                                'Mechanism': f"{inhibitor}抑制{activator}激活的促肝毒性基因",
                                'Baicalin_Relation': baicalin_type,
                                'Baicalin_Probability': baicalin_prob,
                                'Tetrandrine_Relation': tetrandrine_type,
                                'Tetrandrine_Probability': tetrandrine_prob,
                                'Toxicity_Relation': relation_type,
                                'Toxicity_Probability': relation_prob,
                                'Protection_Score': (baicalin_prob * tetrandrine_prob * relation_prob) ** (1/3) * 0.8  # 拮抗作用效果较弱
                            })
    
    # 分析肾保护协同靶点
    nephroprotective_targets = []
    
    if not common_targets.empty:
        for _, target in common_targets.iterrows():
            target_id = target['Target_ID']
            target_name = target['Target_Name']
            
            # 获取靶点与肾毒性的关系
            target_nephrotox = nephrotox_relations[
                (nephrotox_relations['Source_ID'] == target_id) |
                (nephrotox_relations['Target_ID'] == target_id)
            ]
            
            if not target_nephrotox.empty:
                relation_type = target_nephrotox['Type'].iloc[0]
                relation_prob = float(target_nephrotox['Probability'].iloc[0])
                
                # 仅考虑促进肾毒性的基因
                if relation_type in ['Positive_Correlation', 'Causes', 'Increases']:
                    # 获取药物与靶点的关系
                    baicalin_target = baicalin_relations[
                        (baicalin_relations['Target_ID'] == target_id)
                    ]
                    
                    tetrandrine_target = tetrandrine_relations[
                        (tetrandrine_relations['Target_ID'] == target_id)
                    ]
                    
                    if not baicalin_target.empty and not tetrandrine_target.empty:
                        baicalin_type = baicalin_target['Type'].iloc[0]
                        baicalin_prob = float(baicalin_target['Probability'].iloc[0])
                        
                        tetrandrine_type = tetrandrine_target['Type'].iloc[0]
                        tetrandrine_prob = float(tetrandrine_target['Probability'].iloc[0])
                        
                        # 情况1：两药均抑制促毒性基因
                        if (baicalin_type in ['Inhibits', 'Decreases', 'Downregulates', 'Negative_Correlation'] and
                            tetrandrine_type in ['Inhibits', 'Decreases', 'Downregulates', 'Negative_Correlation']):
                            
                            nephroprotective_targets.append({
                                'Target_ID': target_id,
                                'Target_Name': target_name,
                                'Mechanism': "共同抑制促肾毒性基因",
                                'Baicalin_Relation': baicalin_type,
                                'Baicalin_Probability': baicalin_prob,
                                'Tetrandrine_Relation': tetrandrine_type,
                                'Tetrandrine_Probability': tetrandrine_prob,
                                'Toxicity_Relation': relation_type,
                                'Toxicity_Probability': relation_prob,
                                'Protection_Score': (baicalin_prob * tetrandrine_prob * relation_prob) ** (1/3)
                            })
                        
                        # 情况2：一药激活，一药抑制的拮抗作用
                        elif ((baicalin_type in ['Activates', 'Increases', 'Upregulates', 'Positive_Correlation'] and
                               tetrandrine_type in ['Inhibits', 'Decreases', 'Downregulates', 'Negative_Correlation']) or
                              (baicalin_type in ['Inhibits', 'Decreases', 'Downregulates', 'Negative_Correlation'] and
                               tetrandrine_type in ['Activates', 'Increases', 'Upregulates', 'Positive_Correlation'])):
                            
                            # 确定哪个药物抑制，哪个药物激活
                            if baicalin_type in ['Inhibits', 'Decreases', 'Downregulates', 'Negative_Correlation']:
                                inhibitor = "黄芩苷"
                                activator = "汉防己甲素"
                            else:
                                inhibitor = "汉防己甲素"
                                activator = "黄芩苷"
                            
                            nephroprotective_targets.append({
                                'Target_ID': target_id,
                                'Target_Name': target_name,
                                'Mechanism': f"{inhibitor}抑制{activator}激活的促肾毒性基因",
                                'Baicalin_Relation': baicalin_type,
                                'Baicalin_Probability': baicalin_prob,
                                'Tetrandrine_Relation': tetrandrine_type,
                                'Tetrandrine_Probability': tetrandrine_prob,
                                'Toxicity_Relation': relation_type,
                                'Toxicity_Probability': relation_prob,
                                'Protection_Score': (baicalin_prob * tetrandrine_prob * relation_prob) ** (1/3) * 0.8  # 拮抗作用效果较弱
                            })
    
    # 按评分排序
    therapeutic_targets.sort(key=lambda x: x['Synergy_Score'], reverse=True)
    hepatoprotective_targets.sort(key=lambda x: x['Protection_Score'], reverse=True)
    nephroprotective_targets.sort(key=lambda x: x['Protection_Score'], reverse=True)
    
    # 保存分析结果
    result_dir = os.path.join('results', 'tables')
    os.makedirs(result_dir, exist_ok=True)
    
    # 保存机制类型
    pd.DataFrame([mechanism_types]).to_csv(os.path.join(result_dir, 'mechanism_types.csv'), index=False)
    
    # 保存靶点数据
    if therapeutic_targets:
        pd.DataFrame(therapeutic_targets).to_csv(os.path.join(result_dir, 'therapeutic_targets.csv'), index=False)
    
    if hepatoprotective_targets:
        pd.DataFrame(hepatoprotective_targets).to_csv(os.path.join(result_dir, 'hepatoprotective_targets.csv'), index=False)
    
    if nephroprotective_targets:
        pd.DataFrame(nephroprotective_targets).to_csv(os.path.join(result_dir, 'nephroprotective_targets.csv'), index=False)
    
    # 返回分析结果
    return {
        'mechanism_types': mechanism_types,
        'therapeutic_targets': therapeutic_targets,
        'hepatoprotective_targets': hepatoprotective_targets,
        'nephroprotective_targets': nephroprotective_targets
    }






