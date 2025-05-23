"""
数据输入模块：处理原始数据的加载和预处理
"""

import os
import pandas as pd
import numpy as np
from openpyxl import load_workbook
import config

def process_relationships(relations_file_path):
    """
    Process relationships from CSV file according to specified criteria.
    
    Parameters:
    -----------
    relations_file_path : str
        Path to the enhanced_relations_cleaned.csv file
        
    Returns:
    --------
    processed_relations : pandas DataFrame
        Processed relationships with proper weights
    """
    import pandas as pd
    import numpy as np
    import os
    
    # Read the relationships file
    relations = pd.read_csv(relations_file_path)
    
    # Create a unique identifier for each entity pair
    relations['entity_pair'] = relations['Source_ID'].astype(str) + '_' + relations['Target_ID'].astype(str)
    
    # Classify relationship types
    positive_types = ['Positive_Correlation', 'Association', 'Drug_Target', 'Bind']
    negative_types = ['Negative_Correlation']
    activation_types = ['Positive_Correlation']
    inhibition_types = ['Negative_Correlation']
    
    # Convert probability for negative relationships to negative values
    relations.loc[relations['Type'].isin(negative_types), 'Probability'] = -relations.loc[relations['Type'].isin(negative_types), 'Probability']
    
    # Identify entity pairs with both activation and inhibition relationships
    conflict_pairs = []
    for pair in relations['entity_pair'].unique():
        pair_relations = relations[relations['entity_pair'] == pair]
        if any(pair_relations['Type'].isin(activation_types)) and any(pair_relations['Type'].isin(inhibition_types)):
            conflict_pairs.append(pair)
    
    # Remove conflicting relationships
    relations = relations[~relations['entity_pair'].isin(conflict_pairs)]
    
    # Define priority order for relationship types
    priority_order = {
        'Positive_Correlation': 1,
        'Negative_Correlation': 1,
        'Bind': 2,
        'Drug_Target': 2,
        'Association': 3,
        'Correlation': 3,
        'Drug_Protein': 4,
        'Disease_Protein': 4,
        'Target_Disease': 4,
        'Drug_Interaction': 5,
        'Cotreatment': 6,
        'Biomarker_Disease': 7
    }
    
    # Add priority to each relationship
    relations['priority'] = relations['Type'].map(priority_order).fillna(999)
    
    # Sort by entity pair and priority
    relations = relations.sort_values(['entity_pair', 'priority'])
    
    # Keep only the highest priority relationship for each entity pair
    processed_relations = relations.drop_duplicates(subset=['entity_pair'], keep='first')
    
    # Remove temporary columns
    processed_relations = processed_relations.drop(['entity_pair', 'priority'], axis=1)
    
    print(f"Original relations: {len(relations)}")
    print(f"Processed relations after removing conflicts and prioritizing: {len(processed_relations)}")
    
    # Save the processed relationships
    os.makedirs('data/processed/relationships', exist_ok=True)
    processed_relations.to_csv('data/processed/relationships/processed_relationships.csv', index=False)
    
    return processed_relations

def load_and_process_data():
    """加载并处理数据"""
    print("读取焦点实体数据...")
    focal_entities = pd.read_csv(config.FOCAL_ENTITIES_FILE)
    
    # 打印列名以进行调试
    print("焦点实体数据列名:", focal_entities.columns.tolist())
    
    print("读取关系数据...")
    # Process the relationships according to our criteria
    enhanced_relations = process_relationships(config.ENHANCED_RELATIONS_FILE)
    
    # 打印列名以进行调试
    print("关系数据列名:", enhanced_relations.columns.tolist())
    
    # 存储所有关系数据
    all_relations = enhanced_relations
    
    # 强大的实体搜索函数
    def find_entity(entities, search_terms, id_column='Source_ID'):
        """
        在数据框中查找实体，支持多种搜索策略
        
        参数:
        entities - 数据框
        search_terms - 要搜索的术语列表
        id_column - ID列的名称
        
        返回:
        匹配的实体数据框
        """
        result = pd.DataFrame()
        
        # 尝试在所有文本列中查找
        for col in entities.columns:
            # 跳过非字符串类型的列
            if entities[col].dtype != 'object':
                continue
                
            for term in search_terms:
                # 尝试精确匹配
                temp_result = entities[entities[col] == term]
                if len(temp_result) > 0:
                    result = temp_result
                    print(f"找到实体 '{term}' 在列 '{col}' 中 (精确匹配)")
                    break
                    
                # 尝试包含匹配
                temp_result = entities[entities[col].astype(str).str.contains(term, na=False, case=False)]
                if len(temp_result) > 0:
                    result = temp_result
                    print(f"找到实体 '{term}' 在列 '{col}' 中 (包含匹配)")
                    break
            
            if len(result) > 0:
                break
        
        # 如果找到多个结果，只保留第一个
        if len(result) > 1:
            print(f"警告: 找到多个匹配 '{search_terms}'. 使用第一个匹配项。")
            result = result.iloc[[0]]
            
        # 如果没有找到，创建一个带有错误消息的dummy实体
        if len(result) == 0:
            print(f"错误: 无法找到实体 '{search_terms}'. 创建dummy实体。")
            dummy_data = {id_column: [-1]}
            # 为数据框中的所有列创建空值
            for col in entities.columns:
                if col not in dummy_data:
                    dummy_data[col] = [None if col != 'Name' else search_terms[0]]
            result = pd.DataFrame(dummy_data)
        
        return result
    
    # 提取焦点实体，使用多个可能的名称变体
    baicalin = find_entity(focal_entities, ['黄芩苷', '黄苷', 'Baicalin'])
    tetrandrine = find_entity(focal_entities, ['粉防己碱', '汉防己甲素', 'Tetrandrine'])
    silicosis = find_entity(focal_entities, ['硅肺病', '矽肺', 'Silicosis'])
    hepatotox = find_entity(focal_entities, ['肝毒性', 'Hepatotoxicity'])
    nephrotox = find_entity(focal_entities, ['肾毒性', 'Nephrotoxicity'])
    
    # 使用Source_ID而不是ID来识别实体
    baicalin_id = baicalin['Source_ID'].values[0]
    tetrandrine_id = tetrandrine['Source_ID'].values[0]
    silicosis_id = silicosis['Source_ID'].values[0]
    hepatotox_id = hepatotox['Source_ID'].values[0]
    nephrotox_id = nephrotox['Source_ID'].values[0]
    
    print(f"已找到实体 - 黄芩苷(ID:{baicalin_id}), 粉防己碱(ID:{tetrandrine_id}), "
          f"硅肺病(ID:{silicosis_id}), 肝毒性(ID:{hepatotox_id}), 肾毒性(ID:{nephrotox_id})")
    
    # 提取关系
    baicalin_relations = enhanced_relations[enhanced_relations['Source_ID'] == baicalin_id]
    tetrandrine_relations = enhanced_relations[enhanced_relations['Source_ID'] == tetrandrine_id]
    silicosis_relations = enhanced_relations[enhanced_relations['Source_ID'] == silicosis_id]
    hepatotox_relations = enhanced_relations[enhanced_relations['Source_ID'] == hepatotox_id]
    nephrotox_relations = enhanced_relations[enhanced_relations['Source_ID'] == nephrotox_id]
    
    print("保存处理后的数据...")
    processed_data = {
        'baicalin': baicalin,
        'tetrandrine': tetrandrine,
        'silicosis': silicosis,
        'hepatotox': hepatotox,
        'nephrotox': nephrotox,
        'baicalin_relations': baicalin_relations,
        'tetrandrine_relations': tetrandrine_relations,
        'silicosis_relations': silicosis_relations,
        'hepatotox_relations': hepatotox_relations,
        'nephrotox_relations': nephrotox_relations,
        'all_relations': all_relations  # 添加完整的关系数据
    }
    
    # 尝试加载TCMSP数据库
    try:
        tcmsp_db = pd.ExcelFile(config.TCMSP_DB_FILE)
        tcmsp_data = {
            'molecule_target': pd.read_excel(tcmsp_db, sheet_name='v_Molecules_Targets'),
            'herb_molecule': pd.read_excel(tcmsp_db, sheet_name='v_Herbs_Molecules')
        }
        processed_data['tcmsp_data'] = tcmsp_data
    except:
        print(f"无法读取TCMSP数据库文件: {config.TCMSP_DB_FILE}")
    
    return processed_data

def extract_targets_from_relations_complex(relations, entity_type, entity_id=None):
    """
    从关系数据中提取目标ID列表，考虑复杂的双向关系
    """
    targets = []
    target_relationships = []
    
    # 类型映射字典
    type_mapping = {
        'Drug': 'Chemical',
        'Hepatotoxicity': 'Disease', 
        'Nephrotoxicity': 'Disease',
        'Disease': 'Disease'
    }
    
    # 获取数据库中的实际类型
    db_type = type_mapping.get(entity_type, entity_type)
    
    # 打印调试信息
    print(f"Extracting targets - Type: {db_type}, ID: {entity_id}")
    print(f"Relationship data shape: {relations.shape}")
    
    if not relations.empty:
        print(f"First relationship: Source_ID={relations['Source_ID'].iloc[0]}, Target_ID={relations['Target_ID'].iloc[0]}")
    
    # 确保ID为字符串类型，避免类型不匹配问题
    if entity_id is not None and not isinstance(entity_id, str):
        entity_id = str(entity_id)
    
    # 将所有ID转换为字符串类型
    relations['Source_ID'] = relations['Source_ID'].astype(str)
    relations['Target_ID'] = relations['Target_ID'].astype(str)
    
    # 根据实体类型确定筛选条件
    if db_type == 'Chemical':  # 药物
        # 情况1: 药物作为Source，基因作为Target
        drug_as_source = relations[
            (relations['Source_Type'] == 'Chemical') & 
            (relations['Target_Type'] == 'Gene')
        ]
        
        # 情况2: 基因作为Source，药物作为Target
        drug_as_target = relations[
            (relations['Source_Type'] == 'Gene') & 
            (relations['Target_Type'] == 'Chemical')
        ]
        
        # 根据指定的实体ID进行过滤
        if entity_id is not None:
            drug_as_source = drug_as_source[drug_as_source['Source_ID'] == entity_id]
            drug_as_target = drug_as_target[drug_as_target['Target_ID'] == entity_id]
            
            print(f"Rows matching Source_ID={entity_id}: {len(drug_as_source)}")
            print(f"Rows matching Target_ID={entity_id}: {len(drug_as_target)}")
        
        # 提取靶点ID和关系详情
        targets_from_source = list(drug_as_source['Target_ID'].unique())
        for _, row in drug_as_source.iterrows():
            target_relationships.append({
                'source_id': row['Source_ID'],
                'source_name': row['Source_Name'],
                'target_id': row['Target_ID'],
                'target_name': row['Target_Name'],
                'relationship': row['Type'],
                'probability': row['Probability'],
                'direction': 'drug_to_target'
            })
            
        targets_from_target = list(drug_as_target['Source_ID'].unique())
        for _, row in drug_as_target.iterrows():
            target_relationships.append({
                'source_id': row['Target_ID'],  # 反转以保持一致性
                'source_name': row['Target_Name'],
                'target_id': row['Source_ID'],
                'target_name': row['Source_Name'],
                'relationship': row['Type'],
                'probability': row['Probability'],
                'direction': 'target_to_drug'
            })
            
        targets = targets_from_source + targets_from_target
        
        print(f"Found {len(targets_from_source)} targets with drug as source")
        print(f"Found {len(targets_from_target)} targets with drug as target")
        
    elif db_type == 'Disease':  # 疾病/毒性
        # 情况1: 疾病作为Source，基因作为Target
        disease_as_source = relations[
            (relations['Source_Type'] == 'Disease') & 
            (relations['Target_Type'] == 'Gene')
        ]
        
        # 情况2: 基因作为Source，疾病作为Target
        disease_as_target = relations[
            (relations['Source_Type'] == 'Gene') & 
            (relations['Target_Type'] == 'Disease')
        ]
        
        # 根据指定的实体ID进行过滤
        if entity_id is not None:
            disease_as_source = disease_as_source[disease_as_source['Source_ID'] == entity_id]
            disease_as_target = disease_as_target[disease_as_target['Target_ID'] == entity_id]
            
            print(f"Rows matching Source_ID={entity_id}: {len(disease_as_source)}")
            print(f"Rows matching Target_ID={entity_id}: {len(disease_as_target)}")
        
        # 提取靶点ID和关系详情
        targets_from_source = list(disease_as_source['Target_ID'].unique())
        for _, row in disease_as_source.iterrows():
            target_relationships.append({
                'source_id': row['Source_ID'],
                'source_name': row['Source_Name'],
                'target_id': row['Target_ID'],
                'target_name': row['Target_Name'],
                'relationship': row['Type'],
                'probability': row['Probability'],
                'direction': 'disease_to_target'
            })
            
        targets_from_target = list(disease_as_target['Source_ID'].unique())
        for _, row in disease_as_target.iterrows():
            target_relationships.append({
                'source_id': row['Target_ID'],  # 反转以保持一致性
                'source_name': row['Target_Name'],
                'target_id': row['Source_ID'],
                'target_name': row['Source_Name'],
                'relationship': row['Type'],
                'probability': row['Probability'],
                'direction': 'target_to_disease'
            })
            
        targets = targets_from_source + targets_from_target
        
        print(f"Found {len(targets_from_source)} targets with disease as source")
        print(f"Found {len(targets_from_target)} targets with disease as target")
    
    # 标准化格式 - 转换为字符串并移除重复项
    targets = [str(t) for t in targets]
    targets = list(set(targets))  # 移除重复项
    
    print(f"Found {len(targets)} targets for {entity_type} (ID: {entity_id})")
    
    # 如果找到靶点，打印前5个样本
    if targets:
        print(f"First 5 target samples: {targets[:5]}")
    
    # 保存靶点和关系信息
    entity_name = f"{entity_type}_{entity_id}"
    
    # 创建目录
    os.makedirs('data/processed/targets', exist_ok=True)
    
    # 保存靶点列表
    target_df = pd.DataFrame({'Target_ID': targets})
    target_df.to_csv(f'data/processed/targets/{entity_name}_targets.csv', index=False)
    
    # 保存关系详情
    if target_relationships:
        rel_df = pd.DataFrame(target_relationships)
        rel_df.to_csv(f'data/processed/targets/{entity_name}_relationships.csv', index=False)
    
    return targets

def load_enhanced_relations_data():
    """加载增强关系数据并提取相关实体"""
    import pandas as pd
    import os
    
    # 寻找数据文件
    relations_file = 'data/raw/enhanced_relations_cleaned.csv'
    
    if not os.path.exists(relations_file):
        print(f"错误: 找不到关系数据文件 {relations_file}")
        return None
    
    # 加载CSV数据
    try:
        relations_df = pd.read_csv(relations_file)
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


def load_gene_pairs_data():
    """加载基因对关系数据"""
    import os
    import pandas as pd
    import json
    from collections import defaultdict
    
    # 寻找基因对数据文件
    possible_paths = [
        'gene_pairs.csv',
        'gene_relationships.csv',
        os.path.join('data', 'processed', 'gene_pairs.csv'),
        os.path.join('data', 'raw', 'gene_pairs.csv'),
        os.path.join('..', 'data', 'processed', 'gene_pairs.csv')
    ]
    
    gene_pairs_file = None
    for path in possible_paths:
        if os.path.exists(path):
            gene_pairs_file = path
            break
    
    # 如果找到CSV文件，加载为DataFrame再转为字典
    if gene_pairs_file and gene_pairs_file.endswith('.csv'):
        print(f"从CSV加载基因对数据: {gene_pairs_file}")
        df = pd.read_csv(gene_pairs_file)
        
        # 创建基因对字典
        gene_pairs = {}
        for _, row in df.iterrows():
            # 确保必要的列存在
            if 'Gene1' in df.columns and 'Gene2' in df.columns:
                # 使用基因名称作为键
                gene1 = row['Gene1'] if 'Gene1' in row else str(row.get('Gene1_ID', ''))
                gene2 = row['Gene2'] if 'Gene2' in row else str(row.get('Gene2_ID', ''))
                
                # 获取基因名称（如果有）
                gene1_name = row['Gene1_Name'] if 'Gene1_Name' in row else gene1
                gene2_name = row['Gene2_Name'] if 'Gene2_Name' in row else gene2
                
                # 创建键
                key = f"{gene1_name}_{gene2_name}"
                
                # 获取通路信息（如果有）
                pathway = row.get('Pathway', '') 
                
                # 存储
                gene_pairs[key] = {
                    'Gene1_Name': gene1_name,
                    'Gene2_Name': gene2_name,
                    'Pathway': pathway
                }
                
                # 添加其他属性
                for col in df.columns:
                    if col not in ['Gene1', 'Gene2', 'Gene1_Name', 'Gene2_Name']:
                        gene_pairs[key][col] = row[col]
                        
        print(f"加载了 {len(gene_pairs)} 个基因对关系")
        
    # 如果找不到文件，创建模拟数据用于测试    
    else:
        print("未找到基因对数据文件，使用替代方案...")
        # 创建一些常见的通路和基因关联用于测试
        pathways = {
            "Inflammatory_Pathway": ["IL1B", "TNF", "IL6", "NFKB1", "TLR4", "STAT1", "STAT3", "iNOS"],
            "Oxidative_Stress_Pathway": ["SOD1", "CAT", "GPX1", "NRF2", "KEAP1", "NOX2", "JUN", "FOS"],
            "Fibrosis_Pathway": ["TGFB1", "SMAD3", "CTGF", "COL1A1", "MMP2", "TIMP1", "α-smooth muscle actin"],
            "Apoptosis_Pathway": ["BCL2", "BAX", "CASP3", "CASP9", "TP53", "FAS", "FASLG", "CD95", "FasL"],
            "Liver_Metabolism_Pathway": ["CYP1A2", "CYP2E1", "CYP3A4", "GSTM1", "UGT1A1", "ABCB1", "PTEN"],
            "Kidney_Function_Pathway": ["AQP1", "AQP2", "SLC22A6", "SLC22A8", "UMOD", "KCNJ1", "NPHS1"]
        }
        
        # 创建基因对
        gene_pairs = {}
        for pathway_name, genes in pathways.items():
            for i in range(len(genes)):
                for j in range(i+1, len(genes)):
                    gene1 = genes[i]
                    gene2 = genes[j]
                    key = f"{gene1}_{gene2}"
                    gene_pairs[key] = {
                        'Gene1_Name': gene1,
                        'Gene2_Name': gene2,
                        'Pathway': pathway_name,
                        'Interaction_Type': 'Co-expression',
                        'Score': 0.75
                    }
                    
                # 也添加一些单基因的通路信息
                single_key = f"{genes[i]}_pathway"
                gene_pairs[single_key] = {
                    'Gene1_Name': genes[i],
                    'Gene2_Name': 'NA',
                    'Pathway': pathway_name,
                    'Interaction_Type': 'Pathway_Member',
                    'Score': 1.0
                }
                
        print(f"创建了 {len(gene_pairs)} 个模拟基因对关系")
        
    return gene_pairs



