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
    Calculate indirect relationship probabilities based on an improved PSR algorithm that addresses
    probability saturation issues through diversity-weighted path analysis, context-aware clustering,
    and biological pathway-specific normalization.
    
    Parameters:
    source_target_relations - Source-to-target relationships data (e.g., drug-gene)
    target_target_relations - Target-to-target relationships data (e.g., gene-disease)
    
    Returns:
    DataFrame of indirect relationships with probabilities
    """
    import pandas as pd
    import numpy as np
    import json
    import os
    import math
    from collections import Counter
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    
    print("\n===== Starting Indirect Relationship Probability Calculation =====")
    print(f"Source data: {source_target_relations.shape[0]} rows × {source_target_relations.shape[1]} columns")
    print(f"Target data: {target_target_relations.shape[0]} rows × {target_target_relations.shape[1]} columns")
    
    # Load relationship type mapping
    rel_type_map = {}
    try:
        # Try multiple possible paths
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
            print(f"Successfully loaded {len(rel_type_map)} relationship type mappings, location: {rel_type_file}")
            
            # Display some sample relationship types
            sample_rel_types = list(rel_type_map.items())[:5]
            for rel_id, rel_data in sample_rel_types:
                print(f"  Relation ID:{rel_id} -> {rel_data['name']} (Correlation:{rel_data['correlation']}, Strength:{rel_data['precision']})")
        else:
            print(f"Warning: Relationship type mapping file not found. Tried paths: {', '.join(possible_paths)}")
            # Create basic relationship type mappings as fallback
            rel_type_map = {
                '2': {'name': 'Positive_Correlation', 'correlation': 1, 'precision': 4, 'causative': True, 'protective': False},
                '3': {'name': 'Negative_Correlation', 'correlation': -1, 'precision': 4, 'causative': False, 'protective': True},
                '11': {'name': 'Causes', 'correlation': 1, 'precision': 3, 'causative': True, 'protective': False},
                '16': {'name': 'Treats', 'correlation': -1, 'precision': 3, 'causative': False, 'protective': True}
            }
            print(f"Using built-in basic relationship type mappings: {len(rel_type_map)} relation types")
    except Exception as e:
        print(f"Error loading relationship type mappings: {e}")
        # Create basic relationship type mappings as fallback
        rel_type_map = {
            '2': {'name': 'Positive_Correlation', 'correlation': 1, 'precision': 4, 'causative': True, 'protective': False},
            '3': {'name': 'Negative_Correlation', 'correlation': -1, 'precision': 4, 'causative': False, 'protective': True}
        }
        print(f"Using default relationship type mappings: {len(rel_type_map)} relation types")
    
    # Data preparation and statistics
    source_columns = source_target_relations.columns.tolist()
    target_columns = target_target_relations.columns.tolist()
    print(f"Source data columns: {source_columns}")
    print(f"Target data columns: {target_columns}")
    
    # Ensure data type consistency
    source_target_relations = source_target_relations.copy()
    target_target_relations = target_target_relations.copy()
    
    # Convert IDs to strings for matching
    source_target_relations['Source_ID'] = source_target_relations['Source_ID'].astype(str)
    source_target_relations['Target_ID'] = source_target_relations['Target_ID'].astype(str)
    target_target_relations['Source_ID'] = target_target_relations['Source_ID'].astype(str)
    target_target_relations['Target_ID'] = target_target_relations['Target_ID'].astype(str)
    
    # Extract unique drug and disease IDs (considering bidirectional relationships)
    # Drugs can be in Source_ID or Target_ID with type Chemical
    drug_ids_as_source = source_target_relations[source_target_relations['Source_Type'] == 'Chemical']['Source_ID'].unique()
    drug_ids_as_target = source_target_relations[source_target_relations['Target_Type'] == 'Chemical']['Target_ID'].unique()
    drug_ids = np.union1d(drug_ids_as_source, drug_ids_as_target)
    
    # Diseases can be in Source_ID or Target_ID with type Disease
    disease_ids_as_source = target_target_relations[target_target_relations['Source_Type'] == 'Disease']['Source_ID'].unique()
    disease_ids_as_target = target_target_relations[target_target_relations['Target_Type'] == 'Disease']['Target_ID'].unique()
    disease_ids = np.union1d(disease_ids_as_source, disease_ids_as_target)
    
    print(f"Found {len(drug_ids)} drug entities: {drug_ids[:3]}...")
    print(f"Found {len(disease_ids)} disease entities: {disease_ids[:3]}...")
    
    # Get all drug targets (considering bidirectional relationships)
    drug_targets = set()
    for _, row in source_target_relations.iterrows():
        # Drug is source, gene is target
        if row['Source_Type'] == 'Chemical' and row['Target_Type'] == 'Gene':
            drug_targets.add(row['Target_ID'])
        # Gene is source, drug is target (reverse relationship)
        elif row['Source_Type'] == 'Gene' and row['Target_Type'] == 'Chemical':
            drug_targets.add(row['Source_ID'])
    
    # Get all disease targets (considering bidirectional relationships)
    disease_targets = set()
    for _, row in target_target_relations.iterrows():
        # Disease is source, gene is target
        if row['Source_Type'] == 'Disease' and row['Target_Type'] == 'Gene':
            disease_targets.add(row['Target_ID'])
        # Gene is source, disease is target (reverse relationship)
        elif row['Source_Type'] == 'Gene' and row['Target_Type'] == 'Disease':
            disease_targets.add(row['Source_ID'])
    
    # Find common targets
    intermediate_targets = drug_targets.intersection(disease_targets)
    print(f"Found {len(intermediate_targets)} common intermediate targets")
    if len(intermediate_targets) > 0:
        print(f"Sample intermediate targets: {list(intermediate_targets)[:5]}...")
    
    # Initialize results list
    indirect_relations = []
    
    # Calculate indirect relationships for each drug-disease pair
    drug_disease_pairs_processed = 0
    valid_paths_found = 0
    
    # Path weight calculation function with toxicity awareness
    def calculate_path_weight(rel_type_id, rel_type, prob_value, entity_pair_context):
        """Calculate path weight based on relationship type, probability value, and biological context"""
        # Decode the entity pair context
        source_type, target_type, is_toxicity_context = entity_pair_context
        
        # Base weight
        weight = 1.0
        
        # Weight adjustment based on relationship type
        if rel_type_id in rel_type_map:
            # Increase weight for high precision relationships
            precision = rel_type_map[rel_type_id]['precision']
            weight *= (1.0 + precision * 0.1)  # Add 0.1x weight per precision point
            
            # Context-aware weighting
            if is_toxicity_context:
                # For toxicity contexts, causative relationships are more important
                if rel_type_map[rel_type_id]['causative']:
                    weight *= 1.2
                # Chemical→Gene and Gene→Toxicity having the same direction (both positive or both negative)
                # is weighted more for toxicity-causing paths
                if source_type == 'Chemical' and target_type == 'Gene':
                    if rel_type_map[rel_type_id]['correlation'] == 1:
                        weight *= 1.15
            else:
                # For therapeutic contexts, protective relationships are more important
                if rel_type_map[rel_type_id]['protective']:
                    weight *= 1.2
        elif rel_type == "Positive_Correlation" or rel_type == "Causes":
            weight *= 1.3
            if is_toxicity_context:
                weight *= 1.2
        elif rel_type == "Negative_Correlation" or rel_type == "Treats":
            weight *= 1.3
            if not is_toxicity_context:
                weight *= 1.2
        
        # Weight adjustment based on probability strength
        strength = abs(prob_value)
        weight *= (0.5 + strength)  # Higher strength, higher weight
        
        return weight
    
    # Evaluate toxicity context function
    def is_toxicity_context(disease_name):
        """Determine if a disease context represents a toxicity condition"""
        toxicity_keywords = ['toxicity', 'toxicities', 'poisoning', 'injury', 'damage', 
                            'failure', '毒性', '损伤', '中毒', '衰竭']
        return any(keyword in disease_name.lower() for keyword in toxicity_keywords)
    
    # Cluster paths by similarity function with diversity boost
    def cluster_paths(path_details, n_clusters=8, max_paths_per_cluster=3):
        """Cluster paths by biological similarity and select diverse representatives from each cluster"""
        if len(path_details) <= n_clusters:
            return path_details  # No need to cluster if fewer paths than clusters
            
        # Extract path features for clustering
        features = []
        for path in path_details:
            # Create feature vector for each path
            feature = [
                path['drug_gene_prob'], 
                path['gene_disease_prob'],
                1 if path['drug_gene_dir'] == 'positive' else -1,
                1 if path['gene_disease_dir'] == 'positive' else -1,
                path['path_weight'],
                path['weighted_probability']
            ]
            features.append(feature)
        
        # Normalize features
        try:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            # Use K-means clustering
            k = min(n_clusters, len(features))
            kmeans = KMeans(n_clusters=k, random_state=42)
            clusters = kmeans.fit_predict(scaled_features)
            
            # Get cluster diversity metrics
            cluster_sizes = Counter(clusters)
            cluster_diversity = {}
            
            for i in range(k):
                cluster_paths = [p for j, p in enumerate(path_details) if clusters[j] == i]
                if cluster_paths:
                    # Calculate diversity within cluster by looking at unique gene targets and relation types
                    unique_genes = len(set(p['intermediate_target_id'] for p in cluster_paths))
                    unique_rel_types = len(set(p['drug_gene_type'] + '_' + p['gene_disease_type'] for p in cluster_paths))
                    cluster_diversity[i] = (unique_genes / len(cluster_paths)) * (unique_rel_types / len(cluster_paths))
            
            # Select representative paths from each cluster, prioritizing diverse clusters
            representative_paths = []
            for i, diversity in sorted(cluster_diversity.items(), key=lambda x: x[1], reverse=True):
                cluster_paths = [p for j, p in enumerate(path_details) if clusters[j] == i]
                if cluster_paths:
                    # Sort by weighted probability first
                    sorted_paths = sorted(cluster_paths, key=lambda x: x['weighted_probability'], reverse=True)
                    
                    # Take top paths, but ensure at least one from smaller, high-diversity clusters
                    paths_to_take = min(max_paths_per_cluster, max(1, int(len(sorted_paths) * 0.3)))
                    selected_paths = sorted_paths[:paths_to_take]
                    
                    # Add diversity boost for paths with unique target genes
                    used_genes = set()
                    for p in selected_paths:
                        if p['intermediate_target_id'] not in used_genes:
                            p['weighted_probability'] *= 1.1  # Diversity boost
                            used_genes.add(p['intermediate_target_id'])
                    
                    representative_paths.extend(selected_paths)
            
            # Ensure we have paths
            if not representative_paths:
                return sorted(path_details, key=lambda x: x['weighted_probability'], reverse=True)[:n_clusters]
                
            return representative_paths
        except Exception as e:
            print(f"    Error during path clustering: {e}")
            # Fallback: return top paths by weighted probability
            return sorted(path_details, key=lambda x: x['weighted_probability'], reverse=True)[:n_clusters]
    
    # Enhanced probability calculation with reduced saturation and biological pathway integration
    def calculate_combined_probability(paths, alpha=0.3, beta=0.15, is_toxicity=False):
        """Calculate combined probability with reduced saturation and pathway-specific weights"""
        if not paths:
            return 0.0
        
        # Group paths by biological subsystems to model pathway effects
        pathway_groups = {}
        for path in paths:
            target = path['intermediate_target_id']
            target_name = path['intermediate_target_name'].lower()
            
            # Assign to a broad biological pathway category
            if any(kw in target_name for kw in ['nf-kb', 'tnf', 'il-', 'ifn', 'tlr']):
                pathway = 'inflammation'
            elif any(kw in target_name for kw in ['casp', 'bcl', 'bax', 'apoptosis', 'p53']):
                pathway = 'apoptosis'
            elif any(kw in target_name for kw in ['cyp', 'p450', 'gst', 'ugt']):
                pathway = 'metabolism'
            elif any(kw in target_name for kw in ['mapk', 'erk', 'jnk', 'pi3k', 'akt']):
                pathway = 'signaling'
            elif any(kw in target_name for kw in ['sod', 'cat', 'gpx', 'nrf2']):
                pathway = 'antioxidant'
            else:
                pathway = 'other'
            
            if pathway not in pathway_groups:
                pathway_groups[pathway] = []
            pathway_groups[pathway].append(path)
        
        # Assign pathway importance weights based on context
        pathway_weights = {
            'inflammation': 1.2 if is_toxicity else 1.1,
            'apoptosis': 1.3 if is_toxicity else 1.0,
            'metabolism': 1.4 if is_toxicity else 0.9,
            'signaling': 1.0 if is_toxicity else 1.2,
            'antioxidant': 0.8 if is_toxicity else 1.3,
            'other': 1.0
        }
        
        # Calculate probability within each pathway using a non-saturating approach
        pathway_probs = []
        for pathway, group_paths in pathway_groups.items():
            # No more than 3 paths per pathway to prevent saturation
            top_paths = sorted(group_paths, key=lambda x: x['weighted_probability'], reverse=True)[:3]
            
            if top_paths:
                # Primary contribution - highest probability path
                primary_prob = top_paths[0]['weighted_probability'] * 0.7
                
                # Secondary contributions with diminishing returns
                secondary_prob = sum(p['weighted_probability'] * math.exp(-beta * i) * 0.3
                                     for i, p in enumerate(top_paths[1:], 1)) 
                
                # Combine with non-linear approach to avoid saturation
                combined = primary_prob + secondary_prob
                
                # Scale by pathway weight
                pathway_probs.append((combined * pathway_weights[pathway], len(group_paths)))
        
        if not pathway_probs:
            return 0.0
        
        # Balance pathway contributions
        total_weighted_prob = sum(prob * min(1.0, math.log1p(count) / 3.0) for prob, count in pathway_probs)
        total_weight = sum(min(1.0, math.log1p(count) / 3.0) for _, count in pathway_probs)
        
        if total_weight == 0:
            return 0.0
            
        avg_pathway_prob = total_weighted_prob / total_weight
        
        # Apply sigmoid transformation to get final probability (0.0-1.0 range)
        # Parameters adjusted to spread probabilities more evenly across the range
        return 1.0 / (1.0 + math.exp(-4.0 * (avg_pathway_prob - 0.6)))
    
    # Calculate dynamic probability cap based on evidence quality
    def calculate_dynamic_cap(path_count, path_diversity, relationship_type, is_toxicity):
        """Calculate dynamic probability cap based on evidence quality and relationship type"""
        # Base cap - lower for questionable relationships
        base_cap = 0.6 if is_toxicity else 0.65
        
        # Path count factor - cap rises with more paths, but with diminishing returns
        count_factor = min(0.12, math.log1p(path_count/20) / 12)
        
        # Diversity factor - higher diversity means more robust evidence
        diversity_factor = min(0.15, path_diversity * 0.1)
        
        # Relationship type modifiers
        rel_mod = 0.0
        if relationship_type == "therapeutic":
            rel_mod = 0.05  # Slightly higher cap for therapeutic relationships
        elif relationship_type == "toxicity":
            rel_mod = -0.02  # Slightly lower cap for toxicity claims
        
        # Calculate final cap
        final_cap = base_cap + count_factor + diversity_factor + rel_mod
        
        # Ensure reasonable bounds
        return max(0.5, min(0.9, final_cap))
    
    # Directly iterate through drug-disease pairs
    for drug_id in drug_ids:
        for disease_id in disease_ids:
            drug_disease_pairs_processed += 1
            
            # Get drug name
            drug_name_rows = source_target_relations[
                ((source_target_relations['Source_ID'] == drug_id) & 
                 (source_target_relations['Source_Type'] == 'Chemical')) |
                ((source_target_relations['Target_ID'] == drug_id) & 
                 (source_target_relations['Target_Type'] == 'Chemical'))
            ]
            drug_name = "Unknown Drug"
            if not drug_name_rows.empty:
                if 'Source_Name' in drug_name_rows.iloc[0] and drug_name_rows.iloc[0]['Source_Type'] == 'Chemical':
                    drug_name = drug_name_rows.iloc[0]['Source_Name']
                elif 'Target_Name' in drug_name_rows.iloc[0] and drug_name_rows.iloc[0]['Target_Type'] == 'Chemical':
                    drug_name = drug_name_rows.iloc[0]['Target_Name']
            
            # Get disease name
            disease_name_rows = target_target_relations[
                ((target_target_relations['Source_ID'] == disease_id) & 
                 (target_target_relations['Source_Type'] == 'Disease')) |
                ((target_target_relations['Target_ID'] == disease_id) & 
                 (target_target_relations['Target_Type'] == 'Disease'))
            ]
            disease_name = "Unknown Disease"
            if not disease_name_rows.empty:
                if 'Source_Name' in disease_name_rows.iloc[0] and disease_name_rows.iloc[0]['Source_Type'] == 'Disease':
                    disease_name = disease_name_rows.iloc[0]['Source_Name']
                elif 'Target_Name' in disease_name_rows.iloc[0] and disease_name_rows.iloc[0]['Target_Type'] == 'Disease':
                    disease_name = disease_name_rows.iloc[0]['Target_Name']
            
            # Check if this is a toxicity context
            is_toxic_context = is_toxicity_context(disease_name)
            relationship_type = "toxicity" if is_toxic_context else "therapeutic"
            
            # Debug output
            if drug_disease_pairs_processed <= 3 or drug_disease_pairs_processed % 100 == 0:
                print(f"Processing drug-disease pair #{drug_disease_pairs_processed}: {drug_name} (ID:{drug_id}) -> {disease_name} (ID:{disease_id})")
                if is_toxic_context:
                    print(f"  Detected toxicity context for: {disease_name}")
            
            # Initialize path probability lists
            path_probabilities = []
            path_weights = []
            path_details = []
            
            # Initialize example path calculations
            example_paths = []
            
            # Iterate through common targets looking for paths
            for target_id in intermediate_targets:
                # Find drug→gene relationships (considering bidirectional)
                drug_gene_relations = source_target_relations[
                    ((source_target_relations['Source_ID'] == drug_id) & 
                     (source_target_relations['Target_ID'] == target_id) &
                     (source_target_relations['Source_Type'] == 'Chemical') &
                     (source_target_relations['Target_Type'] == 'Gene')) |
                    ((source_target_relations['Source_ID'] == target_id) & 
                     (source_target_relations['Target_ID'] == drug_id) &
                     (source_target_relations['Source_Type'] == 'Gene') &
                     (source_target_relations['Target_Type'] == 'Chemical'))
                ]
                
                # Find gene→disease relationships (considering bidirectional)
                gene_disease_relations = target_target_relations[
                    ((target_target_relations['Source_ID'] == target_id) & 
                     (target_target_relations['Target_ID'] == disease_id) &
                     (target_target_relations['Source_Type'] == 'Gene') &
                     (target_target_relations['Target_Type'] == 'Disease')) |
                    ((target_target_relations['Source_ID'] == disease_id) & 
                     (target_target_relations['Target_ID'] == target_id) &
                     (target_target_relations['Source_Type'] == 'Disease') &
                     (target_target_relations['Target_Type'] == 'Gene'))
                ]
                
                # If path found, calculate probability
                if not drug_gene_relations.empty and not gene_disease_relations.empty:
                    for idx_dg, drug_gene in drug_gene_relations.iterrows():
                        for idx_gd, gene_disease in gene_disease_relations.iterrows():
                            # Get target name
                            target_name = ""
                            if drug_gene['Source_Type'] == 'Gene':
                                target_name = drug_gene['Source_Name']
                            else:
                                target_name = drug_gene['Target_Name']
                            
                            # Get probabilities
                            drug_gene_prob = abs(float(drug_gene['Probability'])) if 'Probability' in drug_gene else 0.5
                            gene_disease_prob = abs(float(gene_disease['Probability'])) if 'Probability' in gene_disease else 0.5
                            
                            # Get relationship type IDs
                            dg_type_id = str(drug_gene['Type_ID']) if 'Type_ID' in drug_gene else '0'
                            gd_type_id = str(gene_disease['Type_ID']) if 'Type_ID' in gene_disease else '0'
                            
                            # Get relationship types
                            dg_type = drug_gene['Type'] if 'Type' in drug_gene else "Unknown"
                            gd_type = gene_disease['Type'] if 'Type' in gene_disease else "Unknown"
                            
                            # Determine drug-gene relationship direction
                            drug_gene_dir = "positive"  # Default is positive
                            
                            # Use relationship type mapping to determine direction
                            if dg_type_id in rel_type_map:
                                if rel_type_map[dg_type_id]['correlation'] == -1:
                                    drug_gene_dir = "negative"
                            elif dg_type == "Negative_Correlation" or dg_type == "Treats":
                                drug_gene_dir = "negative"
                            
                            # Determine gene-disease relationship direction
                            gene_disease_dir = "positive"  # Default is positive
                            
                            # Use relationship type mapping to determine direction
                            if gd_type_id in rel_type_map:
                                if rel_type_map[gd_type_id]['correlation'] == -1:
                                    gene_disease_dir = "negative"
                            elif gd_type == "Negative_Correlation" or gd_type == "Treats":
                                gene_disease_dir = "negative"
                            
                            # Define entity pair context for weighting
                            drug_gene_context = ('Chemical', 'Gene', is_toxic_context)
                            gene_disease_context = ('Gene', 'Disease', is_toxic_context)
                            
                            # Calculate path weights
                            dg_weight = calculate_path_weight(dg_type_id, dg_type, drug_gene_prob, drug_gene_context)
                            gd_weight = calculate_path_weight(gd_type_id, gd_type, gene_disease_prob, gene_disease_context)
                            
                            # Adjust weights for toxicity vs therapeutic contexts
                            if is_toxic_context:
                                # For toxicity contexts:
                                # - Weight paths where drug upregulates a gene that causes toxicity higher
                                # - Weight paths where drug downregulates a gene that suppresses toxicity higher
                                if ((drug_gene_dir == "positive" and gene_disease_dir == "positive") or
                                    (drug_gene_dir == "negative" and gene_disease_dir == "negative")):
                                    path_weight = (dg_weight + gd_weight) * 1.2 / 2.0
                                else:
                                    path_weight = (dg_weight + gd_weight) * 0.8 / 2.0
                            else:
                                # For therapeutic contexts:
                                # - Weight paths where drug upregulates a gene that reduces disease higher
                                # - Weight paths where drug downregulates a gene that worsens disease higher
                                if ((drug_gene_dir == "positive" and gene_disease_dir == "negative") or
                                    (drug_gene_dir == "negative" and gene_disease_dir == "positive")):
                                    path_weight = (dg_weight + gd_weight) * 1.2 / 2.0
                                else:
                                    path_weight = (dg_weight + gd_weight) * 0.8 / 2.0
                            
                            # Calculate base path probability
                            base_path_prob = drug_gene_prob * gene_disease_prob
                            
                            # Ensure minimum probability
                            if base_path_prob < 0.001:
                                base_path_prob = 0.001
                            
                            # Adjust based on relationship direction
                            if is_toxic_context:
                                # For toxicity, same direction enhances toxicity
                                if drug_gene_dir == gene_disease_dir:
                                    path_prob = base_path_prob * 1.3
                                else:
                                    path_prob = base_path_prob * 0.7
                            else:
                                # For therapeutic, opposite direction enhances therapeutic effect
                                if drug_gene_dir != gene_disease_dir:
                                    path_prob = base_path_prob * 1.3
                                else:
                                    path_prob = base_path_prob * 0.7
                            
                            # Apply scaling factor to reduce saturation
                            scaling_factor = 0.2 + (0.6 / (1 + math.exp(len(path_probabilities) / 20)))
                            path_prob *= scaling_factor
                            
                            # Apply path weight
                            weighted_path_prob = path_prob * path_weight
                            
                            # Add to path lists
                            path_probabilities.append(path_prob)
                            path_weights.append(path_weight)
                            
                            # Record path details
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
                                'path_weight': path_weight,
                                'weighted_probability': weighted_path_prob
                            }
                            path_details.append(path_detail)
                            
                            # Collect a few example paths for detailed output
                            if len(example_paths) < 5:
                                example_path = {
                                    'target': target_name,
                                    'drug_gene': f"{drug_name}-{target_name} ({dg_type}): {drug_gene_prob:.4f}",
                                    'gene_disease': f"{target_name}-{disease_name} ({gd_type}): {gene_disease_prob:.4f}",
                                    'base_prob': f"{base_path_prob:.4f}",
                                    'direction_adjusted': f"{path_prob:.4f}",
                                    'weighted': f"{weighted_path_prob:.4f}",
                                    'weight': f"{path_weight:.2f}"
                                }
                                example_paths.append(example_path)
            
            # If paths found, calculate overall indirect probability
            if path_probabilities:
                valid_paths_found += 1
                
                # Output example path calculations
                print(f"\n  Found {len(path_probabilities)} valid paths for {drug_name}-{disease_name}")
                print(f"  Sample path calculations (showing first {min(5, len(example_paths))} of {len(path_probabilities)}):")
                for i, path in enumerate(example_paths[:5]):
                    print(f"    Path {i+1}: Via {path['target']}")
                    print(f"      {path['drug_gene']}")
                    print(f"      {path['gene_disease']}")
                    print(f"      Base prob: {path['base_prob']} → Direction adjusted: {path['direction_adjusted']} → Weight({path['weight']}): {path['weighted']}")
                
                # Calculate weighted path probability statistics
                weighted_probs = [p * w for p, w in zip(path_probabilities, path_weights)]
                avg_prob = sum(path_probabilities) / len(path_probabilities)
                avg_weight = sum(path_weights) / len(path_weights)
                avg_weighted_prob = sum(weighted_probs) / len(weighted_probs)
                
                # Get top weighted probability for cap calculation
                top_weighted_prob = max(weighted_probs) if weighted_probs else 0.5
                
                print(f"  Path statistics: Avg prob={avg_prob:.4f}, Avg weight={avg_weight:.2f}, Avg weighted prob={avg_weighted_prob:.4f}")
                
                # Path clustering and selection if many paths
                if len(path_details) > 10:
                    print(f"  Performing path clustering analysis on {len(path_details)} paths...")
                    # Determine optimal number of clusters based on path count
                    n_clusters = min(max(5, int(len(path_details) / 15)), 12)
                    clustered_paths = cluster_paths(path_details, n_clusters=n_clusters)
                    print(f"  Identified {len(clustered_paths)} representative paths from {n_clusters} clusters")
                    
                    # Calculate path diversity metrics
                    path_type_pairs = [(p['drug_gene_type'], p['gene_disease_type']) for p in path_details[:50]]
                    path_dir_pairs = [(p['drug_gene_dir'], p['gene_disease_dir']) for p in path_details[:50]]
                    unique_path_types = len(set(path_type_pairs))
                    unique_dir_pairs = len(set(path_dir_pairs))
                    type_diversity = unique_path_types / min(50, len(path_details))
                    dir_diversity = unique_dir_pairs / min(50, len(path_details))
                    
                    # Combined diversity score
                    path_diversity = (type_diversity + dir_diversity) / 2
                    
                    # Print representative paths
                    print(f"  Representative paths ({len(clustered_paths)} of {len(path_details)}):")
                    for i, path in enumerate(clustered_paths[:5]):
                        print(f"    Cluster rep {i+1}: {path['intermediate_target_name']}, " +
                              f"Drug→Gene: {path['drug_gene_type']}({path['drug_gene_dir']}), " +
                              f"Gene→Disease: {path['gene_disease_type']}({path['gene_disease_dir']}), " +
                              f"Weight: {path['path_weight']:.2f}")
                    
                    # Calculate probability using clustered paths with toxicity awareness
                    cluster_prob = calculate_combined_probability(clustered_paths, is_toxicity=is_toxic_context)
                    print(f"  Path diversity: {path_diversity:.3f} (from {unique_path_types} unique relationship types and {unique_dir_pairs} direction patterns)")
                    print(f"  Clustered path probability: {cluster_prob:.4f}")
                    
                    # Use clustered probability
                    final_prob = cluster_prob
                else:
                    # For few paths, use reduced saturation approach
                    # Define a non-linear function with reduced saturation
                    log_scale = 0.15  # Severe reduction to prevent saturation
                    
                    # Directly calculate individual path contributions
                    normalized_paths = []
                    for p, w in zip(path_probabilities, path_weights):
                        # Scale down probability to prevent saturation
                        norm_p = p * log_scale
                        # Apply non-linear transformation
                        transformed_p = math.log1p(norm_p * w)
                        normalized_paths.append(transformed_p)
                    
                    # Sum contributions with diminishing returns
                    if normalized_paths:
                        sum_contrib = sum(normalized_paths)
                        # Apply sigmoid to spread probabilities
                        log_transformed_prob = 1.0 / (1.0 + math.exp(-2.0 * (sum_contrib - 1.0)))
                    else:
                        log_transformed_prob = 0.0
                    
                    # Also calculate using combined probability function for comparison
                    direct_prob = calculate_combined_probability(path_details, is_toxicity=is_toxic_context)
                    
                    # Average both methods
                    final_prob = (log_transformed_prob + direct_prob) / 2
                    
                    # Calculate path diversity for few paths
                    path_type_pairs = [(p['drug_gene_type'], p['gene_disease_type']) for p in path_details]
                    path_dir_pairs = [(p['drug_gene_dir'], p['gene_disease_dir']) for p in path_details]
                    unique_path_types = len(set(path_type_pairs))
                    unique_dir_pairs = len(set(path_dir_pairs))
                    
                    type_diversity = unique_path_types / len(path_details) if path_details else 0
                    dir_diversity = unique_dir_pairs / len(path_details) if path_details else 0
                    path_diversity = (type_diversity + dir_diversity) / 2
                    
                    print(f"  Path diversity: {path_diversity:.3f} (from {unique_path_types} unique relationship types and {unique_dir_pairs} direction patterns)")
                    print(f"  Probability calculations:")
                    print(f"    Log-transformed: {log_transformed_prob:.4f}")
                    print(f"    Direct calculation: {direct_prob:.4f}")
                    print(f"    Final probability: {final_prob:.4f}")
                
                # Calculate dynamic probability cap
                prob_cap = calculate_dynamic_cap(
                    len(path_probabilities), 
                    path_diversity, 
                    relationship_type, 
                    is_toxic_context
                )
                
                print(f"  Dynamic probability cap: {prob_cap:.4f} (based on {len(path_probabilities)} paths, " + 
                      f"diversity {path_diversity:.3f}, relationship type: {relationship_type})")
                
                # Apply small path count factor with reduced impact
                path_count_factor = min(0.05, math.log1p(len(path_probabilities) / 100) / 30)
                adjusted_prob = min(prob_cap, final_prob + path_count_factor)
                
                print(f"    Path count reward ({path_count_factor:.3f}) → Final probability: {adjusted_prob:.4f}")
                
                # Add to results list
                indirect_relations.append({
                    'Source_ID': drug_id,
                    'Source_Name': drug_name,
                    'Target_ID': disease_id,
                    'Target_Name': disease_name,
                    'Relationship_Type': relationship_type,
                    'Indirect_Probability': adjusted_prob,
                    'Path_Count': len(path_probabilities),
                    'Path_Diversity': path_diversity,
                    'Path_Details': path_details,
                    'Probability_Cap': prob_cap,
                    'Is_Toxicity_Context': is_toxic_context
                })
    
    # Create result DataFrame
    result_df = pd.DataFrame(indirect_relations) if indirect_relations else pd.DataFrame()
    
    # Results statistics
    print(f"\nProcessed {drug_disease_pairs_processed} drug-disease pairs, found {valid_paths_found} valid paths.")
    
    if not result_df.empty:
        print(f"Results: {result_df.shape[0]} rows × {result_df.shape[1]} columns")
        print(f"Result columns: {result_df.columns.tolist()}")
        
        if 'Indirect_Probability' in result_df.columns:
            print(f"Indirect probability statistics: \n{result_df['Indirect_Probability'].describe()}")
            print(f"Highest indirect probability: {result_df['Indirect_Probability'].max():.4f}")
            if result_df.shape[0] > 1:
                print(f"Indirect probability standard deviation: {result_df['Indirect_Probability'].std():.4f}")
                print(f"Probability range: {result_df['Indirect_Probability'].min():.4f} - {result_df['Indirect_Probability'].max():.4f}")
                
        # Report statistics by relationship type
        if 'Relationship_Type' in result_df.columns:
            types = result_df['Relationship_Type'].unique()
            for rel_type in types:
                type_df = result_df[result_df['Relationship_Type'] == rel_type]
                print(f"\n{rel_type.capitalize()} relationship statistics:")
                print(f"  Count: {type_df.shape[0]} relationships")
                if 'Indirect_Probability' in type_df.columns:
                    print(f"  Probability range: {type_df['Indirect_Probability'].min():.4f} - {type_df['Indirect_Probability'].max():.4f}")
                    print(f"  Average probability: {type_df['Indirect_Probability'].mean():.4f}")
                    
    else:
        print("Warning: Calculation result is an empty DataFrame!")
    
    print("===== Indirect Relationship Probability Calculation Completed =====\n")
    return result_df


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






