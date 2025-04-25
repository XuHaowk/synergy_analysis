"""
配置文件: 设置路径和参数
"""

import os

# 数据路径配置
DATA_DIR = 'data'
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
RESULTS_DIR = 'results'

# 输入文件路径
FOCAL_ENTITIES_FILE = os.path.join(RAW_DATA_DIR, 'focal_entities.csv')
ENHANCED_RELATIONS_FILE = os.path.join(RAW_DATA_DIR, 'enhanced_relations_cleaned.csv')
GENE_PAIRS_FILE = os.path.join(RAW_DATA_DIR, 'gene_pairs.csv')
GENE_PAIRS_DB_FILE = os.path.join(RAW_DATA_DIR, 'gene_pairs_db_relations.csv')
GENE_PAIRS_PUBMED_FILE = os.path.join(RAW_DATA_DIR, 'gene_pairs_pubmed_relations.csv')
TCMSP_DB_FILE = os.path.join(RAW_DATA_DIR, 'TCMSP_Database.xlsx')

# 分析参数
PROBABILITY_THRESHOLD = 0.7  # 关系概率阈值
MIN_COMMON_TARGETS = 3       # 最小共同靶点数量
SIGNIFICANT_TARGET_THRESHOLD = 0.8  # 重要靶点概率阈值

# 实体ID配置 (这些可能需要根据实际数据调整)
BAICALIN_KEYWORDS = ['黄芩苷', '黄苷', 'Baicalin']
TETRANDRINE_KEYWORDS = ['粉防己碱', '汉防己甲素', 'Tetrandrine']
SILICOSIS_KEYWORDS = ['硅肺病', '矽肺', 'Silicosis']
HEPATOTOX_KEYWORDS = ['肝毒性', 'Hepatotoxicity']
NEPHROTOX_KEYWORDS = ['肾毒性', 'Nephrotoxicity']

# 创建必要的目录
os.makedirs(RAW_DATA_DIR, exist_ok=True)
os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_DIR, 'relationships'), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DATA_DIR, 'targets'), exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'figures'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, 'tables'), exist_ok=True)