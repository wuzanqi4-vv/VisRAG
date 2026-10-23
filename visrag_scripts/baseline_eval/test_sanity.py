import sys
sys.path.append("/home/wuzanqi/VisRAG/visrag_scripts/baseline_eval")
import config
from data_loader import load_dataset

print("1. Config:")
print("  DATA DIR:", config.BASE_DATA_DIR)
print("  KEY:", config.QWEN_API_KEY[:8] + "...")
print("  URL:", config.QWEN_BASE_URL)
print("  MODEL:", config.QWEN_MODEL_NAME)

print("2. Data Loading:")
ns_items = load_dataset("nation_standard_bench", config.NATION_STANDARD_BENCH_DIR)
print("  NationStandardBench loaded count:", len(ns_items))

doc_items = load_dataset("doc_bench", config.DOC_BENCH_DIR)
print("  DocBench loaded count:", len(doc_items))
