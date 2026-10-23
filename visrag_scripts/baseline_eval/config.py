"""Global configuration for visual RAG baseline evaluation.

This module provides path definitions, model configurations, and runtime
hyperparameters following Google Python Style.
"""

import os
from pathlib import Path


def _load_env_file(env_path: Path) -> None:
    """Loads key-value pairs from a .env file into os.environ if not already set."""
    if not env_path.exists():
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("\"'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception as e:
        print(f"[Warning] Failed to read .env file from {env_path}: {e}")


def _get_default_dataset_dir() -> Path:
    """Returns dataset directory depending on local or remote environment."""
    remote_path = Path("/home/wuzanqi/Experience")
    local_path = Path(r"D:\Experience")
    if remote_path.exists():
        return remote_path
    return local_path


def _get_default_model_dir() -> Path:
    """Returns model directory depending on local or remote environment."""
    remote_path = Path("/home/wuzanqi/ModelScope")
    local_path = Path(r"D:\ModelScope")
    if remote_path.exists():
        return remote_path
    return local_path


# Base Directories
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
BASE_DATA_DIR = _get_default_dataset_dir()
BASE_MODEL_DIR = _get_default_model_dir()

# Load .env from workspace root
ENV_FILE_PATH = WORKSPACE_ROOT / ".env"
_load_env_file(ENV_FILE_PATH)

# Dataset Paths
NATION_STANDARD_BENCH_DIR = BASE_DATA_DIR / "NationStandardBench"
DOC_BENCH_DIR = BASE_DATA_DIR / "docBench"

# Local Pre-trained Model Paths (ModelScope)
VISRAG_RET_PATH = BASE_MODEL_DIR / "VisRAG-Ret"
COLPALI_PATH = BASE_MODEL_DIR / "colpali-v1.3-hf"

# Evaluation Hyperparameters
TOP_K: int = 5
PDF_RENDER_DPI: int = 200

# Cache and Output Directories
CACHE_DIR = WORKSPACE_ROOT / "visrag_scripts" / "baseline_eval" / "cache"
CACHE_IMAGES_DIR = CACHE_DIR / "rendered_pages"
CACHE_INDEX_DIR = CACHE_DIR / "indices"
RESULTS_DIR = WORKSPACE_ROOT / "visrag_scripts" / "baseline_eval" / "results"

# Ensure runtime directories exist
for path in [CACHE_IMAGES_DIR, CACHE_INDEX_DIR, RESULTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Qwen API Settings (read from .env or environment variables, supporting common aliases)
QWEN_API_KEY: str = (
    os.getenv("DASHSCOPE_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or os.getenv("QWEN_API_KEY", "")
)
QWEN_BASE_URL: str = (
    os.getenv("DASHSCOPE_BASE_URL")
    or os.getenv("OPENAI_BASE_URL")
    or os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
)
QWEN_MODEL_NAME: str = (
    os.getenv("QWEN_VL_MODEL")
    or os.getenv("MODEL_NAME", "qwen-vl-max")
)
