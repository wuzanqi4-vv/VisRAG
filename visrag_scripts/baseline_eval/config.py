"""Explicit server configuration without import-time file writes."""
import os
from pathlib import Path
import shlex

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
TOP_K = 5
PDF_RENDER_DPI = 200


def load_settings(env_file: str | Path | None = None) -> dict:
    """Loads .env values; process environment takes precedence."""
    values = {}
    path = Path(env_file) if env_file else WORKSPACE_ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:]
            key, separator, value = line.partition("=")
            if not separator or not key.strip().isidentifier():
                raise ValueError(f"Invalid environment entry: {path}")
            values[key.strip()] = " ".join(shlex.split(value, comments=True))
    values.update(os.environ)

    def select(*names, default=""):
        return next((values[n] for n in names if values.get(n)), default)

    return {
        "api_key": select("CHAT_API_KEY", "DASHSCOPE_API_KEY",
                          "OPENAI_API_KEY", "QWEN_API_KEY"),
        "base_url": select("CHAT_BASE_URL", "DASHSCOPE_BASE_URL",
                           "OPENAI_BASE_URL", "QWEN_BASE_URL"),
        "model_name": select("CHAT_MODEL", "QWEN_VL_MODEL", "MODEL_NAME"),
        "api_format": select("CHAT_API_FORMAT", default="chat_completions"),
        "data_dir": Path(select("BENCHMARK_DATA_DIR",
                                default="/home/wuzanqi/Experience")),
        "model_dir": Path(select("BENCHMARK_MODEL_DIR",
                                 default="/home/wuzanqi/ModelScope")),
        "artifact_dir": Path(select(
            "BENCHMARK_ARTIFACT_DIR",
            default=str(WORKSPACE_ROOT / "artifacts" / "baseline_eval"))),
    }


def validate_api(settings: dict) -> None:
    """Rejects missing/placeholder settings before any expensive work."""
    for key in ("api_key", "base_url", "model_name"):
        value = settings[key]
        if not value or "replace_with_" in value or ".example" in value:
            raise ValueError(f"Configure {key} in server .env.")
    if settings["api_format"] != "chat_completions":
        raise ValueError("Use CHAT_API_FORMAT=chat_completions.")
    url = settings["base_url"].rstrip("/")
    if not url.startswith(("http://", "https://")):
        raise ValueError("CHAT_BASE_URL must be an HTTP(S) root.")
    if url.endswith("/chat/completions") or "@" in url or "?" in url:
        raise ValueError(
            "Use an API root without credentials/query parameters.")
