import os
from pathlib import Path


def _find_env_file() -> Path | None:
    current = Path(__file__).resolve()
    for folder in [current.parent] + list(current.parents):
        candidate = folder / ".env"
        if candidate.exists():
            return candidate
    return None


def _load_dotenv() -> None:
    env_file = _find_env_file()
    if env_file is None:
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable: {name}. "
            f"Please configure it in .env."
        )
    return value


_load_dotenv()

OPENAI_API_KEY = _require_env("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "deepseek-chat").strip()
HF_ENDPOINT = os.getenv("HF_ENDPOINT", "https://hf-mirror.com").strip()
HF_HOME = os.getenv("HF_HOME", "").strip()
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base").strip()

os.environ["HF_ENDPOINT"] = HF_ENDPOINT
if HF_HOME:
    os.environ["HF_HOME"] = HF_HOME
