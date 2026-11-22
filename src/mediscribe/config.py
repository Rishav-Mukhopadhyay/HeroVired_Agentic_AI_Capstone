"""Central settings, loaded once from the .env file.

Every other module does `from mediscribe.config import settings` instead of
reading os.environ directly, so all configuration lives in one place.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT_DIR / ".env", extra="ignore")

    # LLMs
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.5-flash"
    gemini_embed_model: str = "gemini-embedding-001"
    anthropic_api_key: str = ""
    claude_model: str = "claude-haiku-5-5"

    # Monitoring
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    # Local services
    fhir_base_url: str = "http://localhost:8080/fhir"
    postgres_url: str = "postgresql+psycopg://mediscribe:mediscribe@localhost:5432/mediscribe"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "mediscribe123"

    # Speech-to-text
    whisper_model: str = "base"


settings = Settings()
