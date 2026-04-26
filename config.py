from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import getpass

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    openai_api_key: SecretStr = Field(
        validation_alias=AliasChoices("OPENAI_API_KEY", "api_key", "API_KEY")
    )
    model_name: str = Field(default="gpt-4o-mini", validation_alias=AliasChoices("MODEL_NAME", "model_name"))

    # Langfuse
    langfuse_public_key: SecretStr = Field(validation_alias=AliasChoices("LANGFUSE_PUBLIC_KEY"))
    langfuse_secret_key: SecretStr = Field(validation_alias=AliasChoices("LANGFUSE_SECRET_KEY"))
    langfuse_base_url: str = Field(
        default="https://us.cloud.langfuse.com",
        validation_alias=AliasChoices("LANGFUSE_BASE_URL"),
    )
    langfuse_prompt_label: str = Field(default="production", validation_alias=AliasChoices("LANGFUSE_PROMPT_LABEL"))
    langfuse_prompt_cache_ttl_seconds: int = Field(default=300, validation_alias=AliasChoices("LANGFUSE_PROMPT_CACHE_TTL_SECONDS"))
    langfuse_default_user_id: str = Field(
        default_factory=lambda: getpass.getuser() or "local-user",
        validation_alias=AliasChoices("LANGFUSE_DEFAULT_USER_ID", "LANGFUSE_USER_ID"),
    )
    langfuse_environment: str = Field(default="development", validation_alias=AliasChoices("LANGFUSE_ENVIRONMENT"))
    langfuse_default_tags: str = Field(default="multi-agent-research,hw12", validation_alias=AliasChoices("LANGFUSE_DEFAULT_TAGS"))

    planner_prompt_name: str = Field(default="mas-planner-system", validation_alias=AliasChoices("LANGFUSE_PLANNER_PROMPT_NAME"))
    researcher_prompt_name: str = Field(default="mas-researcher-system", validation_alias=AliasChoices("LANGFUSE_RESEARCHER_PROMPT_NAME"))
    critic_prompt_name: str = Field(default="mas-critic-system", validation_alias=AliasChoices("LANGFUSE_CRITIC_PROMPT_NAME"))
    supervisor_prompt_name: str = Field(default="mas-supervisor-system", validation_alias=AliasChoices("LANGFUSE_SUPERVISOR_PROMPT_NAME"))
    report_revision_prompt_name: str = Field(default="mas-report-revision-system", validation_alias=AliasChoices("LANGFUSE_REPORT_REVISION_PROMPT_NAME"))

    # Web search
    max_search_results: int = 5
    max_search_content_length: int = 4000
    max_url_content_length: int = 8000

    # RAG
    embedding_model: str = "text-embedding-3-small"
    data_dir: str = "data"
    index_dir: str = "index"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    retrieval_top_k: int = 8
    rerank_top_n: int = 3
    semantic_k: int = 8
    bm25_k: int = 8
    reranker_model: str = "BAAI/bge-reranker-base"

    # Runtime
    output_dir: str = "output"
    max_iterations: int = 8
    max_revision_rounds: int = 2
    request_timeout_seconds: int = 30
    report_preview_chars: int = 1200

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def data_path(self) -> Path:
        return BASE_DIR / self.data_dir

    @property
    def index_path(self) -> Path:
        return BASE_DIR / self.index_dir

    @property
    def output_path(self) -> Path:
        return BASE_DIR / self.output_dir

    @property
    def langfuse_tags(self) -> list[str]:
        return [tag.strip() for tag in self.langfuse_default_tags.split(",") if tag.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


APP_TITLE = "Multi-Agent Research System"
SEPARATOR = "=" * 68
