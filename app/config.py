"""
PHN Technology WhatsApp Agent — Configuration Module

Centralizes all environment variables and application settings.
Uses Pydantic Settings for validation and type safety.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Application configuration loaded from environment variables."""

    # --- App ---
    app_env: str = Field(default="development", description="Environment: development | production")
    log_level: str = Field(default="info", description="Logging level")

    # --- Ollama ---
    ollama_base_url: str = Field(
        default="http://ollama:11434",
        description="Ollama API base URL (use http://localhost:11434 for local dev)"
    )
    ollama_chat_model: str = Field(
        default="qwen2.5:7b-instruct",
        description="Ollama model for chat/reasoning"
    )
    ollama_embed_model: str = Field(
        default="nomic-embed-text",
        description="Ollama model for text embeddings"
    )

    # --- WhatsApp Business API ---
    whatsapp_verify_token: str = Field(
        default="phn-tech-verify-2024",
        description="Webhook verification token (you define this)"
    )
    whatsapp_access_token: str = Field(
        default="",
        description="Meta WhatsApp Business API access token"
    )
    whatsapp_phone_number_id: str = Field(
        default="",
        description="WhatsApp Business phone number ID"
    )

    # --- Storage Paths ---
    chroma_persist_dir: str = Field(
        default="/app/chroma_data",
        description="ChromaDB persistence directory"
    )
    sqlite_db_path: str = Field(
        default="/app/data/phn_agent.db",
        description="SQLite database file path"
    )
    knowledge_base_dir: str = Field(
        default="/app/knowledge_base",
        description="Directory containing knowledge base JSON files"
    )

    # --- Agent Settings ---
    rag_top_k: int = Field(default=4, description="Number of documents to retrieve")
    llm_temperature: float = Field(default=0.3, description="LLM temperature for generation")
    max_context_messages: int = Field(default=10, description="Max history messages to include")

    # --- Company Info ---
    company_name: str = Field(default="PHN Technology Pvt Limited")
    company_location: str = Field(default="Pune, Maharashtra")
    company_type: str = Field(default="EdTech")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """Returns cached settings instance. Call once at startup."""
    return Settings()
