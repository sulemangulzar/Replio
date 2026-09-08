from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    pinecone_api_key : str
    openai_api_key : str | None = None
    embedding_model : str
    tavily_api_key : str
    pinecone_index_name : str
    pinecone_namespace : str
    top_k: int = 4

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings() #type: ignore
