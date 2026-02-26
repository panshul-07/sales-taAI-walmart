from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Walmart AI Sales Chatbot"
    environment: str = "development"
    api_prefix: str = "/api/v1"

    postgres_user: str = "walmart"
    postgres_password: str = "walmart"
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_db: str = "walmart_sales"

    redis_url: str = "redis://redis:6379/0"
    chroma_host: str = "chroma"
    chroma_port: int = 8000

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    jwt_secret: str = "change_me"
    jwt_algo: str = "HS256"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
