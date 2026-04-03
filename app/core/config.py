from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    best_model: str = "random_forest"
    model_rf_path: str = "models/random_forest.pkl"
    model_xgb_path: str = "models/xgboost_model.pkl"
    debug: bool = True
    allowed_origins: str = "http://localhost:5173"
    max_upload_size_mb: int = 5

    class Config:
        env_file = ".env"


settings = Settings()