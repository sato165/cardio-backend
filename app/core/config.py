from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Nuevas rutas para los artefactos del pipeline real (Colombia)
    MODEL_PATH: str = "models/random_forest.pkl"
    SCALER_PATH: str = "models/scaler.pkl"
    PCA_PATH: str = "models/pca.pkl"

    debug: bool = True
    allowed_origins: str = "http://localhost:5173"
    max_upload_size_mb: int = 5

    class Config:
        env_file = ".env"


settings = Settings()