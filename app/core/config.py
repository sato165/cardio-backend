from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Rutas a los artefactos del pipeline real (Colombia) — k=4 clusters
    MODEL_PATH:    str = "models/random_forest.pkl"
    SCALER_PATH:   str = "models/scaler.pkl"
    PCA_PATH:      str = "models/pca.pkl"
    IMPUTER_PATH:  str = "models/imputer.pkl"
    COLUMNAS_PATH: str = "models/columnas_modelo.pkl"

    debug: bool = True
    allowed_origins: str = "http://localhost:5173"
    max_upload_size_mb: int = 5

    class Config:
        env_file = ".env"


settings = Settings()