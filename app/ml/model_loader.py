import joblib
from pathlib import Path
from app.core.config import settings

# Singleton que contendrá los tres artefactos
_artifacts = None


def get_artifacts():
    """
    Retorna un diccionario con el modelo, el scaler y el PCA.
    La primera vez los carga desde disco; luego usa la copia en memoria.
    """
    global _artifacts
    if _artifacts is None:
        _artifacts = _load_artifacts()
    return _artifacts


def _load_artifacts():
    paths = {
        "model": Path(settings.MODEL_PATH),
        "scaler": Path(settings.SCALER_PATH),
        "pca": Path(settings.PCA_PATH),
    }

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró el artefacto '{name}' en '{path}'. "
                "Verifica que los archivos .pkl estén en la carpeta models/."
            )

    artifacts = {
        "model": joblib.load(paths["model"]),
        "scaler": joblib.load(paths["scaler"]),
        "pca": joblib.load(paths["pca"]),
    }

    print("✓ Artefactos cargados: RandomForest, StandardScaler, PCA")
    return artifacts