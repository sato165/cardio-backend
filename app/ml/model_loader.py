# cardio-backend/app/ml/model_loader.py
import joblib
import shap
from pathlib import Path
from app.core.config import settings

_artifacts = None

def get_artifacts():
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

    model = joblib.load(paths["model"])
    scaler = joblib.load(paths["scaler"])
    pca = joblib.load(paths["pca"])

    # SHAP TreeExplainer para RandomForest (exacto y rápido)
    explainer = shap.TreeExplainer(model)

    artifacts = {
        "model": model,
        "scaler": scaler,
        "pca": pca,
        "explainer": explainer,
    }

    print("✓ Artefactos cargados: RandomForest, StandardScaler, PCA, SHAP TreeExplainer")
    return artifacts