# cardio-backend/app/ml/model_loader.py
import scipy.stats.distributions
import scipy.stats._distn_infrastructure
import sys
import os
import joblib
import shap
from pathlib import Path
from app.core.config import settings

# Determinar la raíz de la aplicación:
# - En desarrollo: directorio del proyecto (donde está main.py)
# - En el ejecutable: carpeta temporal de PyInstaller
if getattr(sys, 'frozen', False):
    BASE_DIR = Path(sys._MEIPASS)
else:
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

_artifacts = None

def get_artifacts():
    global _artifacts
    if _artifacts is None:
        _artifacts = _load_artifacts()
    return _artifacts

def _load_artifacts():
    paths = {
        "model":    BASE_DIR / settings.MODEL_PATH,
        "scaler":   BASE_DIR / settings.SCALER_PATH,
        "pca":      BASE_DIR / settings.PCA_PATH,
        "imputer":  BASE_DIR / settings.IMPUTER_PATH,
        "columnas": BASE_DIR / settings.COLUMNAS_PATH,
    }

    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró el artefacto '{name}' en '{path}'. "
                "Verifica que los archivos .pkl estén en la carpeta models/."
            )

    model    = joblib.load(paths["model"])
    scaler   = joblib.load(paths["scaler"])
    pca      = joblib.load(paths["pca"])
    imputer  = joblib.load(paths["imputer"])
    columnas = joblib.load(paths["columnas"])

    explainer = shap.TreeExplainer(model)

    artifacts = {
        "model":    model,
        "scaler":   scaler,
        "pca":      pca,
        "imputer":  imputer,
        "columnas": columnas,
        "explainer": explainer,
    }

    print("✓ Artefactos cargados: RandomForest, StandardScaler, PCA (11 componentes), KNNImputer, columnas_modelo, SHAP TreeExplainer")
    return artifacts