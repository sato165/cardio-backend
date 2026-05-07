# cardio-backend/app/ml/model_loader.py
import scipy.stats.distributions          # ← nuevo
import scipy.stats._distn_infrastructure # ← nuevo
import sys                          # ← NUEVO
import os                           # ← NUEVO
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
    # Subimos dos niveles desde model_loader.py (app/ml) hasta la raíz del proyecto
    BASE_DIR = Path(__file__).resolve().parent.parent.parent

_artifacts = None

def get_artifacts():
    global _artifacts
    if _artifacts is None:
        _artifacts = _load_artifacts()
    return _artifacts

def _load_artifacts():
    paths = {
        "model": BASE_DIR / settings.MODEL_PATH,   # ← ajustado
        "scaler": BASE_DIR / settings.SCALER_PATH, # ← ajustado
        "pca": BASE_DIR / settings.PCA_PATH,       # ← ajustado
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

    explainer = shap.TreeExplainer(model)

    artifacts = {
        "model": model,
        "scaler": scaler,
        "pca": pca,
        "explainer": explainer,
    }

    print("✓ Artefactos cargados: RandomForest, StandardScaler, PCA, SHAP TreeExplainer")
    return artifacts