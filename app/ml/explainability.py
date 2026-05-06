# cardio-backend/app/ml/explainability.py
import numpy as np
from app.ml.model_loader import get_artifacts

def compute_shap_values(features_array: np.ndarray):
    """
    Calcula los valores SHAP para cada muestra, cada feature (22) y cada clase (3).
    
    features_array: array de shape (n, 22) con valores crudos ya winsorizados.
    Retorna: array de shape (n, 22, 3) con los valores SHAP en unidades originales.
    """
    artifacts = get_artifacts()
    scaler = artifacts["scaler"]
    pca = artifacts["pca"]
    explainer = artifacts["explainer"]

    # 1. Escalar
    X_scaled = scaler.transform(features_array)      # (n, 22)

    # 2. PCA
    X_pca = pca.transform(X_scaled)                  # (n, 11)

    # 3. SHAP en espacio PCA (TreeExplainer, muy rápido)
    shap_pca = explainer.shap_values(X_pca)          # lista de 3 arrays (n, 11) para multiclass
    if isinstance(shap_pca, list):
        shap_pca = np.stack(shap_pca, axis=-1)       # (n, 11, 3)   (n=batch, f=11 PCA, c=3 clases)

    # 4. Mapeo lineal de vuelta a las features originales (escaladas)
    componentes = pca.components_                    # (11, 22)
    # X_pca = X_scaled @ componentes.T   =>   shap_scaled = shap_pca @ componentes   (por clase)
    # Vectorizado con einsum: n=batch, f=11 PCA, c=3 clases, p=22 features originales
    shap_scaled = np.einsum('nfc,fp->npc', shap_pca, componentes)   # (n, 22, 3)

    # 5. Deshacer el escalado: contribución en unidades originales = shap_scaled / std
    scale = scaler.scale_                            # (22,)
    shap_original = shap_scaled / scale[:, np.newaxis]  # broadcasting (n, 22, 3)

    return shap_original