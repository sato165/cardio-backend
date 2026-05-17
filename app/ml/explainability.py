# cardio-backend/app/ml/explainability.py
import numpy as np
from app.ml.model_loader import get_artifacts


def compute_shap_values(features_array: np.ndarray) -> np.ndarray:
    """
    Calcula valores SHAP para cada feature original y cada clase (4 clusters).

    Entrada:
        features_array: (n, n_features) — valores ya winsorizados e imputados,
                        en el orden de columnas_modelo.pkl

    Salida:
        shap_original: (n, n_features, 4) — contribuciones en unidades originales
    """
    artifacts = get_artifacts()
    scaler    = artifacts["scaler"]
    pca       = artifacts["pca"]
    explainer = artifacts["explainer"]

    n_classes = 4   # k=4 clusters — modelo final

    # 1. Estandarizar
    X_scaled = scaler.transform(features_array)        # (n, n_features)

    # 2. Proyectar con PCA (11 componentes)
    X_pca = pca.transform(X_scaled)                    # (n, 11)

    # 3. SHAP en espacio PCA con TreeExplainer
    shap_pca = explainer.shap_values(X_pca)            # lista de 4 arrays (n, 11)

    if isinstance(shap_pca, list):
        # Multiclase: lista de n_classes arrays (n, 11) → (n, 11, n_classes)
        shap_pca = np.stack(shap_pca, axis=-1)
    elif isinstance(shap_pca, np.ndarray) and shap_pca.ndim == 2:
        # Binario o single-output inesperado: expandir dimensión de clase
        shap_pca = shap_pca[:, :, np.newaxis]

    # Validar que tenemos las 4 clases
    if shap_pca.shape[-1] != n_classes:
        raise ValueError(
            f"Se esperaban {n_classes} clases en SHAP pero se obtuvieron "
            f"{shap_pca.shape[-1]}. Verifica que el modelo sea k=4."
        )

    # 4. Mapeo lineal PCA → features originales escaladas
    #    X_pca = X_scaled @ componentes.T
    #    ⟹ shap_scaled = shap_pca @ componentes   (por clase)
    #    einsum: n=batch, f=11 PCA, c=4 clases, p=n_features originales
    componentes  = pca.components_                     # (11, n_features)
    shap_scaled  = np.einsum('nfc,fp->npc', shap_pca, componentes)  # (n, n_features, 4)

    # 5. Deshacer estandarización: dividir por std de cada feature
    scale         = scaler.scale_                      # (n_features,)
    shap_original = shap_scaled / scale[:, np.newaxis] # (n, n_features, 4)

    return shap_original