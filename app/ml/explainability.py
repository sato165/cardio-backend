# cardio-backend/app/ml/explainability.py
import numpy as np
from app.ml.model_loader import get_artifacts


def compute_shap_values(features_array: np.ndarray) -> np.ndarray:
    """
    Calcula valores SHAP aproximados en espacio de features originales para
    cada feature y cada una de las 4 clases del modelo final (k=4).

    Pipeline:
        features_array  →  StandardScaler  →  PCA(11)  →  RandomForest

    La retropropagación SHAP sigue el camino inverso:
        shap_pca  ---(PCA^T)-->  shap_scaled  ---(1/scale)-->  shap_original

    Parámetros
    ----------
    features_array : np.ndarray, shape (n, n_features)
        Valores ya winsorizados e imputados, en el orden de columnas_modelo.pkl.

    Retorna
    -------
    shap_original : np.ndarray, shape (n, n_features, 4)
        Contribuciones SHAP por feature y por clase, en unidades originales.
    """
    artifacts = get_artifacts()
    scaler    = artifacts["scaler"]
    pca       = artifacts["pca"]
    explainer = artifacts["explainer"]

    n_classes   = 4          # k=4 clusters — modelo final
    n_pca_comps = pca.n_components_   # 11

    # ── 1. Estandarizar ────────────────────────────────────────────────────
    X_scaled = scaler.transform(features_array)   # (n, n_features)

    # ── 2. Proyectar con PCA ───────────────────────────────────────────────
    X_pca = pca.transform(X_scaled)               # (n, n_pca_comps)

    # ── 3. SHAP en espacio PCA ─────────────────────────────────────────────
    # check_additivity=False evita falsos errores por la aproximación lineal
    # introducida al retropropagar a través de PCA.
    shap_raw = explainer.shap_values(X_pca, check_additivity=False)

    # Normalizar a array (n, n_pca_comps, n_classes)
    if isinstance(shap_raw, list):
        # Multiclase estándar: lista de n_classes arrays (n, n_pca_comps)
        if len(shap_raw) != n_classes:
            raise ValueError(
                f"Se esperaban {n_classes} clases pero TreeExplainer devolvió "
                f"{len(shap_raw)}. ¿El modelo es k=4?"
            )
        shap_pca = np.stack(shap_raw, axis=-1)   # (n, n_pca_comps, n_classes)
    elif isinstance(shap_raw, np.ndarray):
        if shap_raw.ndim == 3 and shap_raw.shape[-1] == n_classes:
            shap_pca = shap_raw                  # ya tiene el formato correcto
        elif shap_raw.ndim == 2:
            # Single-output inesperado → añadir dimensión de clase
            shap_pca = shap_raw[:, :, np.newaxis]
        else:
            raise ValueError(
                f"Formato SHAP inesperado: shape={shap_raw.shape}"
            )
    else:
        raise TypeError(f"Tipo SHAP no reconocido: {type(shap_raw)}")

    # ── 4. Retropropagar PCA → features escaladas ──────────────────────────
    # X_pca = X_scaled @ V^T   donde V = pca.components_ (n_pca_comps, n_features)
    # Por linealidad:
    #   shap_scaled[n, p, c] = Σ_f  shap_pca[n, f, c] * V[f, p]
    #
    # einsum: n=batch, f=n_pca_comps, c=n_classes, p=n_features
    V = pca.components_                          # (n_pca_comps, n_features)
    shap_scaled = np.einsum('nfc,fp->npc', shap_pca, V)   # (n, n_features, 4)

    # ── 5. Deshacer estandarización → unidades originales ─────────────────
    # X_scaled = (X - mean) / scale  →  shap_original = shap_scaled / scale
    # scale tiene shape (n_features,); necesitamos (1, n_features, 1) para
    # dividir correctamente el array (n, n_features, 4).
    scale = scaler.scale_                        # (n_features,)
    shap_original = shap_scaled / scale[np.newaxis, :, np.newaxis]

    return shap_original   # (n, n_features, 4)
