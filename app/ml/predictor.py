import pandas as pd
from app.ml.model_loader import get_artifacts

# Mapeo de clusters — modelo final k=4 (notebook_proy_final)
NOMBRES_CLUSTERS = {
    0: "Cardiovascular",
    1: "Bajo riesgo",
    2: "Cardiometabólico",
    3: "Cardiorrenal"
}

DESCRIPCIONES_CLUSTERS = {
    0: (
        "Perfil con alteración lipídica predominante y riesgo cardiovascular elevado. "
        "Se recomienda seguimiento cardiológico, control estricto de lípidos y "
        "evaluación de factores de riesgo modificables."
    ),
    1: (
        "Perfil de bajo riesgo cardiovascular. "
        "Se recomienda mantener hábitos saludables, dieta balanceada y "
        "revisión anual de rutina."
    ),
    2: (
        "Perfil cardiometabólico: resistencia metabólica con alteración lipídica asociada. "
        "Se recomienda control metabólico, seguimiento endocrinológico y "
        "modificación de hábitos alimentarios."
    ),
    3: (
        "Perfil cardiorrenal: disfunción renal con alteración lipídica significativa. "
        "Se recomienda evaluación nefrológica urgente y manejo agresivo del colesterol."
    ),
}


def predecir(features: pd.DataFrame) -> dict:
    """
    Recibe un DataFrame con las features preprocesadas (winsorización +
    imputación ya aplicadas en preprocessing.py), aplica escalado + PCA
    y devuelve el cluster predicho, probabilidades e interpretación clínica.

    Pipeline:
        1. Estandarización con StandardScaler
        2. Proyección PCA (11 componentes)
        3. Predicción con RandomForest → cluster + probabilidades por cluster
    """
    artifacts = get_artifacts()
    model  = artifacts["model"]
    scaler = artifacts["scaler"]
    pca    = artifacts["pca"]

    # 1. Escalar
    scaled = scaler.transform(features.values)

    # 2. Proyectar con PCA (11 componentes)
    pca_proj = pca.transform(scaled)

    # 3. Predecir cluster y probabilidades
    cluster_id = int(model.predict(pca_proj)[0])
    probs      = model.predict_proba(pca_proj)[0]  # array de 4 elementos

    # Construir dict de probabilidades por fenotipo
    probabilidades = [
        {
            "cluster_id":   idx,
            "cluster_name": NOMBRES_CLUSTERS[idx],
            "probability":  round(float(prob), 4),
        }
        for idx, prob in enumerate(probs)
    ]

    return {
        "predicted_cluster": cluster_id,
        "cluster_name":      NOMBRES_CLUSTERS[cluster_id],
        "description":       DESCRIPCIONES_CLUSTERS[cluster_id],
        "probabilities":     probabilidades,
    }