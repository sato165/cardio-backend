import pandas as pd
from app.ml.model_loader import get_artifacts

# Nombres y descripciones de clusters (del notebook real)
NOMBRES_CLUSTERS = {
    0: "Cardio-renal",
    1: "Cardiovascular Inflamatorio",
    2: "Bajo Riesgo"
}

DESCRIPCIONES_CLUSTERS = {
    0: ("Perfil de disfunción renal con alteración lipídica. "
        "Se recomienda evaluación nefrológica y manejo agresivo del colesterol."),
    1: ("Perfil inflamatorio-cardiovascular: presión arterial elevada y glucosa "
        "en rango de prediabetes. Seguimiento cardiológico y control metabólico prioritarios."),
    2: ("Perfil de bajo riesgo cardiovascular. "
        "Se recomienda mantener hábitos saludables y revisión anual de rutina.")
}


def predecir(features: pd.DataFrame) -> dict:
    """
    Recibe un DataFrame con las 22 features (ya preprocesadas),
    aplica escalado + PCA y devuelve el cluster predicho,
    las probabilidades y la interpretación clínica.
    """
    artifacts = get_artifacts()
    model = artifacts["model"]
    scaler = artifacts["scaler"]
    pca = artifacts["pca"]

    # 1. Escalar
    scaled = scaler.transform(features.values)

    # 2. Proyectar a PCA
    pca_proj = pca.transform(scaled)

    # 3. Predecir cluster y probabilidades
    cluster_id = int(model.predict(pca_proj)[0])
    probs = model.predict_proba(pca_proj)[0]  # array de 3 elementos

    # Construir lista ordenada de probabilidades por cluster
    probabilidades = []
    for idx, prob in enumerate(probs):
        probabilidades.append({
            "cluster_id": idx,
            "cluster_name": NOMBRES_CLUSTERS[idx],
            "probability": round(float(prob), 4)
        })

    return {
        "predicted_cluster": cluster_id,
        "cluster_name": NOMBRES_CLUSTERS[cluster_id],
        "description": DESCRIPCIONES_CLUSTERS[cluster_id],
        "probabilities": probabilidades,
    }