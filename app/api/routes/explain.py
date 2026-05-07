# cardio-backend/app/api/routes/explain.py
from fastapi import APIRouter
import numpy as np
from app.schemas.input_schema import PredictionInput
from app.schemas.output_schema import ExplainabilityOutput, FeatureSHAP
from app.ml.explainability import compute_shap_values
from app.ml.preprocessing import preparar_features
from app.ml.predictor import predecir
from app.ml.model_loader import get_artifacts

router = APIRouter()

NOMBRES_CLUSTERS = {
    0: "Cardio-renal",
    1: "Cardiovascular Inflamatorio",
    2: "Bajo Riesgo"
}

FEATURE_NAMES = [
    "creatinina", "celulas_medias", "glucosa", "granulocitos", "hdl",
    "hematocrito", "hemoglobina", "ldl", "leucocitos", "linfocitos",
    "plaquetas", "trigliceridos", "edad", "sexo", "zona",
    "ap_hipertension", "ta_sistolica", "ta_diastolica", "peso", "talla",
    "imc", "TFG"
]

@router.post("/explain", response_model=ExplainabilityOutput)
def explain_prediction(datos: PredictionInput):
    # 1. Preprocesar y obtener predicción principal
    df_features = preparar_features(datos)           # DataFrame (1, 22) con clipping
    resultado = predecir(df_features)                # dict con cluster y probs

    # 2. Valores SHAP en features originales (array)
    features_array = df_features.values.astype(float) # (1, 22)
    shap_vals = compute_shap_values(features_array)   # (1, 22, 3)
    shap_sample = shap_vals[0]                        # (22, 3)

    # Extraer los valores del paciente (primer registro)
    paciente_vals = df_features.iloc[0].to_dict()

    # 3. Construir respuesta por cluster
    shap_by_cluster = {}
    for cluster_id, nombre in NOMBRES_CLUSTERS.items():
        shap_clase = shap_sample[:, cluster_id]       # (22,)
        features_shap = []
        for i, fname in enumerate(FEATURE_NAMES):
            features_shap.append(
                FeatureSHAP(
                    feature=fname,
                    shap_value=float(round(shap_clase[i], 6)),
                    feature_value=round(paciente_vals.get(fname, 0.0), 4)  # valor winsorizado
                )
            )
        shap_by_cluster[nombre] = features_shap

    # 4. Valores base (expected value) del modelo
    explainer = get_artifacts()["explainer"]
    expected = explainer.expected_value
    if isinstance(expected, list):
        base_values = {NOMBRES_CLUSTERS[i]: expected[i] for i in range(3)}
    else:
        if hasattr(expected, 'ndim') and expected.ndim == 1:
            base_values = {NOMBRES_CLUSTERS[i]: expected[i] for i in range(3)}
        else:
            base_values = {n: expected for n in NOMBRES_CLUSTERS.values()}

    return ExplainabilityOutput(
        predicted_cluster=resultado["predicted_cluster"],
        cluster_name=resultado["cluster_name"],
        shap_values=shap_by_cluster,
        base_values=base_values,
    )