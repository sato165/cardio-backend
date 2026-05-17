# cardio-backend/app/api/routes/explain.py
from fastapi import APIRouter
import numpy as np
from app.schemas.input_schema  import PredictionInput
from app.schemas.output_schema import ExplainabilityOutput, FeatureSHAP
from app.ml.explainability     import compute_shap_values
from app.ml.preprocessing      import preparar_features
from app.ml.predictor          import predecir, NOMBRES_CLUSTERS
from app.ml.model_loader       import get_artifacts

router = APIRouter()


@router.post(
    "/explain",
    response_model=ExplainabilityOutput,
    summary="Explicabilidad SHAP — k=4 clusters",
    description=(
        "Devuelve los valores SHAP en unidades originales para cada variable "
        "y cada uno de los 4 perfiles clínicos, junto con el cluster predicho."
    ),
)
def explain_prediction(datos: PredictionInput) -> ExplainabilityOutput:

    # 1. Preprocesar (winsorización + imputación)
    df_features = preparar_features(datos)              # DataFrame (1, n_features)

    # 2. Predicción principal
    resultado = predecir(df_features)

    # 3. SHAP en features originales
    features_array = df_features.values.astype(float)  # (1, n_features)
    shap_vals      = compute_shap_values(features_array)  # (1, n_features, 4)
    shap_sample    = shap_vals[0]                       # (n_features, 4)

    # Nombres de features en el orden de columnas_modelo.pkl
    feature_names = list(df_features.columns)
    paciente_vals = df_features.iloc[0].to_dict()

    # 4. Construir respuesta por cluster (4 perfiles)
    shap_by_cluster: dict[str, list[FeatureSHAP]] = {}
    for cluster_id, nombre in NOMBRES_CLUSTERS.items():
        shap_clase = shap_sample[:, cluster_id]         # (n_features,)
        shap_by_cluster[nombre] = [
            FeatureSHAP(
                feature=fname,
                shap_value=float(round(float(shap_clase[i]), 6)),
                feature_value=round(float(paciente_vals.get(fname, 0.0)), 4),
            )
            for i, fname in enumerate(feature_names)
        ]

    # 5. Valores base (expected_value) del explainer
    explainer = get_artifacts()["explainer"]
    expected  = explainer.expected_value
    n_classes = len(NOMBRES_CLUSTERS)   # 4

    if isinstance(expected, (list, np.ndarray)):
        arr = np.array(expected).flatten()
        if len(arr) == n_classes:
            base_values = {NOMBRES_CLUSTERS[i]: float(arr[i]) for i in range(n_classes)}
        else:
            # Fallback: repetir el valor escalar para cada cluster
            base_values = {nombre: float(arr[0]) for nombre in NOMBRES_CLUSTERS.values()}
    else:
        base_values = {nombre: float(expected) for nombre in NOMBRES_CLUSTERS.values()}

    return ExplainabilityOutput(
        predicted_cluster=resultado["predicted_cluster"],
        cluster_name=resultado["cluster_name"],
        shap_values=shap_by_cluster,
        base_values=base_values,
    )