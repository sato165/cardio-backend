import shap
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer

from app.ml.model_loader import get_model
from app.ml.preprocessing import FEATURE_ORDER

# ---------------------------------------------------------------------------
# Nombres legibles para el médico — uno por cada feature del modelo
# ---------------------------------------------------------------------------
NOMBRES_CLINICOS = {
    "age":            "Edad",
    "gender":         "Género",
    "height":         "Altura",
    "weight":         "Peso",
    "ap_hi":          "Presión sistólica",
    "ap_lo":          "Presión diastólica",
    "cholesterol":    "Colesterol",
    "gluc":           "Glucosa",
    "smoke":          "Tabaquismo",
    "alco":           "Consumo de alcohol",
    "active":         "Actividad física",
    "bmi":            "Índice de masa corporal (IMC)",
    "age_range":      "Grupo de edad",
    "bp_category":    "Categoría de presión arterial",
    "pulse_pressure": "Pulso de presión",
    "metabolic_risk": "Score de riesgo metabólico",
}

# Advertencia fija para variables con sesgo de subregistro documentado en el EDA
ADVERTENCIAS_SUBREGISTRO = {
    "smoke": (
        "Nota clínica: el tabaquismo está subregistrado en este dataset "
        "(solo 8.8% de positivos autorreportados). El impacto mostrado puede "
        "subestimar el riesgo real del tabaquismo."
    ),
    "alco": (
        "Nota clínica: el consumo de alcohol está subregistrado en este dataset "
        "(solo 5.4% de positivos autorreportados). El impacto mostrado puede "
        "subestimar el riesgo real del consumo de alcohol."
    ),
}


def explicar_shap(features: pd.DataFrame) -> list[dict]:
    """
    Calcula los valores SHAP para una predicción individual y los convierte
    en una lista de factores con nombre clínico, impacto numérico y descripción
    legible para el médico. Retorna los factores ordenados por impacto absoluto.
    """
    model = get_model()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    # Para clasificación binaria shap_values puede ser lista [clase_0, clase_1]
    if isinstance(shap_values, list):
        valores = shap_values[1][0]
    else:
        valores = shap_values[0]

    factores = []
    for feature, valor_shap in zip(FEATURE_ORDER, valores):
        nombre   = NOMBRES_CLINICOS.get(feature, feature)
        nivel    = _nivel_impacto(abs(valor_shap))
        descripcion = _descripcion_clinica(feature, valor_shap, features[feature].iloc[0])

        factor = {
            "factor":      nombre,
            "impacto":     round(float(valor_shap), 4),
            "descripcion": descripcion,
            "nivel":       nivel,
        }

        if feature in ADVERTENCIAS_SUBREGISTRO:
            factor["advertencia"] = ADVERTENCIAS_SUBREGISTRO[feature]

        factores.append(factor)

    # Ordenar por impacto absoluto descendente
    factores.sort(key=lambda x: abs(x["impacto"]), reverse=True)
    return factores


def explicar_lime(features: pd.DataFrame, datos_entrenamiento: pd.DataFrame) -> list[dict]:
    """
    Calcula la explicación LIME como alternativa a SHAP.
    Requiere una muestra del dataset de entrenamiento para construir
    el espacio de perturbación local.
    """
    model = get_model()

    explainer = LimeTabularExplainer(
        training_data=datos_entrenamiento[FEATURE_ORDER].values,
        feature_names=FEATURE_ORDER,
        class_names=["Sin cardio", "Con cardio"],
        mode="classification",
        random_state=42,
    )

    explicacion = explainer.explain_instance(
        data_row=features.values[0],
        predict_fn=model.predict_proba,
        num_features=len(FEATURE_ORDER),
    )

    factores = []
    for feature_label, peso in explicacion.as_list():
        # LIME usa strings como "feature <= valor" — extraemos el nombre
        nombre_feature = _extraer_nombre_feature(feature_label)
        nombre_clinico = NOMBRES_CLINICOS.get(nombre_feature, feature_label)
        nivel = _nivel_impacto(abs(peso))

        factor = {
            "factor":      nombre_clinico,
            "impacto":     round(float(peso), 4),
            "descripcion": feature_label,
            "nivel":       nivel,
        }

        if nombre_feature in ADVERTENCIAS_SUBREGISTRO:
            factor["advertencia"] = ADVERTENCIAS_SUBREGISTRO[nombre_feature]

        factores.append(factor)

    factores.sort(key=lambda x: abs(x["impacto"]), reverse=True)
    return factores


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _nivel_impacto(impacto_abs: float) -> str:
    """Clasifica el impacto absoluto en tres niveles clínicos."""
    if impacto_abs >= 0.10:
        return "crítico"
    elif impacto_abs >= 0.04:
        return "moderado"
    return "leve"


def _descripcion_clinica(feature: str, valor_shap: float, valor_paciente) -> str:
    """
    Genera una frase en español clínico describiendo cómo ese factor
    afecta el riesgo del paciente específico.
    """
    direccion = "aumenta" if valor_shap > 0 else "reduce"
    pct = abs(round(valor_shap * 100, 1))

    descripciones = {
        "ap_hi":          f"Su presión sistólica de {valor_paciente} mmHg {direccion} el riesgo en un {pct}%.",
        "ap_lo":          f"Su presión diastólica de {valor_paciente} mmHg {direccion} el riesgo en un {pct}%.",
        "bp_category":    f"Su categoría de presión arterial ({_nombre_bp(valor_paciente)}) {direccion} el riesgo en un {pct}%.",
        "pulse_pressure": f"Su pulso de presión de {valor_paciente} mmHg {direccion} el riesgo en un {pct}%.",
        "age":            f"Su edad de {valor_paciente} años {direccion} el riesgo en un {pct}%.",
        "age_range":      f"Su grupo de edad ({_nombre_age_range(valor_paciente)}) {direccion} el riesgo en un {pct}%.",
        "bmi":            f"Su IMC de {valor_paciente} ({_nombre_bmi(valor_paciente)}) {direccion} el riesgo en un {pct}%.",
        "cholesterol":    f"Su colesterol ({_nombre_ordinal_3(valor_paciente)}) {direccion} el riesgo en un {pct}%.",
        "gluc":           f"Su glucosa ({_nombre_ordinal_3(valor_paciente)}) {direccion} el riesgo en un {pct}%.",
        "metabolic_risk": f"Su score de riesgo metabólico de {valor_paciente}/3 {direccion} el riesgo en un {pct}%.",
        "smoke":          f"El tabaquismo reportado {direccion} el riesgo en un {pct}%.",
        "alco":           f"El consumo de alcohol reportado {direccion} el riesgo en un {pct}%.",
        "active":         f"La {'actividad física' if valor_paciente == 1 else 'inactividad física'} {direccion} el riesgo en un {pct}%.",
        "gender":         f"El género {direccion} el riesgo en un {pct}%.",
        "height":         f"La altura de {valor_paciente} cm {direccion} el riesgo en un {pct}%.",
        "weight":         f"El peso de {valor_paciente} kg {direccion} el riesgo en un {pct}%.",
    }

    return descripciones.get(
        feature,
        f"Este factor {direccion} el riesgo en un {pct}%."
    )


def _nombre_bp(codigo: int) -> str:
    return {1: "Normal", 2: "Elevada", 3: "HTA grado 1", 4: "HTA grado 2"}.get(codigo, str(codigo))


def _nombre_age_range(codigo: int) -> str:
    return {1: "menor de 40", 2: "40-49 años", 3: "50-59 años", 4: "60 o más"}.get(codigo, str(codigo))


def _nombre_bmi(bmi: float) -> str:
    if bmi < 18.5:
        return "bajo peso"
    elif bmi < 25:
        return "normal"
    elif bmi < 30:
        return "sobrepeso"
    elif bmi < 35:
        return "obesidad grado I"
    elif bmi < 40:
        return "obesidad grado II"
    return "obesidad grado III"


def _nombre_ordinal_3(codigo: int) -> str:
    return {1: "normal", 2: "por encima de lo normal", 3: "muy por encima de lo normal"}.get(codigo, str(codigo))


def _extraer_nombre_feature(lime_label: str) -> str:
    """Extrae el nombre del feature de un string de LIME como 'ap_hi <= 140'."""
    for feature in FEATURE_ORDER:
        if lime_label.startswith(feature):
            return feature
    return lime_label.split(" ")[0]
