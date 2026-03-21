import pandas as pd
from app.schemas.input_schema import CardiovascularInput

# Orden exacto de features con el que fue entrenado el modelo.
# Si se cambia aquí, debe cambiarse también en el notebook de entrenamiento.
FEATURE_ORDER = [
    "age", "gender", "height", "weight",
    "ap_hi", "ap_lo", "cholesterol", "gluc",
    "smoke", "alco", "active",
    "bmi", "age_range", "bp_category",
    "pulse_pressure", "metabolic_risk",
]


def preparar_features(datos: CardiovascularInput) -> pd.DataFrame:
    """
    Recibe los datos validados del schema de entrada y retorna un
    DataFrame con exactamente las mismas columnas y orden que usó
    el modelo durante el entrenamiento.
    """
    age_years = datos.age_days / 365.25
    bmi       = datos.weight / (datos.height / 100) ** 2

    fila = {
        "age":            round(age_years, 1),
        "gender":         datos.gender,
        "height":         datos.height,
        "weight":         datos.weight,
        "ap_hi":          datos.ap_hi,
        "ap_lo":          datos.ap_lo,
        "cholesterol":    datos.cholesterol,
        "gluc":           datos.gluc,
        "smoke":          datos.smoke,
        "alco":           datos.alco,
        "active":         datos.active,
        "bmi":            round(bmi, 1),
        "age_range":      _calcular_age_range(age_years),
        "bp_category":    _calcular_bp_category(datos.ap_hi, datos.ap_lo),
        "pulse_pressure": datos.ap_hi - datos.ap_lo,
        "metabolic_risk": _calcular_metabolic_risk(datos.cholesterol, datos.gluc, bmi),
    }

    return pd.DataFrame([fila])[FEATURE_ORDER]


def _calcular_age_range(age_years: float) -> int:
    """
    Categoría ordinal de edad (misma lógica que 03_feature_engineering.ipynb).
    1 = menor de 40 · 2 = 40-49 · 3 = 50-59 · 4 = 60 o más
    """
    if age_years < 40:
        return 1
    elif age_years < 50:
        return 2
    elif age_years < 60:
        return 3
    return 4


def _calcular_bp_category(ap_hi: int, ap_lo: int) -> int:
    """
    Clasificación AHA de presión arterial (misma lógica que 03_feature_engineering.ipynb).
    1 = Normal · 2 = Elevada · 3 = HTA grado 1 · 4 = HTA grado 2
    """
    if ap_hi < 120 and ap_lo < 80:
        return 1
    elif ap_hi < 130 and ap_lo < 80:
        return 2
    elif ap_hi < 140 or ap_lo < 90:
        return 3
    return 4


def _calcular_metabolic_risk(colesterol: int, gluc: int, bmi: float) -> int:
    """
    Score de riesgo metabólico sumado de 0 a 3
    (misma lógica que 03_feature_engineering.ipynb).
    """
    return (
        int(colesterol > 1) +
        int(gluc > 1) +
        int(bmi >= 30)
    )
