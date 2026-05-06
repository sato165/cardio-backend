import pandas as pd
from app.schemas.input_schema import PredictionInput  # nuevo schema

# Orden exacto de las 22 columnas con el que se entrenó el modelo real
FEATURE_ORDER = [
    "creatinina", "celulas_medias", "glucosa", "granulocitos",
    "hdl", "hematocrito", "hemoglobina", "ldl", "leucocitos",
    "linfocitos", "plaquetas", "trigliceridos", "edad", "sexo",
    "zona", "ap_hipertension", "ta_sistolica", "ta_diastolica",
    "peso", "talla", "imc", "TFG"
]

# Límites de winsorización (clipping) definidos en el notebook real
LIMITES = {
    "creatinina": (0, 2.0),
    "celulas_medias": (0, 20.0),
    "glucosa": (25.0, 492.0),
    "hdl": (0, 120.0),
    "hematocrito": (14.0, 63.0),
    "hemoglobina": (7.0, 21.0),
    "ldl": (0, 404.6),
    "trigliceridos": (0, 420.0),
    "edad": (6, 110),
    "ta_sistolica": (60.5, 220.0),
    "ta_diastolica": (40.0, 120.0),
    "peso": (9.0, 170.0),
    "talla": (1.27, 1.97),
    "imc": (4.51, 60.0),
    "TFG": (11.47, 197.39),
    # variables sin límite documentado: se dejan sin clip
}


def _aplicar_clipping(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica winsorización por límites clínicos a las columnas definidas.
    Modifica el DataFrame in-place.
    """
    for col, (lim_inf, lim_sup) in LIMITES.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lim_inf, upper=lim_sup)
    return df


def preparar_features(datos: PredictionInput) -> pd.DataFrame:
    """
    Recibe los datos validados del nuevo schema de entrada y retorna un
    DataFrame con las 22 columnas en el orden exacto que espera el modelo.
    """
    # Construir diccionario con todas las claves
    fila = {
        "creatinina": datos.creatinina,
        "celulas_medias": datos.celulas_medias,
        "glucosa": datos.glucosa,
        "granulocitos": datos.granulocitos,
        "hdl": datos.hdl,
        "hematocrito": datos.hematocrito,
        "hemoglobina": datos.hemoglobina,
        "ldl": datos.ldl,
        "leucocitos": datos.leucocitos,
        "linfocitos": datos.linfocitos,
        "plaquetas": datos.plaquetas,
        "trigliceridos": datos.trigliceridos,
        "edad": datos.edad,
        "sexo": datos.sexo,
        "zona": datos.zona,
        "ap_hipertension": datos.ap_hipertension,
        "ta_sistolica": datos.ta_sistolica,
        "ta_diastolica": datos.ta_diastolica,
        "peso": datos.peso,
        "talla": datos.talla,
        "imc": datos.imc,
        "TFG": datos.TFG,
    }

    # Crear DataFrame y asegurar el orden correcto
    df = pd.DataFrame([fila])[FEATURE_ORDER]

    # Aplicar clipping
    df = _aplicar_clipping(df)

    return df