import numpy as np
import pandas as pd
from app.schemas.input_schema import PredictionInput
from app.ml.model_loader import get_artifacts

# Límites de winsorización (clipping) definidos por expertos clínicos — notebook final
LIMITES = {
    "c_total":       (0,     550.0),
    "creatinina":    (0,     2.0),
    "glucosa":       (25.0,  492.0),
    "hdl":           (0,     120.0),
    "hemoglobina":   (7.0,   21.0),
    "ldl":           (0,     404.6),
    "trigliceridos": (0,     420.0),
    "edad":          (6,     110),
    "ta_sistolica":  (60.5,  220.0),
    "ta_diastolica": (40.0,  120.0),
    "peso":          (9.0,   170.0),
    "talla":         (1.27,  1.97),   # en METROS
    "imc":           (4.51,  60.0),
    "TFG":           (11.47, 197.39),
}


def _aplicar_clipping(df: pd.DataFrame) -> pd.DataFrame:
    """Winsorización por límites clínicos expertos."""
    for col, (lim_inf, lim_sup) in LIMITES.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lim_inf, upper=lim_sup)
    return df


def preparar_features(datos: PredictionInput) -> pd.DataFrame:
    """
    Recibe los datos validados del schema de entrada y retorna un DataFrame
    preprocesado listo para escalar y proyectar con PCA.

    Pipeline:
        1. Construir fila con todos los campos del input
        2. Convertir talla de cm (frontend) a metros
        3. Reordenar columnas según columnas_modelo.pkl
        4. Winsorización con límites clínicos
        5. Imputación KNN (por si llega algún nulo)
    """
    artifacts = get_artifacts()
    columnas  = artifacts["columnas"]   # lista cargada de columnas_modelo.pkl
    imputer   = artifacts["imputer"]    # KNNImputer entrenado

    # 1. Construir fila completa desde el input
    fila = {
        "c_total":        datos.c_total        if hasattr(datos, "c_total") else np.nan,
        "creatinina":     datos.creatinina,
        "glucosa":        datos.glucosa,
        "hdl":            datos.hdl,
        "hemoglobina":    datos.hemoglobina,
        "ldl":            datos.ldl,
        "leucocitos":     datos.leucocitos,
        "plaquetas":      datos.plaquetas,
        "trigliceridos":  datos.trigliceridos,
        "edad":           datos.edad,
        "sexo":           datos.sexo,
        "zona":           datos.zona,
        "ap_hipertension": datos.ap_hipertension,
        "ta_sistolica":   datos.ta_sistolica,
        "ta_diastolica":  datos.ta_diastolica,
        "peso":           datos.peso,
        "talla":          datos.talla,
        "imc":            datos.imc,
        "TFG":            datos.TFG,
    }

    df = pd.DataFrame([fila])

    # 2. Convertir talla de cm → metros (el frontend envía cm)
    if "talla" in df.columns:
        df["talla"] = df["talla"] / 100.0

    # 3. Reordenar columnas exactamente como en el entrenamiento
    #    — añadir con NaN las columnas que pudieran faltar en el input
    for col in columnas:
        if col not in df.columns:
            df[col] = np.nan
    df = df[columnas]

    # 4. Winsorización
    df = _aplicar_clipping(df)

    # 5. Imputación KNN (cubre cualquier nulo que llegue)
    cols_num = df.select_dtypes(include=[np.number]).columns.tolist()
    df[cols_num] = imputer.transform(df[cols_num])

    return df