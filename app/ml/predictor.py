import pandas as pd
from app.ml.model_loader import get_model


def predecir(features: pd.DataFrame) -> dict:
    """
    Recibe el DataFrame de features preparado por preprocessing.py
    y retorna la clase predicha, la probabilidad y el nivel de riesgo.
    """
    model = get_model()

    clase      = int(model.predict(features)[0])
    probabilidad = float(model.predict_proba(features)[0][1])
    nivel      = _nivel_de_riesgo(probabilidad)

    return {
        "clase":        clase,
        "probabilidad": round(probabilidad, 4),
        "nivel_riesgo": nivel,
    }


def _nivel_de_riesgo(probabilidad: float) -> str:
    """
    Convierte la probabilidad numérica en una etiqueta clínica.
    Umbrales definidos con Mayerlis para que sean interpretables por el médico.
    """
    if probabilidad >= 0.70:
        return "Alto"
    elif probabilidad >= 0.45:
        return "Moderado"
    return "Bajo"
