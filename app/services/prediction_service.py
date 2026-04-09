"""
prediction_service.py
Orquesta el flujo completo: preprocesamiento → predicción → explicabilidad.
"""

from pydantic import ValidationError

from app.ml.preprocessing import preparar_features
from app.ml.predictor     import predecir
from app.ml.explainer     import explicar_shap
from app.schemas.input_schema  import CardiovascularInput
from app.schemas.output_schema import (
    PredictionOutput, FactorExplicacion,
    UploadOutput, CampoFaltante, DatosPaciente,
)


def predecir_desde_formulario(datos: CardiovascularInput) -> PredictionOutput:
    """Flujo del endpoint POST /api/predict."""
    features       = preparar_features(datos)
    resultado      = predecir(features)
    explicabilidad = explicar_shap(features)

    return PredictionOutput(
        riesgo_cardiovascular=resultado["clase"],
        probabilidad=resultado["probabilidad"],
        nivel_riesgo=resultado["nivel_riesgo"],
        explicabilidad=[FactorExplicacion(**f) for f in explicabilidad],
    )


def predecir_desde_extraccion(campos: dict) -> UploadOutput:
    """
    Flujo compartido por los endpoints de JSON y PDF.
    Incluye datos_paciente en la respuesta para el resumen visual del frontend.
    """
    faltantes = campos.get("campos_faltantes", [])

    # Construir DatosPaciente con lo que se extrajo hasta ahora
    datos_paciente = DatosPaciente(
        age_days    =campos.get("age_days"),
        gender      =campos.get("gender"),
        height      =campos.get("height"),
        weight      =campos.get("weight"),
        ap_hi       =campos.get("ap_hi"),
        ap_lo       =campos.get("ap_lo"),
        cholesterol =campos.get("cholesterol"),
        gluc        =campos.get("gluc"),
        smoke       =campos.get("smoke"),
        alco        =campos.get("alco"),
        active      =campos.get("active"),
    )

    if faltantes:
        nombres = [f["campo"] for f in faltantes]
        return UploadOutput(
            campos_faltantes=[CampoFaltante(**f) for f in faltantes],
            prediccion=None,
            datos_paciente=datos_paciente,
            mensaje=(
                f"Faltan {len(faltantes)} campo(s) para completar la predicción: "
                f"{', '.join(nombres)}. Por favor ingréselos manualmente."
            ),
        )

    try:
        input_datos = CardiovascularInput(
            age_days    =campos["age_days"],
            gender      =campos["gender"],
            height      =campos["height"],
            weight      =campos["weight"],
            ap_hi       =campos["ap_hi"],
            ap_lo       =campos["ap_lo"],
            cholesterol =campos["cholesterol"],
            gluc        =campos["gluc"],
            smoke       =campos["smoke"],
            alco        =campos["alco"],
            active      =campos["active"],
        )
    except ValidationError as e:
        campos_invalidos = []
        for error in e.errors():
            campo = str(error["loc"][-1])
            campos_invalidos.append(CampoFaltante(
                campo=campo,
                descripcion=f"Valor inválido ({error.get('input')}): {error['msg']}",
            ))
        return UploadOutput(
            campos_faltantes=campos_invalidos,
            prediccion=None,
            datos_paciente=datos_paciente,
            mensaje=(
                f"{len(campos_invalidos)} campo(s) tienen valores fuera del rango aceptado. "
                f"Por favor corrija los valores e intente nuevamente."
            ),
        )

    prediccion = predecir_desde_formulario(input_datos)

    return UploadOutput(
        campos_faltantes=[],
        prediccion=prediccion,
        datos_paciente=datos_paciente,
        mensaje="Predicción completada exitosamente.",
    )
