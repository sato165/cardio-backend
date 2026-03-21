"""
prediction_service.py
Orquesta el flujo completo: preprocesamiento → predicción → explicabilidad.
Es la única capa que conoce todos los módulos de ml/ y services/.
Las rutas (routes/) solo llaman a este servicio — nunca importan ml/ directamente.
"""

from pydantic import ValidationError

from app.ml.preprocessing import preparar_features
from app.ml.predictor     import predecir
from app.ml.explainer     import explicar_shap
from app.schemas.input_schema  import CardiovascularInput
from app.schemas.output_schema import PredictionOutput, FactorExplicacion, UploadOutput, CampoFaltante


def predecir_desde_formulario(datos: CardiovascularInput) -> PredictionOutput:
    """
    Flujo del endpoint POST /api/predict.
    Recibe los datos ya validados por Pydantic y retorna la predicción completa.
    """
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
    Recibe el dict extraído por json_extractor o pdf_extractor,
    verifica campos faltantes y retorna predicción o lista de pendientes.
    """
    faltantes = campos.get("campos_faltantes", [])

    if faltantes:
        nombres = [f["campo"] for f in faltantes]
        return UploadOutput(
            campos_faltantes=[CampoFaltante(**f) for f in faltantes],
            prediccion=None,
            mensaje=(
                f"Faltan {len(faltantes)} campo(s) para completar la predicción: "
                f"{', '.join(nombres)}. Por favor ingréselos manualmente."
            ),
        )

    # Todos los campos presentes — validar rangos antes de predecir
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
        # Los valores existen pero están fuera de rango clínico aceptado.
        # Convertir a lista de campos con descripción legible para el médico.
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
            mensaje=(
                f"{len(campos_invalidos)} campo(s) tienen valores fuera del rango aceptado. "
                f"Por favor corrija los valores e intente nuevamente."
            ),
        )

    prediccion = predecir_desde_formulario(input_datos)

    return UploadOutput(
        campos_faltantes=[],
        prediccion=prediccion,
        mensaje="Predicción completada exitosamente.",
    )
