# prediction_service.py 

"""
prediction_service.py
Orquesta el flujo completo: preprocesamiento → predicción → explicabilidad
y añade los cálculos de Framingham 2008 y SCC Colombia si hay datos suficientes.
"""

from pydantic import ValidationError

from app.ml.preprocessing import preparar_features
from app.ml.predictor     import predecir
from app.ml.explainer     import explicar_shap
from app.schemas.input_schema  import CardiovascularInput
from app.schemas.output_schema import (
    PredictionOutput, FactorExplicacion,
    UploadOutput, CampoFaltante, DatosPaciente,
    RiesgoComparativo,
)
from app.ml.framingham_calculator import calcular_framingham, campos_faltantes_framingham
from app.ml.scc_calculator import calcular_scc


def predecir_desde_formulario(datos: CardiovascularInput) -> PredictionOutput:
    """Flujo del endpoint POST /api/predict."""
    features       = preparar_features(datos)
    resultado      = predecir(features)
    explicabilidad = explicar_shap(features)

    # Intenta construir el comparativo Framingham / SCC
    riesgo_comparativo = _calcular_comparativo(datos)

    return PredictionOutput(
        riesgo_cardiovascular=resultado["clase"],
        probabilidad=resultado["probabilidad"],
        nivel_riesgo=resultado["nivel_riesgo"],
        explicabilidad=[FactorExplicacion(**f) for f in explicabilidad],
        riesgo_comparativo=riesgo_comparativo,
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
        colesterol_total_mgdl=campos.get("colesterol_total_mgdl"),
        hdl_mgdl=campos.get("hdl_mgdl"),
        diabetes=campos.get("diabetes"),
        tratamiento_antihipertensivo=campos.get("tratamiento_antihipertensivo"),
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

    # Validar y construir objeto CardiovascularInput para la predicción
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
            # Opcionales (pueden ser None)
            colesterol_total_mgdl=campos.get("colesterol_total_mgdl"),
            hdl_mgdl=campos.get("hdl_mgdl"),
            diabetes=campos.get("diabetes"),
            tratamiento_antihipertensivo=campos.get("tratamiento_antihipertensivo"),
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

    # Determinar si hacen falta datos para Framingham
    framingham_faltante = None
    if (prediccion.riesgo_comparativo and
        not prediccion.riesgo_comparativo.datos_suficientes and
        prediccion.riesgo_comparativo.campos_faltantes_framingham):
        
        # Mapeo de nombres internos a nombres de parámetros que espera upload.py
        mapeo_campos = {
            "colesterol_total_mgdl": "colesterol_total_mgdl",
            "hdl_mgdl": "hdl_mgdl",
            "diabetes": "diabetes",
            "tratamiento_hta": "tratamiento_antihipertensivo",
            "tratamiento_antihipertensivo": "tratamiento_antihipertensivo",
        }
        descripciones = {
            "colesterol_total_mgdl": "Colesterol total (mg/dL)",
            "hdl_mgdl": "Colesterol HDL (mg/dL)",
            "diabetes": "Diabetes",
            "tratamiento_hta": "Tratamiento antihipertensivo",
            "tratamiento_antihipertensivo": "Tratamiento antihipertensivo",
        }
        
        framingham_faltante = [
            CampoFaltante(
                campo=mapeo_campos.get(nombre, nombre),
                descripcion=descripciones.get(nombre, nombre)
            )
            for nombre in prediccion.riesgo_comparativo.campos_faltantes_framingham
        ]
        
    return UploadOutput(
        campos_faltantes=[],
        prediccion=prediccion,
        datos_paciente=datos_paciente,
        mensaje="Predicción completada exitosamente.",
        framingham_faltante=framingham_faltante,
    )

def _calcular_comparativo(datos: CardiovascularInput) -> RiesgoComparativo:
    edad = round(datos.age_days / 365.25)

    # Verificar datos necesarios para Framingham
    datos_fram = {
        "colesterol_total_mgdl": datos.colesterol_total_mgdl,
        "hdl_mgdl": datos.hdl_mgdl,
        "diabetes": datos.diabetes,
        "tratamiento_hta": datos.tratamiento_antihipertensivo,
    }
    faltantes = campos_faltantes_framingham(datos_fram)
    if faltantes:
        return RiesgoComparativo(
            datos_suficientes=False,
            campos_faltantes_framingham=[f["campo"] for f in faltantes],
        )

    # Llamar a Framingham con tu nueva función
    fram = calcular_framingham(
        edad=edad,
        sexo=datos.gender,
        colesterol_total=datos.colesterol_total_mgdl,
        hdl=datos.hdl_mgdl,
        presion_sistolica=datos.ap_hi,
        tratamiento_antihipertensivo=bool(datos.tratamiento_antihipertensivo),
        fuma=bool(datos.smoke),
        diabetes=bool(datos.diabetes),
    )

    if not fram["aplicable"]:
        return RiesgoComparativo(
            datos_suficientes=False,
            mensaje_no_aplicable=fram["descripcion"],
        )

    scc = calcular_scc(
        edad=edad,
        sexo=datos.gender,
        colesterol_total=datos.colesterol_total_mgdl,
        hdl=datos.hdl_mgdl,
        presion_sistolica=datos.ap_hi,
        tratamiento_antihipertensivo=bool(datos.tratamiento_antihipertensivo),
        fuma=bool(datos.smoke),
        diabetes=bool(datos.diabetes),
    )

    return RiesgoComparativo(
        framingham_porcentaje=fram["porcentaje"],
        framingham_nivel=fram["nivel"],
        framingham_descripcion=fram["descripcion"],
        scc_porcentaje=scc["porcentaje_scc"],
        scc_porcentaje_framingham=scc["porcentaje_framingham"],
        scc_nivel=scc["nivel"],
        scc_descripcion=scc["descripcion"],
        factor_ajuste=scc["factor_ajuste"],
        datos_suficientes=True,
    )