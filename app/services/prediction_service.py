from pydantic import ValidationError
from app.ml.preprocessing import preparar_features
from app.ml.predictor     import predecir
# from app.ml.explainer import explicar_shap   # ya no se usa, se puede eliminar después
from app.schemas.input_schema  import PredictionInput
from app.schemas.output_schema import (
    PredictionOutput, ClusterProb,
    UploadOutput, CampoFaltante, DatosPaciente,
    RiesgoComparativo,
)
from app.ml.framingham_calculator import calcular_framingham, campos_faltantes_framingham
from app.ml.scc_calculator import calcular_scc


def predecir_desde_formulario(datos: PredictionInput) -> PredictionOutput:
    """Flujo del endpoint POST /api/predict (nuevo modelo de clusters)."""
    features  = preparar_features(datos)
    resultado = predecir(features)  # dict con cluster predicho y probabilidades

    # Construir lista de probabilidades
    probabilidades = [ClusterProb(**p) for p in resultado["probabilities"]]

    # Intentar cálculo comparativo con Framingham/SCC si hay datos
    riesgo_comparativo = _calcular_comparativo(datos)

    return PredictionOutput(
        predicted_cluster=resultado["predicted_cluster"],
        cluster_name=resultado["cluster_name"],
        description=resultado["description"],
        probabilities=probabilidades,
        riesgo_comparativo=riesgo_comparativo,
    )


def _calcular_comparativo(datos: PredictionInput) -> RiesgoComparativo:
    """Intenta calcular Framingham y SCC con los campos opcionales presentes."""
    # Framingham requiere: colesterol_total, diabetes, tratamiento_antihipertensivo, fuma
    # Además usa edad, sexo, hdl, ta_sistolica (ya obligatorios)
    datos_fram = {
        "colesterol_total_mgdl": datos.colesterol_total_mgdl,
        "diabetes": datos.diabetes,
        "tratamiento_antihipertensivo": datos.tratamiento_antihipertensivo,
        "fuma": datos.fuma,
    }
    faltantes = [k for k, v in datos_fram.items() if v is None]
    if faltantes:
        return RiesgoComparativo(
            datos_suficientes=False,
            campos_faltantes_framingham=faltantes,
        )

    # Todos los datos opcionales presentes; calcular
    fram = calcular_framingham(
        edad=datos.edad,
        sexo=datos.sexo,  # 0 mujer, 1 hombre
        colesterol_total=datos.colesterol_total_mgdl,
        hdl=datos.hdl,
        presion_sistolica=datos.ta_sistolica,
        tratamiento_antihipertensivo=bool(datos.tratamiento_antihipertensivo),
        fuma=bool(datos.fuma),
        diabetes=bool(datos.diabetes),
    )

    if not fram["aplicable"]:
        return RiesgoComparativo(
            datos_suficientes=False,
            # podrías agregar un mensaje extra si quieres
        )

    scc = calcular_scc(
        edad=datos.edad,
        sexo=datos.sexo,
        colesterol_total=datos.colesterol_total_mgdl,
        hdl=datos.hdl,
        presion_sistolica=datos.ta_sistolica,
        tratamiento_antihipertensivo=bool(datos.tratamiento_antihipertensivo),
        fuma=bool(datos.fuma),
        diabetes=bool(datos.diabetes),
    )

    return RiesgoComparativo(
        framingham_porcentaje=fram["porcentaje"],
        framingham_nivel=fram["nivel"],
        scc_porcentaje=scc.get("porcentaje_scc"),
        scc_nivel=scc.get("nivel"),
        datos_suficientes=True,
    )


def predecir_desde_extraccion(campos: dict) -> UploadOutput:
    """
    Flujo para carga de JSON/PDF.
    Espera un dict con las 22 claves del nuevo modelo + opcionales.
    """
    faltantes = []
    # Comprobar los 22 campos obligatorios
    obligatorios = [
        "creatinina", "celulas_medias", "glucosa", "granulocitos",
        "hdl", "hematocrito", "hemoglobina", "ldl", "leucocitos",
        "linfocitos", "plaquetas", "trigliceridos", "edad",
        "sexo", "zona", "ap_hipertension", "ta_sistolica",
        "ta_diastolica", "peso", "talla", "imc", "TFG"
    ]
    for campo in obligatorios:
        if campo not in campos or campos[campo] is None:
            faltantes.append({
                "campo": campo,
                "descripcion": f"Campo obligatorio '{campo}' faltante"
            })

    # Construir DatosPaciente con lo que haya
    datos_paciente = DatosPaciente(
        creatinina=campos.get("creatinina"),
        celulas_medias=campos.get("celulas_medias"),
        glucosa=campos.get("glucosa"),
        granulocitos=campos.get("granulocitos"),
        hdl=campos.get("hdl"),
        hematocrito=campos.get("hematocrito"),
        hemoglobina=campos.get("hemoglobina"),
        ldl=campos.get("ldl"),
        leucocitos=campos.get("leucocitos"),
        linfocitos=campos.get("linfocitos"),
        plaquetas=campos.get("plaquetas"),
        trigliceridos=campos.get("trigliceridos"),
        edad=campos.get("edad"),
        sexo=campos.get("sexo"),
        zona=campos.get("zona"),
        ap_hipertension=campos.get("ap_hipertension"),
        ta_sistolica=campos.get("ta_sistolica"),
        ta_diastolica=campos.get("ta_diastolica"),
        peso=campos.get("peso"),
        talla=campos.get("talla"),
        imc=campos.get("imc"),
        TFG=campos.get("TFG"),
        colesterol_total_mgdl=campos.get("colesterol_total_mgdl"),
        diabetes=campos.get("diabetes"),
        tratamiento_antihipertensivo=campos.get("tratamiento_antihipertensivo"),
        fuma=campos.get("fuma"),
    )

    if faltantes:
        return UploadOutput(
            campos_faltantes=[CampoFaltante(**f) for f in faltantes],
            prediccion=None,
            datos_paciente=datos_paciente,
            mensaje=f"Faltan {len(faltantes)} campo(s) obligatorio(s).",
        )

    # Intentar construir PredictionInput
    try:
        input_datos = PredictionInput(
            creatinina=campos["creatinina"],
            celulas_medias=campos["celulas_medias"],
            glucosa=campos["glucosa"],
            granulocitos=campos["granulocitos"],
            hdl=campos["hdl"],
            hematocrito=campos["hematocrito"],
            hemoglobina=campos["hemoglobina"],
            ldl=campos["ldl"],
            leucocitos=campos["leucocitos"],
            linfocitos=campos["linfocitos"],
            plaquetas=campos["plaquetas"],
            trigliceridos=campos["trigliceridos"],
            edad=campos["edad"],
            sexo=campos["sexo"],
            zona=campos["zona"],
            ap_hipertension=campos["ap_hipertension"],
            ta_sistolica=campos["ta_sistolica"],
            ta_diastolica=campos["ta_diastolica"],
            peso=campos["peso"],
            talla=campos["talla"],
            imc=campos["imc"],
            TFG=campos["TFG"],
            colesterol_total_mgdl=campos.get("colesterol_total_mgdl"),
            diabetes=campos.get("diabetes"),
            tratamiento_antihipertensivo=campos.get("tratamiento_antihipertensivo"),
            fuma=campos.get("fuma"),
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
            mensaje="Algunos campos tienen valores inválidos.",
        )

    prediccion = predecir_desde_formulario(input_datos)
    return UploadOutput(
        campos_faltantes=[],
        prediccion=prediccion,
        datos_paciente=datos_paciente,
        mensaje="Predicción completada exitosamente.",
    )