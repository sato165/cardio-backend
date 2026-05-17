from pydantic import ValidationError
from app.ml.preprocessing import preparar_features
from app.ml.predictor     import predecir
from app.schemas.input_schema  import PredictionInput
from app.schemas.output_schema import (
    PredictionOutput, ClusterProb,
    UploadOutput, CampoFaltante, DatosPaciente,
    RiesgoComparativo,
)
from app.ml.framingham_calculator import calcular_framingham, campos_faltantes_framingham
from app.ml.scc_calculator import calcular_scc


# Campos obligatorios alineados con columnas_modelo.pkl — modelo final k=4
CAMPOS_OBLIGATORIOS = [
    "c_total", "creatinina", "glucosa", "hdl", "hemoglobina",
    "ldl", "leucocitos", "plaquetas", "trigliceridos",
    "edad", "sexo", "zona", "ap_hipertension",
    "ta_sistolica", "ta_diastolica", "peso", "talla", "imc", "TFG"
]

# Rangos de validación — talla en cm (como llega del frontend)
RANGOS_VALIDACION = {
    "c_total":        (0,      550.0),
    "creatinina":     (0,      2.0),
    "glucosa":        (25.0,   492.0),
    "hdl":            (0,      120.0),
    "hemoglobina":    (7.0,    21.0),
    "ldl":            (0,      404.6),
    "leucocitos":     (0,      None),
    "plaquetas":      (0,      None),
    "trigliceridos":  (0,      420.0),
    "edad":           (6,      110),
    "ta_sistolica":   (60.5,   220.0),
    "ta_diastolica":  (40.0,   120.0),
    "peso":           (9.0,    170.0),
    "talla":          (127.0,  197.0),   # cm
    "imc":            (4.51,   60.0),
    "TFG":            (11.47,  197.39),
}


# ─────────────────────────────────────────────
# Predicción desde formulario manual
# ─────────────────────────────────────────────

def predecir_desde_formulario(datos: PredictionInput) -> PredictionOutput:
    features  = preparar_features(datos)
    resultado = predecir(features)

    probabilidades     = [ClusterProb(**p) for p in resultado["probabilities"]]
    riesgo_comparativo = _calcular_comparativo(datos)

    return PredictionOutput(
        predicted_cluster=resultado["predicted_cluster"],
        cluster_name=resultado["cluster_name"],
        description=resultado["description"],
        probabilities=probabilidades,
        riesgo_comparativo=riesgo_comparativo,
    )


def _calcular_comparativo(datos: PredictionInput) -> RiesgoComparativo:
    datos_fram = {
        "colesterol_total_mgdl":        datos.colesterol_total_mgdl,
        "diabetes":                     datos.diabetes,
        "tratamiento_antihipertensivo": datos.tratamiento_antihipertensivo,
        "fuma":                         datos.fuma,
    }
    faltantes = [k for k, v in datos_fram.items() if v is None]
    if faltantes:
        return RiesgoComparativo(
            datos_suficientes=False,
            campos_faltantes_framingham=faltantes,
        )

    fram = calcular_framingham(
        edad=datos.edad,
        sexo=datos.sexo,
        colesterol_total=datos.colesterol_total_mgdl,
        hdl=datos.hdl,
        presion_sistolica=datos.ta_sistolica,
        tratamiento_antihipertensivo=bool(datos.tratamiento_antihipertensivo),
        fuma=bool(datos.fuma),
        diabetes=bool(datos.diabetes),
    )

    if not fram["aplicable"]:
        return RiesgoComparativo(datos_suficientes=False)

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


# ─────────────────────────────────────────────
# Predicción desde extracción de archivo
# ─────────────────────────────────────────────

def predecir_desde_extraccion(campos: dict) -> UploadOutput:

    # 1. Verificar campos obligatorios faltantes
    faltantes = []
    for campo in CAMPOS_OBLIGATORIOS:
        if campo not in campos or campos[campo] is None:
            faltantes.append({
                "campo": campo,
                "descripcion": f"Campo obligatorio '{campo}' faltante"
            })

    # 2. Construir DatosPaciente con las variables del modelo final
    datos_paciente = DatosPaciente(
        c_total=campos.get("c_total"),
        creatinina=campos.get("creatinina"),
        glucosa=campos.get("glucosa"),
        hdl=campos.get("hdl"),
        hemoglobina=campos.get("hemoglobina"),
        ldl=campos.get("ldl"),
        leucocitos=campos.get("leucocitos"),
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

    # 3. Validar con PredictionInput y capturar errores de rango
    try:
        input_datos = PredictionInput(
            c_total=campos["c_total"],
            creatinina=campos["creatinina"],
            glucosa=campos["glucosa"],
            hdl=campos["hdl"],
            hemoglobina=campos["hemoglobina"],
            ldl=campos["ldl"],
            leucocitos=campos["leucocitos"],
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
            if campo in RANGOS_VALIDACION:
                mini, maxi = RANGOS_VALIDACION[campo]
                if mini is not None and maxi is not None:
                    rango_str = f"{mini} – {maxi}"
                elif mini is not None:
                    rango_str = f"≥ {mini}"
                elif maxi is not None:
                    rango_str = f"≤ {maxi}"
                else:
                    rango_str = "sin límite definido"
                desc = f"Valor fuera de rango ({error.get('input')}). Rango aceptable: {rango_str}."
            else:
                desc = f"Valor inválido ({error.get('input')}): {error['msg']}"
            campos_invalidos.append(CampoFaltante(campo=campo, descripcion=desc))

        return UploadOutput(
            campos_faltantes=campos_invalidos,
            prediccion=None,
            datos_paciente=datos_paciente,
            mensaje="Algunos campos tienen valores inválidos.",
        )

    # 4. Ejecutar predicción completa
    prediccion = predecir_desde_formulario(input_datos)
    return UploadOutput(
        campos_faltantes=[],
        prediccion=prediccion,
        datos_paciente=datos_paciente,
        mensaje="Predicción completada exitosamente.",
    )