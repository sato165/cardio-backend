"""
json_extractor.py
Extrae los 22 campos del modelo real desde una historia clínica en JSON.

Busca por múltiples alias y en secciones anidadas típicas de HIS.
Los campos opcionales de Framingham se extraen si están presentes.
"""

from typing import Any, Optional
from datetime import date

# ---------------------------------------------------------------------------
# 22 campos obligatorios del nuevo modelo (dataset real Colombia)
# ---------------------------------------------------------------------------
CAMPOS_OBLIGATORIOS = {
    "creatinina":          "Creatinina sérica (mg/dL)",
    "celulas_medias":      "Volumen corpuscular medio (fL)",
    "glucosa":             "Glucosa en ayunas (mg/dL)",
    "granulocitos":        "Granulocitos (%)",
    "hdl":                 "Colesterol HDL (mg/dL)",
    "hematocrito":         "Hematocrito (%)",
    "hemoglobina":         "Hemoglobina (g/dL)",
    "ldl":                 "Colesterol LDL (mg/dL)",
    "leucocitos":          "Leucocitos (10³/µL)",
    "linfocitos":          "Linfocitos (%)",
    "plaquetas":           "Plaquetas (10³/µL)",
    "trigliceridos":       "Triglicéridos (mg/dL)",
    "edad":                "Edad (años)",
    "sexo":                "Sexo (0=mujer, 1=hombre)",
    "zona":                "Zona (0=rural, 1=urbana)",
    "ap_hipertension":     "Antecedente personal de hipertensión (0/1)",
    "ta_sistolica":        "Presión sistólica (mmHg)",
    "ta_diastolica":       "Presión diastólica (mmHg)",
    "peso":                "Peso (kg)",
    "talla":               "Talla (m)",
    "imc":                 "Índice de Masa Corporal (kg/m²)",
    "TFG":                 "Tasa de Filtración Glomerular (mL/min/1.73m²)",
}

# 4 campos opcionales para Framingham / SCC
CAMPOS_OPCIONALES = {
    "colesterol_total_mgdl":      "Colesterol total (mg/dL)",
    "diabetes":                   "Diabetes mellitus (0/1)",
    "tratamiento_antihipertensivo": "Tratamiento antihipertensivo (0/1)",
    "fuma":                       "Fumador actual (0/1)",
}

# Aliases para búsqueda flexible
_ALIAS: dict[str, list[str]] = {
    # Laboratorio
    "creatinina":          ["creatinina", "creatinina_serica", "creatinina_mgdl", "serum_creatinine"],
    "celulas_medias":      ["celulas_medias", "volumen_corpuscular_medio", "vcm", "mcv"],
    "glucosa":             ["glucosa", "glucosa_ayunas", "glucose", "glucosa_mgdl", "gluc", "glicemia"],
    "granulocitos":        ["granulocitos", "granulocitos_porcentaje", "granulocytes_pct"],
    "hdl":                 ["hdl", "hdl_colesterol", "colesterol_hdl", "hdl_mgdl", "hdl_cholesterol"],
    "hematocrito":         ["hematocrito", "hematocrito_pct", "hematocrit", "hct"],
    "hemoglobina":         ["hemoglobina", "hemoglobina_gdl", "hemoglobin", "hb"],
    "ldl":                 ["ldl", "ldl_colesterol", "colesterol_ldl", "ldl_mgdl", "ldl_cholesterol"],
    "leucocitos":          ["leucocitos", "leucocitos_103ul", "leucocytes", "wbc", "white_blood_cells"],
    "linfocitos":          ["linfocitos", "linfocitos_porcentaje", "lymphocytes_pct"],
    "plaquetas":           ["plaquetas", "plaquetas_103ul", "platelets", "plt"],
    "trigliceridos":       ["trigliceridos", "trigliceridos_mgdl", "triglycerides", "tg"],
    # Demográficos
    "edad":                ["edad", "edad_anos", "age", "edad_paciente"],
    "sexo":                ["sexo", "sexo_codigo", "gender", "genero_codigo"],
    "zona":                ["zona", "zona_residencia", "area", "rural_urbano"],
    "ap_hipertension":     ["ap_hipertension", "antecedente_hipertension", "hipertension_arterial", "hta_personal"],
    # Signos vitales y antropometría
    "ta_sistolica":        ["ta_sistolica", "presion_sistolica", "sistolica", "systolic", "ap_hi", "pas"],
    "ta_diastolica":       ["ta_diastolica", "presion_diastolica", "diastolica", "diastolic", "ap_lo", "pad"],
    "peso":                ["peso", "peso_kg", "weight", "weight_kg"],
    "talla":               ["talla", "talla_m", "height", "altura", "estatura"],
    "imc":                 ["imc", "bmi", "indice_masa_corporal", "body_mass_index"],
    "TFG":                 ["tfg", "tfge", "filtracion_glomerular", "egfr", "gfr"],
    # Opcionales Framingham
    "colesterol_total_mgdl": ["colesterol_total_mgdl", "colesterol_total", "ct", "total_cholesterol"],
    "diabetes":             ["diabetes", "diabetico", "dm", "diabetes_mellitus"],
    "tratamiento_antihipertensivo": [
        "tratamiento_antihipertensivo", "antihipertensivos", "toma_antihipertensivos", "hta_tratada"
    ],
    "fuma":                ["fuma", "fumador", "smoke", "smoking", "tabaquismo", "cigarrillo"],
}

_RUTAS_ANIDADAS = [
    "campos_modelo_ia",
    "identificacion_paciente",
    "signos_vitales",
    "datos_antropometricos",
    "examenes_laboratorio",
    "laboratorio",
    "quimica_sanguinea",
    "hemograma",
    "perfil_lipidico",
    "funcion_renal",
    "habitos_vida",
    "antecedentes",
    "medicacion_actual",
]


def extraer_de_json(datos: dict) -> dict[str, Any]:
    """
    Busca los 22 campos obligatorios + 4 opcionales en el JSON.
    Retorna dict con campos encontrados + lista de faltantes.
    """
    todos = list(CAMPOS_OBLIGATORIOS.keys()) + list(CAMPOS_OPCIONALES.keys())
    campos = {campo: None for campo in todos}

    fuentes = [datos] + [
        datos[s] for s in _RUTAS_ANIDADAS if isinstance(datos.get(s), dict)
    ]

    for campo, alias_list in _ALIAS.items():
        for fuente in fuentes:
            valor = _buscar_alias(fuente, alias_list)
            if valor is not None:
                campos[campo] = _normalizar(campo, valor)
                break

    # Derivar edad desde fecha de nacimiento si no se encontró
    if campos["edad"] is None:
        campos["edad"] = _derivar_edad_desde_fecha(datos)

    # Intentar calcular IMC si falta
    if campos["imc"] is None and campos["peso"] and campos["talla"]:
        try:
            campos["imc"] = round(campos["peso"] / (campos["talla"] ** 2), 1)
        except (TypeError, ZeroDivisionError):
            pass

    campos["campos_faltantes"] = [
        {"campo": k, "descripcion": CAMPOS_OBLIGATORIOS[k]}
        for k in CAMPOS_OBLIGATORIOS
        if campos.get(k) is None
    ]
    return campos


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _buscar_alias(fuente: dict, alias_list: list[str]) -> Any:
    for alias in alias_list:
        if alias in fuente and fuente[alias] is not None:
            return fuente[alias]
    return None


def _normalizar(campo: str, valor: Any) -> Any:
    """Convierte el valor al tipo esperado por el modelo."""
    enteros = {"edad", "sexo", "zona", "ap_hipertension", "diabetes", "tratamiento_antihipertensivo", "fuma"}
    floats  = set(CAMPOS_OBLIGATORIOS.keys()) - enteros.union({"edad", "sexo", "zona", "ap_hipertension"})

    if campo in enteros:
        if isinstance(valor, bool):
            return 1 if valor else 0
        if isinstance(valor, str):
            return 1 if valor.lower() in {"true", "sí", "si", "yes", "1", "activo"} else 0
        return int(valor) if valor is not None else None

    if campo in floats:
        return float(valor) if valor is not None else None

    return valor


def _derivar_edad_desde_fecha(datos: dict) -> Optional[int]:
    """Calcula edad en años desde fecha de nacimiento ISO."""
    claves = ["fecha_nacimiento", "birth_date", "date_of_birth", "dob"]
    fuentes = [datos] + [
        datos[s] for s in _RUTAS_ANIDADAS if isinstance(datos.get(s), dict)
    ]
    for fuente in fuentes:
        for clave in claves:
            if clave in fuente and fuente[clave]:
                try:
                    nac = date.fromisoformat(str(fuente[clave]))
                    edad = date.today().year - nac.year
                    if (date.today().month, date.today().day) < (nac.month, nac.day):
                        edad -= 1
                    if 18 <= edad <= 110:
                        return edad
                except (ValueError, TypeError):
                    continue
    return None