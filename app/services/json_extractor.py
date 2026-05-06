"""
json_extractor.py
Extractor mejorado de los 22 campos del modelo real desde una historia clínica en JSON.

- Búsqueda recursiva completa (diccionarios y arrays de objetos).
- Comparación de alias case‑insensitive.
- Lista amplia de alias para cubrir variantes comunes.
- Normalización avanzada de valores textuales.
- Detección de arrays de condiciones (ej. historial_patologico).
- Soporte para FHIR Bundle (transformación inicial a plano).
"""

from typing import Any, Optional, Union
from datetime import date
import json

# ---------------------------------------------------------------------------
# Campos obligatorios y opcionales (sin cambios)
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

CAMPOS_OPCIONALES = {
    "colesterol_total_mgdl":      "Colesterol total (mg/dL)",
    "diabetes":                   "Diabetes mellitus (0/1)",
    "tratamiento_antihipertensivo": "Tratamiento antihipertensivo (0/1)",
    "fuma":                       "Fumador actual (0/1)",
}

# ---------------------------------------------------------------------------
# Alias expandidos (ahora case‑insensitive)
# ---------------------------------------------------------------------------
_ALIAS: dict[str, list[str]] = {
    "creatinina": [
        "creatinina", "creatinina_serica", "creatinina_mgdl", "serum_creatinine",
        "creatinina_mg/dl", "creatinina_mg_dl"
    ],
    "celulas_medias": [
        "celulas_medias", "volumen_corpuscular_medio", "vcm", "mcv",
        "celulas_medias_pct", "celulas_medias_%", "mean_cell_volume"
    ],
    "glucosa": [
        "glucosa", "glucosa_ayunas", "glucose", "glucosa_mgdl", "gluc", "glicemia",
        "glucosa_mg/dl", "glucosa_mg_dl"
    ],
    "granulocitos": [
        "granulocitos", "granulocitos_porcentaje", "granulocytes_pct",
        "granulocitos_pct", "granulocitos_%", "granulocitos_percent"
    ],
    "hdl": [
        "hdl", "hdl_colesterol", "colesterol_hdl", "hdl_mgdl", "hdl_cholesterol",
        "hdl_mg/dl", "hdl_mg_dl"
    ],
    "hematocrito": [
        "hematocrito", "hematocrito_pct", "hematocrit", "hct",
        "hematocrito_%", "hematocrito_percent"
    ],
    "hemoglobina": [
        "hemoglobina", "hemoglobina_gdl", "hemoglobin", "hb",
        "hemoglobina_g/dl", "hemoglobina_g_dl"
    ],
    "ldl": [
        "ldl", "ldl_colesterol", "colesterol_ldl", "ldl_mgdl", "ldl_cholesterol",
        "ldl_mg/dl", "ldl_mg_dl"
    ],
    "leucocitos": [
        "leucocitos", "leucocitos_103ul", "leucocytes", "wbc", "white_blood_cells",
        "leucocitos_mil_ul", "leucocitos_x10³/µl", "leucocitos_10³/µl"
    ],
    "linfocitos": [
        "linfocitos", "linfocitos_porcentaje", "lymphocytes_pct",
        "linfocitos_pct", "linfocitos_%", "linfocitos_percent"
    ],
    "plaquetas": [
        "plaquetas", "plaquetas_103ul", "platelets", "plt",
        "plaquetas_mil_ul", "plaquetas_x10³/µl", "plaquetas_10³/µl"
    ],
    "trigliceridos": [
        "trigliceridos", "trigliceridos_mgdl", "triglycerides", "tg",
        "trigliceridos_mg/dl", "trigliceridos_mg_dl"
    ],
    "edad": [
        "edad", "edad_anos", "age", "edad_paciente", "edad_anios", "edad_años",
        "edad_anios", "edad_paciente_anos", "patient_age"
    ],
    "sexo": [
        "sexo", "sexo_codigo", "gender", "genero_codigo", "sexo_biologico",
        "sexo_biologico_cod", "sexo_paciente", "genero"
    ],
    "zona": [
        "zona", "zona_residencia", "area", "rural_urbano", "procedencia",
        "zona_vivienda", "urban_vs_rural", "tipo_zona"
    ],
    "ap_hipertension": [
        "ap_hipertension", "antecedente_hipertension", "hipertension_arterial",
        "hta_personal", "flag_hipertension", "htn_personal", "hipertension"
    ],
    "ta_sistolica": [
        "ta_sistolica", "presion_sistolica", "sistolica", "systolic", "ap_hi",
        "pas", "sistolica_mmhg", "ta_sistolica_mmhg"
    ],
    "ta_diastolica": [
        "ta_diastolica", "presion_diastolica", "diastolica", "diastolic", "ap_lo",
        "pad", "diastolica_mmhg", "ta_diastolica_mmhg"
    ],
    "peso": [
        "peso", "peso_kg", "weight", "weight_kg", "peso_kg", "peso_corporal_kg"
    ],
    "talla": [
        "talla", "talla_m", "height", "altura", "estatura", "talla_metros",
        "height_m", "talla_mt"
    ],
    "imc": [
        "imc", "bmi", "indice_masa_corporal", "body_mass_index",
        "imc_kg/m2", "bmi_value"
    ],
    "TFG": [
        "tfg", "tfge", "filtracion_glomerular", "egfr", "gfr",
        "tfg_ckd_epi", "tfge_ckd_epi", "tasa_filtracion_glomerular"
    ],
    "colesterol_total_mgdl": [
        "colesterol_total_mgdl", "colesterol_total", "ct", "total_cholesterol",
        "colesterol_total_mg/dl", "colesterol_total_mg_dl"
    ],
    "diabetes": [
        "diabetes", "diabetico", "dm", "diabetes_mellitus", "diabetes_mellitus_tipo2",
        "flag_diabetes", "dm2"
    ],
    "tratamiento_antihipertensivo": [
        "tratamiento_antihipertensivo", "antihipertensivos", "toma_antihipertensivos",
        "hta_tratada", "flag_antihipertensivo", "en_tratamiento_hta"
    ],
    "fuma": [
        "fuma", "fumador", "smoke", "smoking", "tabaquismo", "cigarrillo",
        "flag_fumador", "tabaco", "fumador_activo"
    ],
}

# ---------------------------------------------------------------------------
# Normalización mejorada
# ---------------------------------------------------------------------------
def _normalizar(campo: str, valor: Any) -> Any:
    """Convierte el valor al tipo esperado por el modelo."""
    enteros = {"edad", "sexo", "zona", "ap_hipertension", "diabetes",
               "tratamiento_antihipertensivo", "fuma"}
    floats  = set(CAMPOS_OBLIGATORIOS.keys()) - enteros.union(
        {"edad", "sexo", "zona", "ap_hipertension"}
    )

    if campo in enteros:
        if isinstance(valor, bool):
            return 1 if valor else 0
        if isinstance(valor, str):
            v = valor.strip().lower()
            # Cadenas que representan 1 (positivo/masculino/urbano/presente)
            if v in {"true", "sí", "si", "yes", "1", "activo", "presente", "present",
                     "m", "male", "masculino", "hombre", "varon",
                     "urbano", "urban"}:
                return 1
            # Cadenas que representan 0 (negativo/femenino/rural/ausente)
            elif v in {"false", "no", "0", "inactivo", "ausente", "absent", "negativo",
                       "f", "female", "femenino", "mujer",
                       "rural"}:
                return 0
        return int(valor) if valor is not None else None

    if campo in floats:
        return float(valor) if valor is not None else None

    return valor

# ---------------------------------------------------------------------------
# FHIR Bundle -> dict plano
# ---------------------------------------------------------------------------
def _flatten_fhir_bundle(bundle: dict) -> dict:
    """
    Convierte un FHIR Bundle en un diccionario plano con las claves esperadas.
    Solo procesa recursos comunes: Patient, Condition, Observation, MedicationStatement, DiagnosticReport.
    """
    flat: dict[str, Any] = {}
    entries = bundle.get("entry", [])
    patient = None
    conditions = []
    observations = []
    meds = []
    labs = []

    for e in entries:
        res = e.get("resource", {})
        rtype = res.get("resourceType")
        if rtype == "Patient":
            patient = res
        elif rtype == "Condition":
            conditions.append(res)
        elif rtype == "Observation":
            observations.append(res)
        elif rtype == "MedicationStatement":
            meds.append(res)
        elif rtype == "DiagnosticReport":
            labs.append(res)

    # Patient
    if patient:
        name_entries = patient.get("name", [])
        if name_entries:
            family = name_entries[0].get("family", "")
            given = " ".join(name_entries[0].get("given", []))
            flat["nombre"] = f"{given} {family}".strip()
        gender = patient.get("gender", "")
        flat["sexo"] = gender
        birth = patient.get("birthDate")
        if birth:
            flat["fecha_nacimiento"] = birth
        address = patient.get("address", [])
        if address:
            addr_type = address[0].get("type", "")
            flat["zona"] = addr_type
        age_comp = patient.get("age_computed")
        if age_comp is not None:
            flat["edad"] = age_comp

    # Conditions -> flags
    for cond in conditions:
        code = cond.get("code", {}).get("coding", [{}])[0].get("display", "").lower()
        if "hipertensi" in code:
            flat["ap_hipertension"] = 1
            # tratamiento inferido? mejor no asumir
        if "diabet" in code:
            flat["diabetes"] = 1
        if "tabaquismo" in code or "fumador" in code:
            flat["fuma"] = 1
        if cond.get("managedWith") == "farmacologico":
            flat["tratamiento_antihipertensivo"] = 1

    # MedicationStatement -> tratamiento_antihipertensivo
    for med in meds:
        meds_list = med.get("medicamentos", [])
        for m in meds_list:
            nombre = m.get("nombre", "").lower()
            if any(hta_drug in nombre for hta_drug in ["losartán", "valsartán", "enalapril", "amlodipino"]):
                flat["tratamiento_antihipertensivo"] = 1

    # Observations (vital signs)
    for obs in observations:
        comps = obs.get("components", [])
        for c in comps:
            code = c.get("code", {})
            if code == "8480-6": flat["ta_sistolica"] = c.get("value")
            elif code == "8462-4": flat["ta_diastolica"] = c.get("value")
            elif code == "29463-7": flat["peso"] = c.get("value")
            elif code == "8302-2": flat["talla"] = c.get("value")
            elif code == "39156-5": flat["imc"] = c.get("value")

    # DiagnosticReport (laboratorio)
    for lab in labs:
        result = lab.get("result", {})
        # Hemograma
        hemograma = result.get("hemograma", {})
        if hemograma:
            flat["hemoglobina"] = hemograma.get("hemoglobina", {}).get("value")
            flat["hematocrito"] = hemograma.get("hematocrito", {}).get("value")
            flat["leucocitos"] = hemograma.get("leucocitos", {}).get("value")
            flat["plaquetas"] = hemograma.get("plaquetas", {}).get("value")
            flat["granulocitos"] = hemograma.get("granulocitos", {}).get("value")
            flat["linfocitos"] = hemograma.get("linfocitos", {}).get("value")
            flat["celulas_medias"] = hemograma.get("celulas_medias", {}).get("value")
        # Química
        quimica = result.get("quimica", {})
        if quimica:
            flat["glucosa"] = quimica.get("glucosa", {}).get("value")
            flat["creatinina"] = quimica.get("creatinina", {}).get("value")
            flat["TFG"] = quimica.get("tfg_ckd_epi", {}).get("value") or quimica.get("tfg", {}).get("value")
            flat["colesterol_total_mgdl"] = quimica.get("colesterol_total", {}).get("value")
            flat["hdl"] = quimica.get("hdl", {}).get("value")
            flat["ldl"] = quimica.get("ldl", {}).get("value")
            flat["trigliceridos"] = quimica.get("trigliceridos", {}).get("value")
    return flat

# ---------------------------------------------------------------------------
# Búsqueda recursiva universal (+ arrays + case‑insensitive)
# ---------------------------------------------------------------------------
def _buscar_en_json(nodo: Any, alias_list: list[str]) -> Any:
    """
    Recorre recursivamente dicts y listas, buscando el primer valor para cualquier alias.
    Comparación de claves case‑insensitive.
    """
    if isinstance(nodo, dict):
        # Crear mapa de claves en minúsculas para búsqueda rápida
        lowered = {k.lower(): k for k in nodo.keys()}
        for alias in alias_list:
            if alias in lowered:
                val = nodo[lowered[alias]]
                if val is not None:
                    return val
        # Recursión en valores
        for v in nodo.values():
            encontrado = _buscar_en_json(v, alias_list)
            if encontrado is not None:
                return encontrado
    elif isinstance(nodo, list):
        for item in nodo:
            encontrado = _buscar_en_json(item, alias_list)
            if encontrado is not None:
                return encontrado
    return None

# ---------------------------------------------------------------------------
# Extracción de condiciones desde arrays de objetos (ej. historial_patologico)
# ---------------------------------------------------------------------------
def _extraer_de_arrays_condiciones(datos: dict) -> dict[str, int]:
    """
    Detecta arrays como 'historial_patologico', 'antecedentes_patologicos', etc.
    que contengan objetos con {condicion: ..., presente: true/false}.
    Retorna flags encontrados.
    """
    flags = {}
    posibles_claves = ["historial_patologico", "antecedentes_patologicos",
                       "condiciones_cronicas", "antecedentes_medicos", "problemas"]
    for clave in posibles_claves:
        if clave in datos and isinstance(datos[clave], list):
            for item in datos[clave]:
                if not isinstance(item, dict):
                    continue
                cond = item.get("condicion", item.get("nombre", "")).lower()
                presente = item.get("presente", item.get("activo", False))
                if isinstance(presente, str):
                    presente = presente.lower() in {"true", "sí", "si", "1", "activo", "presente"}
                if "hipertens" in cond:
                    flags["ap_hipertension"] = 1 if presente else 0
                if "diabet" in cond:
                    flags["diabetes"] = 1 if presente else 0
                if "tabaquismo" in cond or "fumador" in cond:
                    flags["fuma"] = 1 if presente else 0
                # tratamiento antihipertensivo se puede inferir si hipertensión+tratamiento?
                # De momento no lo inferimos de esta forma.
    return flags

# ---------------------------------------------------------------------------
# Función principal de extracción
# ---------------------------------------------------------------------------
def extraer_de_json(datos: dict) -> dict[str, Any]:
    """
    Extrae los 22 campos obligatorios + 4 opcionales de cualquier estructura JSON.
    """
    # Si es FHIR Bundle, transformar primero
    if datos.get("resourceType") == "Bundle":
        datos = _flatten_fhir_bundle(datos)

    # Extraer condiciones desde arrays (antes de la búsqueda normal)
    flags_cond = _extraer_de_arrays_condiciones(datos)

    todos = list(CAMPOS_OBLIGATORIOS.keys()) + list(CAMPOS_OPCIONALES.keys())
    campos = {campo: None for campo in todos}

    # Búsqueda principal
    for campo, alias_list in _ALIAS.items():
        if campo in flags_cond:
            campos[campo] = flags_cond[campo]
        else:
            valor = _buscar_en_json(datos, alias_list)
            if valor is not None:
                campos[campo] = _normalizar(campo, valor)

    # Derivar edad desde fecha de nacimiento si no se encontró
    if campos["edad"] is None:
        campos["edad"] = _derivar_edad_desde_fecha(datos)

    # Calcular IMC si falta pero hay peso y talla
    if campos["imc"] is None and campos["peso"] is not None and campos["talla"] is not None:
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


def _derivar_edad_desde_fecha(datos: dict) -> Optional[int]:
    """Busca recursivamente fecha de nacimiento y calcula edad."""
    claves = ["fecha_nacimiento", "birth_date", "date_of_birth", "dob", "fecha_nac", "birthdate"]

    def buscar_fecha(nodo: Any) -> Optional[str]:
        if isinstance(nodo, dict):
            for c in claves:
                if c in nodo and nodo[c]:
                    return str(nodo[c])
            for v in nodo.values():
                res = buscar_fecha(v)
                if res:
                    return res
        elif isinstance(nodo, list):
            for item in nodo:
                res = buscar_fecha(item)
                if res:
                    return res
        return None

    fecha_str = buscar_fecha(datos)
    if not fecha_str:
        return None
    try:
        nac = date.fromisoformat(fecha_str)
        hoy = date.today()
        edad = hoy.year - nac.year
        if (hoy.month, hoy.day) < (nac.month, nac.day):
            edad -= 1
        if 18 <= edad <= 110:
            return edad
    except (ValueError, TypeError):
        pass
    return None