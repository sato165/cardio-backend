"""
json_extractor.py
Mapea los campos de una historia clínica en JSON a las variables
del modelo final (notebook_proy_final) — k=4 clusters.
Versión mejorada: búsqueda recursiva, case‑insensitive, FHIR, arrays de condiciones.

NOTA: La talla se extrae en metros o centímetros y se normaliza a CENTÍMETROS,
tal como lo espera el schema de entrada (input_schema.py) que luego convierte a metros.
"""

from typing import Any, Optional, Union
from datetime import date
import json


# ─── Campos obligatorios — modelo final (19 campos) ──────────────────────────
CAMPOS_OBLIGATORIOS = {
    "c_total":         "Colesterol total (mg/dL)",
    "creatinina":      "Creatinina sérica (mg/dL)",
    "glucosa":         "Glucosa en ayunas (mg/dL)",
    "hdl":             "Colesterol HDL (mg/dL)",
    "hemoglobina":     "Hemoglobina (g/dL)",
    "ldl":             "Colesterol LDL (mg/dL)",
    "leucocitos":      "Leucocitos (10³/µL)",
    "plaquetas":       "Plaquetas (10³/µL)",
    "trigliceridos":   "Triglicéridos (mg/dL)",
    "edad":            "Edad (años)",
    "sexo":            "Sexo (0=mujer, 1=hombre)",
    "zona":            "Zona (0=rural, 1=urbana)",
    "ap_hipertension": "Antecedente personal de HTA (0/1)",
    "ta_sistolica":    "Presión sistólica (mmHg)",
    "ta_diastolica":   "Presión diastólica (mmHg)",
    "peso":            "Peso (kg)",
    "talla":           "Talla (cm)",               # ← se guarda en cm
    "imc":             "IMC (kg/m²)",
    "TFG":             "TFG (mL/min/1.73m²)",
}

CAMPOS_OPCIONALES = {
    "colesterol_total_mgdl":      "Colesterol total (mg/dL) – Framingham",
    "diabetes":                   "Diabetes mellitus (0/1) – Framingham",
    "tratamiento_antihipertensivo": "Tratamiento antihipertensivo (0/1) – Framingham",
    "fuma":                       "Fumador actual (0/1) – Framingham",
}

# ─── Alias expandidos (case‑insensitive) ────────────────────────────────────
_ALIAS: dict[str, list[str]] = {
    "c_total": [
        "c_total", "colesterol_total", "colesterol", "total_cholesterol",
        "colesterol_total_mgdl", "ct"
    ],
    "creatinina": [
        "creatinina", "creatinina_serica", "creatinina_mgdl", "serum_creatinine",
        "creatinina_mg/dl", "creatinina_mg_dl"
    ],
    "glucosa": [
        "glucosa", "glucosa_ayunas", "glucose", "glucosa_mgdl", "gluc", "glicemia",
        "glucosa_mg/dl", "glucosa_mg_dl"
    ],
    "hdl": [
        "hdl", "hdl_colesterol", "colesterol_hdl", "hdl_mgdl", "hdl_cholesterol",
        "hdl_mg/dl", "hdl_mg_dl"
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
        "talla", "talla_m", "talla_cm", "height", "altura", "estatura", "talla_metros",
        "height_m", "talla_mt", "talla_m"
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


# ─── Normalización mejorada ──────────────────────────────────────────────────
def _normalizar(campo: str, valor: Any) -> Any:
    """Convierte el valor al tipo correcto según el campo."""
    enteros = {"edad", "sexo", "zona", "ap_hipertension", "diabetes",
               "tratamiento_antihipertensivo", "fuma"}

    if campo in enteros:
        if isinstance(valor, bool):
            return 1 if valor else 0
        if isinstance(valor, str):
            v = valor.strip().lower()
            if v in {"true", "sí", "si", "yes", "1", "activo", "presente", "present",
                     "m", "male", "masculino", "hombre", "varon",
                     "urbano", "urban"}:
                return 1
            elif v in {"false", "no", "0", "inactivo", "ausente", "absent", "negativo",
                       "f", "female", "femenino", "mujer", "rural"}:
                return 0
        try:
            return int(valor) if valor is not None else None
        except (ValueError, TypeError):
            return None

    # Campos decimales (todos los demás)
    try:
        return float(valor) if valor is not None else None
    except (ValueError, TypeError):
        return None


def _normalizar_talla(valor: Any) -> Optional[float]:
    """
    Convierte cualquier entrada de talla a centímetros (cm) según lo espera input_schema.
    - Si valor <= 3.0 se asume metros → se multiplica por 100.
    - Si valor > 3.0 se asume ya en centímetros.
    - Se redondea a 1 decimal.
    - Valores fuera del rango 50-250 cm se descartan (None).
    """
    try:
        val = float(valor)
        if val <= 3.0:          # metros → centímetros
            val = val * 100.0
        # Validación básica: rangos físicamente posibles
        if val < 50 or val > 250:
            return None
        return round(val, 1)
    except (ValueError, TypeError):
        return None


# ─── FHIR Bundle -> dict plano ──────────────────────────────────────────────
def _flatten_fhir_bundle(bundle: dict) -> dict:
    """
    Convierte un FHIR Bundle en un diccionario plano con las claves esperadas.
    Solo procesa recursos comunes: Patient, Condition, Observation,
    MedicationStatement, DiagnosticReport.
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
            if any(drug in nombre for drug in ["losartán", "valsartán", "enalapril", "amlodipino"]):
                flat["tratamiento_antihipertensivo"] = 1

    # Observations (signos vitales)
    for obs in observations:
        comps = obs.get("components", [])
        for c in comps:
            code = c.get("code", {})
            if code == "8480-6":
                flat["ta_sistolica"] = c.get("value")
            elif code == "8462-4":
                flat["ta_diastolica"] = c.get("value")
            elif code == "29463-7":
                flat["peso"] = c.get("value")
            elif code == "8302-2":
                flat["talla"] = c.get("value")
            elif code == "39156-5":
                flat["imc"] = c.get("value")

    # DiagnosticReport (laboratorio)
    for lab in labs:
        result = lab.get("result", {})
        hemograma = result.get("hemograma", {})
        if hemograma:
            flat["hemoglobina"] = hemograma.get("hemoglobina", {}).get("value")
            flat["leucocitos"] = hemograma.get("leucocitos", {}).get("value")
            flat["plaquetas"] = hemograma.get("plaquetas", {}).get("value")
        quimica = result.get("quimica", {})
        if quimica:
            flat["glucosa"] = quimica.get("glucosa", {}).get("value")
            flat["creatinina"] = quimica.get("creatinina", {}).get("value")
            flat["TFG"] = quimica.get("tfg_ckd_epi", {}).get("value") or quimica.get("tfg", {}).get("value")
            flat["c_total"] = quimica.get("colesterol_total", {}).get("value")
            flat["hdl"] = quimica.get("hdl", {}).get("value")
            flat["ldl"] = quimica.get("ldl", {}).get("value")
            flat["trigliceridos"] = quimica.get("trigliceridos", {}).get("value")
            flat["colesterol_total_mgdl"] = flat["c_total"]

    return flat


# ─── Búsqueda recursiva universal ───────────────────────────────────────────
def _buscar_en_json(nodo: Any, alias_list: list[str]) -> Any:
    """
    Recorre recursivamente dicts y listas, buscando el primer valor para cualquier alias.
    Comparación de claves case‑insensitive.
    """
    if isinstance(nodo, dict):
        lowered = {k.lower(): k for k in nodo.keys()}
        for alias in alias_list:
            if alias in lowered:
                val = nodo[lowered[alias]]
                if val is not None:
                    return val
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


# ─── Extracción desde arrays de condiciones ──────────────────────────────────
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
    return flags


# ─── Derivar edad desde fecha de nacimiento ──────────────────────────────────
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
        if 0 <= edad <= 120:
            return edad
    except (ValueError, TypeError):
        pass
    return None


# ─── Función principal de extracción ────────────────────────────────────────
def extraer_de_json(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Recibe el JSON de historia clínica y devuelve un dict con las
    variables del modelo final (19 obligatorias + 4 opcionales).
    Soporta estructuras anidadas, FHIR Bundle, arrays de condiciones,
    y cálculo automático de edad e IMC.

    NOTA: La talla se normaliza a CENTÍMETROS (cm) para cumplir con el esquema de entrada.
    """
    # Si es FHIR Bundle, transformar primero
    if payload.get("resourceType") == "Bundle":
        payload = _flatten_fhir_bundle(payload)

    # Extraer condiciones desde arrays (antes de búsqueda normal)
    flags_cond = _extraer_de_arrays_condiciones(payload)

    # Inicializar todos los campos a None
    todos = list(CAMPOS_OBLIGATORIOS.keys()) + list(CAMPOS_OPCIONALES.keys())
    campos = {campo: None for campo in todos}

    # Búsqueda principal con alias
    for campo, alias_list in _ALIAS.items():
        if campo in flags_cond:
            campos[campo] = flags_cond[campo]
        else:
            valor = _buscar_en_json(payload, alias_list)
            if valor is not None:
                # Tratamiento especial para talla: convertir a centímetros
                if campo == "talla":
                    valor = _normalizar_talla(valor)
                campos[campo] = _normalizar(campo, valor)

    # Derivar edad desde fecha de nacimiento si no se encontró
    if campos.get("edad") is None:
        campos["edad"] = _derivar_edad_desde_fecha(payload)

    # Calcular IMC si falta pero hay peso y talla (talla en cm, convertir a metros)
    if campos.get("imc") is None:
        peso = campos.get("peso")
        talla_cm = campos.get("talla")
        if peso is not None and talla_cm is not None and talla_cm > 0:
            talla_m = talla_cm / 100.0
            campos["imc"] = round(peso / (talla_m ** 2), 1)

    # Listar campos obligatorios faltantes
    campos["campos_faltantes"] = [
        {"campo": k, "descripcion": CAMPOS_OBLIGATORIOS[k]}
        for k in CAMPOS_OBLIGATORIOS
        if campos.get(k) is None
    ]

    return campos