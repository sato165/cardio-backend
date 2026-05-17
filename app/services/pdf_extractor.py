"""
pdf_extractor.py
Extrae las variables del modelo final (notebook_proy_final) desde PDFs
de historia clínica. Soporta texto nativo (PyMuPDF) y tablas (pdfplumber).
k=4 clusters — variables alineadas con columnas_modelo.pkl.
"""

import re
import fitz
import pdfplumber
import io
from typing import Any, Optional
from datetime import date

MIN_CHARS_TEXTO = 50
SEP = r"[\s|]+"

# ─── Campos obligatorios — modelo final ──────────────────────────────────────
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
    "talla":           "Talla (cm)",
    "imc":             "IMC (kg/m²)",
    "TFG":             "TFG (mL/min/1.73m²)",
}


# ─── Punto de entrada ─────────────────────────────────────────────────────────

def extraer_de_pdfs(archivos_bytes: list[bytes]) -> dict[str, Any]:
    campos_fusionados = _campos_vacios()

    for pdf_bytes in archivos_bytes:
        tipo   = _detectar_tipo(pdf_bytes)
        texto  = _extraer_texto(pdf_bytes, tipo)
        campos = _parsear_campos(texto)

        for clave, valor in campos.items():
            if campos_fusionados.get(clave) is None and valor is not None:
                campos_fusionados[clave] = valor

    # Derivar IMC si peso y talla presentes pero IMC ausente
    if campos_fusionados.get("imc") is None:
        peso  = campos_fusionados.get("peso")
        talla = campos_fusionados.get("talla")
        if peso and talla and talla > 0:
            talla_m = talla / 100.0
            campos_fusionados["imc"] = round(peso / (talla_m ** 2), 1)

    campos_fusionados["campos_faltantes"] = _listar_faltantes(campos_fusionados)
    return campos_fusionados


# ─── Detección de tipo ────────────────────────────────────────────────────────

def _detectar_tipo(pdf_bytes: bytes) -> str:
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            for page in pdf.pages:
                if page.extract_tables():
                    return "texto_tablas"
        return "texto_simple"
    except Exception:
        return "texto_simple"


# ─── Extracción de texto ──────────────────────────────────────────────────────

def _extraer_texto(pdf_bytes: bytes, tipo: str) -> str:
    if tipo == "texto_tablas":
        return _extraer_texto_tablas(pdf_bytes)
    return _extraer_texto_simple(pdf_bytes)


def _extraer_texto_simple(pdf_bytes: bytes) -> str:
    doc   = fitz.open(stream=pdf_bytes, filetype="pdf")
    texto = "\n".join(page.get_text() for page in doc)
    doc.close()
    return texto


def _extraer_texto_tablas(pdf_bytes: bytes) -> str:
    partes = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            texto_libre = page.extract_text() or ""
            partes.append(texto_libre)
            for tabla in page.extract_tables():
                for fila in tabla:
                    partes.append(" | ".join(str(c) if c else "" for c in fila))
    return "\n".join(partes)


# ─── Parseo principal ─────────────────────────────────────────────────────────

def _parsear_campos(texto: str) -> dict[str, Any]:
    t      = texto.lower()
    campos = _campos_vacios()

    # Laboratorio
    campos["c_total"]       = _extraer_num(t, ["c_total", "colesterol_total", "colesterol", "total_cholesterol"], 50, 500)
    campos["creatinina"]    = _extraer_num(t, ["creatinina", "creatinina_serica", "serum_creatinine"], 0.1, 15.0)
    campos["glucosa"]       = _extraer_num(t, ["glucosa", "glucosa_ayunas", "gluc", "glicemia"], 20, 600)
    campos["hdl"]           = _extraer_num(t, ["hdl", "colesterol_hdl", "hdl_colesterol"], 10, 150)
    campos["hemoglobina"]   = _extraer_num(t, ["hemoglobina", "hb", "hemoglobin"], 5, 25)
    campos["ldl"]           = _extraer_num(t, ["ldl", "colesterol_ldl", "ldl_colesterol"], 20, 500)
    campos["leucocitos"]    = _extraer_num(t, ["leucocitos", "wbc", "leucocytes", "white_blood_cells"], 1, 50)
    campos["plaquetas"]     = _extraer_num(t, ["plaquetas", "plt", "platelets"], 10, 1000)
    campos["trigliceridos"] = _extraer_num(t, ["trigliceridos", "tg", "triglycerides"], 20, 2000)

    # Demográficos
    campos["edad"]            = _extraer_edad(t)
    campos["sexo"]            = _extraer_sexo(t)
    campos["zona"]            = _extraer_zona(t)
    campos["ap_hipertension"] = _extraer_binario(t, ["ap_hipertension", "hta_personal",
                                                      "hipertension_arterial", "antecedente_hta"])

    # Signos vitales y antropometría
    campos["ta_sistolica"]  = _extraer_num(t, ["ta_sistolica", "sistolica", "systolic",
                                                "pas", "presion_sistolica"], 50, 300)
    campos["ta_diastolica"] = _extraer_num(t, ["ta_diastolica", "diastolica", "diastolic",
                                                "pad", "presion_diastolica"], 30, 200)
    campos["peso"]          = _extraer_num(t, ["peso", "peso_kg", "weight"], 5, 300)
    campos["talla"]         = _extraer_talla(t)   # devuelve siempre cm
    campos["imc"]           = _extraer_num(t, ["imc", "bmi", "indice_masa_corporal"], 10, 70)
    campos["TFG"]           = _extraer_num(t, ["tfg", "tfge", "filtracion_glomerular",
                                                "egfr", "gfr"], 5, 300)

    # Opcionales Framingham
    campos["colesterol_total_mgdl"]        = _extraer_num(t, ["colesterol_total_mgdl",
                                                               "colesterol_total", "ct",
                                                               "total_cholesterol"], 50, 600)
    campos["diabetes"]                     = _extraer_binario(t, ["diabetes", "dm",
                                                                   "diabetes_mellitus", "diabetico"])
    campos["tratamiento_antihipertensivo"] = _extraer_binario(t, ["tratamiento_antihipertensivo",
                                                                   "antihipertensivos", "hta_tratada"])
    campos["fuma"]                         = _extraer_binario(t, ["fuma", "fumador",
                                                                   "smoke", "tabaquismo"])
    return campos


# ─── Extractores genéricos ────────────────────────────────────────────────────

def _extraer_num(texto: str, aliases: list[str],
                 vmin: float, vmax: float) -> Optional[float]:
    for alias in aliases:
        patron = (r"(?:" + re.escape(alias) + r")" + SEP +
                  r"([\d.]+)\s*(?:mg/dl|%|g/dl|10³/µl|fl|kg|cm|m|mmhg|ml/min)?")
        m = re.search(patron, texto)
        if m:
            val = float(m.group(1))
            if vmin <= val <= vmax:
                return val
    return None


def _extraer_binario(texto: str, aliases: list[str]) -> Optional[int]:
    for alias in aliases:
        m = re.search(r"(?:" + re.escape(alias) + r")" + SEP + r"([01])", texto)
        if m:
            return int(m.group(1))
        if re.search(r"(?:" + re.escape(alias) + r")[:\s]*(s[ií]|yes|true|1|positivo|fumador)", texto):
            return 1
        if re.search(r"(?:" + re.escape(alias) + r")[:\s]*(no|none|false|0|negativo|no fumador)", texto):
            return 0
    return None


# ─── Extractores específicos ──────────────────────────────────────────────────

def _extraer_edad(texto: str) -> Optional[int]:
    for alias in ["edad", "edad_anos", "age", "edad_paciente"]:
        m = re.search(r"(?:" + re.escape(alias) + r")" + SEP + r"(\d{1,3})", texto)
        if m:
            val = int(m.group(1))
            if 6 <= val <= 110:
                return val
    m = re.search(r'(\d{4}-\d{2}-\d{2})', texto)
    if m:
        try:
            nac  = date.fromisoformat(m.group(1))
            edad = date.today().year - nac.year
            if (date.today().month, date.today().day) < (nac.month, nac.day):
                edad -= 1
            if 6 <= edad <= 110:
                return edad
        except ValueError:
            pass
    m = re.search(r'(\d{2,3})\s*años', texto)
    if m and 6 <= int(m.group(1)) <= 110:
        return int(m.group(1))
    return None


def _extraer_sexo(texto: str) -> Optional[int]:
    for alias in ["sexo", "sexo_codigo", "gender", "genero_codigo"]:
        m = re.search(r"(?:" + re.escape(alias) + r")" + SEP + r"([01])", texto)
        if m:
            return int(m.group(1))
    if re.search(r"sexo[:\s]*(femenino|mujer|f\b)", texto):
        return 0
    if re.search(r"sexo[:\s]*(masculino|hombre|m\b)", texto):
        return 1
    if re.search(r"g[eé]nero[:\s]*(femenino|mujer)", texto):
        return 0
    if re.search(r"g[eé]nero[:\s]*(masculino|hombre)", texto):
        return 1
    return None


def _extraer_zona(texto: str) -> Optional[int]:
    for alias in ["zona", "zona_residencia", "area"]:
        m = re.search(r"(?:" + re.escape(alias) + r")" + SEP + r"([01])", texto)
        if m:
            return int(m.group(1))
    if re.search(r"zona[:\s]*rural", texto) or re.search(r"área[:\s]*rural", texto):
        return 0
    if re.search(r"zona[:\s]*urbana", texto) or re.search(r"área[:\s]*urbana", texto):
        return 1
    return None


def _extraer_talla(texto: str) -> Optional[float]:
    """Extrae talla y devuelve siempre en cm."""
    for alias in ["talla", "talla_cm", "talla_m", "altura", "estatura", "height"]:
        patron = r"(?:" + re.escape(alias) + r")" + SEP + r"([\d.]+)\s*(m|cm)?"
        m = re.search(patron, texto)
        if m:
            val    = float(m.group(1))
            unidad = m.group(2) if m.lastindex >= 2 else ""
            if unidad == "m" or val <= 3.0:   # vino en metros → convertir
                val = val * 100.0
            if 100.0 <= val <= 220.0:
                return round(val, 1)
    return None


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _campos_vacios() -> dict[str, Any]:
    todos = list(CAMPOS_OBLIGATORIOS.keys()) + [
        "colesterol_total_mgdl", "diabetes",
        "tratamiento_antihipertensivo", "fuma",
    ]
    return {campo: None for campo in todos}


def _listar_faltantes(campos: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"campo": k, "descripcion": CAMPOS_OBLIGATORIOS[k]}
        for k in CAMPOS_OBLIGATORIOS
        if campos.get(k) is None
    ]