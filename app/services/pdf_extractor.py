"""
pdf_extractor.py
Extrae los campos del modelo desde historias clínicas en PDF.

Soporta tres tipos de PDF:
  - Tipo 1: texto nativo simple       → PyMuPDF (fitz)
  - Tipo 2: texto nativo con tablas   → pdfplumber
  - Tipo 3: escaneado (solo imágenes) → pdf2image + pytesseract

pdfplumber genera dos formatos en la misma cadena de texto:
  - Texto libre:  "altura (cm) 172"
  - Filas tabla:  "altura (cm) | 172"
Los patrones cubren ambos con el separador opcional [|\s]+
"""

import re
import fitz
import pdfplumber
import io
from typing import Any

try:
    import pytesseract
    from pdf2image import convert_from_bytes
    OCR_DISPONIBLE = True
except ImportError:
    OCR_DISPONIBLE = False

MIN_CHARS_TEXTO = 50


# ---------------------------------------------------------------------------
# Punto de entrada público
# ---------------------------------------------------------------------------

def extraer_de_pdfs(archivos_bytes: list[bytes]) -> dict[str, Any]:
    campos_fusionados: dict[str, Any] = _campos_vacios()

    for pdf_bytes in archivos_bytes:
        tipo   = _detectar_tipo(pdf_bytes)
        texto  = _extraer_texto(pdf_bytes, tipo)
        campos = _parsear_campos(texto)

        for clave, valor in campos.items():
            if campos_fusionados.get(clave) is None and valor is not None:
                campos_fusionados[clave] = valor

    campos_fusionados["tipo_pdf_detectado"] = _detectar_tipo(archivos_bytes[0])
    campos_fusionados["campos_faltantes"]   = _listar_faltantes(campos_fusionados)
    return campos_fusionados


# ---------------------------------------------------------------------------
# Detección de tipo
# ---------------------------------------------------------------------------

def _detectar_tipo(pdf_bytes: bytes) -> str:
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        chars_totales = sum(len(page.get_text()) for page in doc)
        num_paginas   = len(doc)
        doc.close()

        if chars_totales < MIN_CHARS_TEXTO * num_paginas:
            return "escaneado"

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            tiene_tablas = any(page.extract_tables() for page in pdf.pages)

        return "texto_tablas" if tiene_tablas else "texto_simple"

    except Exception:
        return "texto_simple"


# ---------------------------------------------------------------------------
# Extracción de texto
# ---------------------------------------------------------------------------

def _extraer_texto(pdf_bytes: bytes, tipo: str) -> str:
    if tipo == "escaneado":
        return _extraer_texto_ocr(pdf_bytes)
    elif tipo == "texto_tablas":
        return _extraer_texto_tablas(pdf_bytes)
    return _extraer_texto_simple(pdf_bytes)


def _extraer_texto_simple(pdf_bytes: bytes) -> str:
    doc   = fitz.open(stream=pdf_bytes, filetype="pdf")
    texto = "\n".join(page.get_text() for page in doc)
    doc.close()
    return texto


def _extraer_texto_tablas(pdf_bytes: bytes) -> str:
    """
    Combina texto libre y filas de tabla en un solo string.
    Resultado: mezcla de "label valor" y "label | valor | ..."
    Los patrones usan [|\\s]+ como separador para cubrir ambos.
    """
    partes = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            texto_libre = page.extract_text() or ""
            partes.append(texto_libre)
            for tabla in page.extract_tables():
                for fila in tabla:
                    partes.append(" | ".join(str(c) if c else "" for c in fila))
    return "\n".join(partes)


def _extraer_texto_ocr(pdf_bytes: bytes) -> str:
    if not OCR_DISPONIBLE:
        return ""
    imagenes = convert_from_bytes(pdf_bytes, dpi=300)
    partes   = []
    for imagen in imagenes:
        texto = pytesseract.image_to_string(imagen, lang="spa+eng", config="--psm 6")
        if len(texto.strip()) >= MIN_CHARS_TEXTO:
            partes.append(texto)
    return "\n".join(partes)


# ---------------------------------------------------------------------------
# Parsing — separador flexible [|\s]+ cubre "label valor" y "label | valor"
# ---------------------------------------------------------------------------

SEP = r"[\s|]+"   # separador entre label y valor en pdfplumber


def _parsear_campos(texto: str) -> dict[str, Any]:
    campos: dict[str, Any] = _campos_vacios()
    t = texto.lower()

    campos["age_days"]    = _extraer_age_days(t)
    campos["gender"]      = _extraer_gender(t)
    campos["height"]      = _extraer_height(t)
    campos["weight"]      = _extraer_weight(t)
    campos["ap_hi"]       = _extraer_ap_hi(t)
    campos["ap_lo"]       = _extraer_ap_lo(t)
    campos["cholesterol"] = _extraer_cholesterol(t)
    campos["gluc"]        = _extraer_gluc(t)
    campos["smoke"]       = _extraer_smoke(t)
    campos["alco"]        = _extraer_alco(t)
    campos["active"]      = _extraer_active(t)

    return campos


def _primero(texto: str, patrones: list[str]):
    """Retorna el primer grupo capturado del primer patrón que coincida."""
    for p in patrones:
        m = re.search(p, texto)
        if m:
            return m.group(1)
    return None


# ---------------------------------------------------------------------------
# Extractores por campo
# ---------------------------------------------------------------------------

def _extraer_age_days(t: str) -> int | None:
    v = _primero(t, [
        r"edad[_\s]*d[ií]as" + SEP + r"(\d+)",
        r"age_days" + SEP + r"(\d+)",
        r"edad \(d[ií]as\)" + SEP + r"(\d+)",
    ])
    if v:
        dias = int(v)
        if 6000 <= dias <= 40000:
            return dias

    # Edad en años → convertir
    v = _primero(t, [
        r"edad \(anos\)" + SEP + r"(\d{2})",
        r"edad[:\s]+(\d{2})\s*a[ñn]",
        r"age[:\s]+(\d{2})\s*year",
        r"(\d{2})\s*a[ñn]os\b",
    ])
    if v and 18 <= int(v) <= 90:
        return round(int(v) * 365.25)

    # Desde fecha de nacimiento
    m = re.search(r'(\d{4}-\d{2}-\d{2})', t)
    if m:
        from datetime import date
        try:
            dias = (date.today() - date.fromisoformat(m.group(1))).days
            if 6000 <= dias <= 40000:
                return dias
        except ValueError:
            pass

    return None


def _extraer_gender(t: str) -> int | None:
    v = _primero(t, [
        r"genero\s*codigo" + SEP + r"([12])",
        r"gender\s*code" + SEP + r"([12])",
        r"gender" + SEP + r"([12])\b",
        r"sexo" + SEP + r"([mf])\b",
    ])
    if v:
        if v in ('1', 'f'):
            return 1
        if v in ('2', 'm'):
            return 2

    if re.search(r"sexo\s*[|\s]+m\b|g[eé]nero[:\s]*(masculino|hombre)", t):
        return 2
    if re.search(r"sexo\s*[|\s]+f\b|g[eé]nero[:\s]*(femenino|mujer)", t):
        return 1

    return None


def _extraer_height(t: str) -> int | None:
    v = _primero(t, [
        r"altura\s*\(cm\)" + SEP + r"(\d{3})",
        r"talla\s*\(cm\)" + SEP + r"(\d{3})",
        r"height_cm" + SEP + r"(\d{3})",
        r"height\s*\(cm\)" + SEP + r"(\d{3})",
        r"altura[:\s]+(\d{3})",
        r"talla[:\s]+(\d{3})",
    ])
    if v and 140 <= int(v) <= 220:
        return int(v)
    return None


def _extraer_weight(t: str) -> float | None:
    v = _primero(t, [
        r"peso\s*\(kg\)" + SEP + r"([\d.]+)",
        r"weight_kg" + SEP + r"([\d.]+)",
        r"weight\s*\(kg\)" + SEP + r"([\d.]+)",
        r"peso[:\s]+([\d.]+)\s*kg",
    ])
    if v and 30 <= float(v) <= 180:
        return float(v)
    return None


def _extraer_ap_hi(t: str) -> int | None:
    v = _primero(t, [
        r"presion\s*sistolica\s*\(mmhg\)" + SEP + r"(\d{2,3})",
        r"presion\s*sistolica[:\s]+(\d{2,3})",
        r"ap_hi" + SEP + r"(\d{2,3})",
        r"systolic[:\s]+(\d{2,3})",
        r"([\d]{2,3})\s*/\s*[\d]{2,3}\s*mmhg",
    ])
    if v and 60 <= int(v) <= 250:
        return int(v)
    return None


def _extraer_ap_lo(t: str) -> int | None:
    # Verificar explícitamente si está marcado como no registrado
    if re.search(r"presion\s*diastolica.*?(no\s*registrado|faltante|null)", t):
        return None
    if re.search(r"ap_lo" + SEP + r"(faltante|null|none|-)", t):
        return None

    v = _primero(t, [
        r"presion\s*diastolica\s*\(mmhg\)" + SEP + r"(\d{2,3})",
        r"presion\s*diastolica[:\s]+(\d{2,3})",
        r"ap_lo" + SEP + r"(\d{2,3})",
        r"diastolic[:\s]+(\d{2,3})",
        r"[\d]{2,3}\s*/\s*([\d]{2,3})\s*mmhg",
    ])
    if v and 40 <= int(v) <= 200:
        return int(v)
    return None


def _extraer_cholesterol(t: str) -> int | None:
    # Código directo del resumen del modelo
    v = _primero(t, [
        r"colesterol[_\s]*c[oó]digo[_\s]*modelo" + SEP + r"([123])",
        r"^cholesterol" + SEP + r"([123])\b",
    ])
    if v:
        return int(v)

    # Código en línea de laboratorio: "colesterol total 238 mg/dl 2 (por encima normal)"
    m = re.search(r"colesterol\s*total" + SEP + r"[\d.]+" + SEP + r"\S+" + SEP + r"([123])\s*\(", t)
    if m:
        return int(m.group(1))

    # Valor mg/dL
    v = _primero(t, [r"colesterol\s*total" + SEP + r"([\d.]+)\s*mg"])
    if v:
        return _colesterol_a_codigo(float(v))

    return None


def _extraer_gluc(t: str) -> int | None:
    # Código directo
    v = _primero(t, [
        r"glucosa[_\s]*c[oó]digo[_\s]*modelo" + SEP + r"([123])",
        r"^gluc" + SEP + r"([123])\b",
    ])
    if v:
        return int(v)

    # Línea de laboratorio: "glucosa en ayunas 112 mg/dl 2 (por encima normal)"
    m = re.search(r"glucosa en ayunas.*?([123])\s*\(", t)
    if m:
        return int(m.group(1))

    # Valor mg/dL
    v = _primero(t, [
        r"glucosa\s*en\s*ayunas" + SEP + r"([\d.]+)",
        r"glucose[:\s]+([\d.]+)\s*mg",
    ])
    if v:
        return _glucosa_a_codigo(float(v))

    return None


def _extraer_smoke(t: str) -> int | None:
    # Código explícito "smoke: 0/1" dentro de la línea
    v = _primero(t, [r"smoke[:\s]*(\d)"])
    if v:
        return int(v)

    # Resumen del modelo: "smoke | 0 | no fuma"
    v = _primero(t, [r"^smoke" + SEP + r"([01])\b"])
    if v:
        return int(v)

    return None


def _extraer_alco(t: str) -> int | None:
    v = _primero(t, [r"alco[:\s]*(\d)"])
    if v:
        return int(v)

    v = _primero(t, [r"^alco" + SEP + r"([01])\b"])
    if v:
        return int(v)

    return None


def _extraer_active(t: str) -> int | None:
    v = _primero(t, [r"active[:\s]*(\d)"])
    if v:
        return int(v)

    v = _primero(t, [r"^active" + SEP + r"([01])\b"])
    if v:
        return int(v)

    return None


# ---------------------------------------------------------------------------
# Conversiones clínicas
# ---------------------------------------------------------------------------

def _colesterol_a_codigo(mgdl: float) -> int:
    if mgdl < 200:
        return 1
    elif mgdl < 240:
        return 2
    return 3


def _glucosa_a_codigo(mgdl: float) -> int:
    if mgdl < 100:
        return 1
    elif mgdl < 126:
        return 2
    return 3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CAMPOS_REQUERIDOS = {
    "age_days":    "Edad (en años o días)",
    "gender":      "Género (masculino / femenino)",
    "height":      "Altura (cm)",
    "weight":      "Peso (kg)",
    "ap_hi":       "Presión sistólica (mmHg)",
    "ap_lo":       "Presión diastólica (mmHg)",
    "cholesterol": "Colesterol (normal / alto / muy alto)",
    "gluc":        "Glucosa (normal / alta / muy alta)",
    "smoke":       "Tabaquismo (sí / no)",
    "alco":        "Consumo de alcohol (sí / no)",
    "active":      "Actividad física (sí / no)",
}


def _campos_vacios() -> dict[str, Any]:
    return {campo: None for campo in CAMPOS_REQUERIDOS}


def _listar_faltantes(campos: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"campo": k, "descripcion": CAMPOS_REQUERIDOS[k]}
        for k in CAMPOS_REQUERIDOS
        if campos.get(k) is None
    ]
