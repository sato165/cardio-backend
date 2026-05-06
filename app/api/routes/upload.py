import json
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Query

from app.schemas.output_schema import UploadOutput
from app.services.prediction_service import predecir_desde_extraccion
from app.services.json_extractor import extraer_de_json
from app.services.pdf_extractor  import extraer_de_pdfs
from app.core.config import settings

router = APIRouter()

BYTES_POR_MB = 1_048_576


def _aplicar_campos_manuales(campos: dict, manuales: dict) -> dict:
    """Rellena campos (obligatorios y opcionales) con los valores manuales proporcionados."""
    for clave, valor in manuales.items():
        if valor is not None and campos.get(clave) is None:
            try:
                # Convertir a número si corresponde
                campos[clave] = int(valor) if '.' not in str(valor) else float(valor)
            except (ValueError, TypeError):
                pass
    # Recalcular campos_faltantes (ahora los 22 obligatorios del modelo)
    from app.services.json_extractor import CAMPOS_OBLIGATORIOS  # actualizado
    campos['campos_faltantes'] = [
        {'campo': k, 'descripcion': CAMPOS_OBLIGATORIOS[k]}
        for k in CAMPOS_OBLIGATORIOS
        if campos.get(k) is None
    ]
    return campos


# Lista de los 22 campos obligatorios para la interfaz Query
OBLIGATORIOS_QUERY = [
    "creatinina", "celulas_medias", "glucosa", "granulocitos",
    "hdl", "hematocrito", "hemoglobina", "ldl", "leucocitos",
    "linfocitos", "plaquetas", "trigliceridos", "edad", "sexo",
    "zona", "ap_hipertension", "ta_sistolica", "ta_diastolica",
    "peso", "talla", "imc", "TFG"
]
OPCIONALES_QUERY = [
    "colesterol_total_mgdl", "diabetes", "tratamiento_antihipertensivo", "fuma"
]


# ── JSON ──────────────────────────────────────────────────────────────────────

@router.post(
    '/upload',
    response_model=UploadOutput,
    summary='Predicción desde historia clínica en JSON',
)
async def predict_upload_json(
    archivo: UploadFile = File(...),
    # 22 obligatorios
    creatinina:      Optional[float] = Query(None),
    celulas_medias:  Optional[float] = Query(None),
    glucosa:         Optional[float] = Query(None),
    granulocitos:    Optional[float] = Query(None),
    hdl:             Optional[float] = Query(None),
    hematocrito:     Optional[float] = Query(None),
    hemoglobina:     Optional[float] = Query(None),
    ldl:             Optional[float] = Query(None),
    leucocitos:      Optional[float] = Query(None),
    linfocitos:      Optional[float] = Query(None),
    plaquetas:       Optional[float] = Query(None),
    trigliceridos:   Optional[float] = Query(None),
    edad:            Optional[int]   = Query(None),
    sexo:            Optional[int]   = Query(None),
    zona:            Optional[int]   = Query(None),
    ap_hipertension: Optional[int]   = Query(None),
    ta_sistolica:    Optional[float] = Query(None),
    ta_diastolica:   Optional[float] = Query(None),
    peso:            Optional[float] = Query(None),
    talla:           Optional[float] = Query(None),
    imc:             Optional[float] = Query(None),
    TFG:             Optional[float] = Query(None),
    # Opcionales Framingham
    colesterol_total_mgdl:      Optional[float] = Query(None),
    diabetes:                   Optional[int]   = Query(None),
    tratamiento_antihipertensivo: Optional[int] = Query(None),
    fuma:                       Optional[int]   = Query(None),
) -> UploadOutput:
    _validar_tamano(archivo, settings.max_upload_size_mb)
    _validar_extension(archivo.filename, ['.json'])

    contenido = await archivo.read()
    try:
        datos = json.loads(contenido)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f'El archivo no es un JSON válido: {e}')

    campos = extraer_de_json(datos)  # devuelve dict con los 22 + opcionales y campos_faltantes

    # Construir dict de manuales combinando obligatorios y opcionales
    manuales = {}
    for campo in OBLIGATORIOS_QUERY + OPCIONALES_QUERY:
        valor = locals().get(campo)
        if valor is not None:
            manuales[campo] = valor

    if manuales:
        campos = _aplicar_campos_manuales(campos, manuales)

    result = predecir_desde_extraccion(campos)
    return result


# ── PDF ───────────────────────────────────────────────────────────────────────

@router.post(
    '/upload/pdf',
    response_model=UploadOutput,
    summary='Predicción desde historia clínica en PDF',
)
async def predict_upload_pdf(
    archivos: list[UploadFile] = File(...),
    # 22 obligatorios
    creatinina:      Optional[float] = Query(None),
    celulas_medias:  Optional[float] = Query(None),
    glucosa:         Optional[float] = Query(None),
    granulocitos:    Optional[float] = Query(None),
    hdl:             Optional[float] = Query(None),
    hematocrito:     Optional[float] = Query(None),
    hemoglobina:     Optional[float] = Query(None),
    ldl:             Optional[float] = Query(None),
    leucocitos:      Optional[float] = Query(None),
    linfocitos:      Optional[float] = Query(None),
    plaquetas:       Optional[float] = Query(None),
    trigliceridos:   Optional[float] = Query(None),
    edad:            Optional[int]   = Query(None),
    sexo:            Optional[int]   = Query(None),
    zona:            Optional[int]   = Query(None),
    ap_hipertension: Optional[int]   = Query(None),
    ta_sistolica:    Optional[float] = Query(None),
    ta_diastolica:   Optional[float] = Query(None),
    peso:            Optional[float] = Query(None),
    talla:           Optional[float] = Query(None),
    imc:             Optional[float] = Query(None),
    TFG:             Optional[float] = Query(None),
    # Opcionales Framingham
    colesterol_total_mgdl:      Optional[float] = Query(None),
    diabetes:                   Optional[int]   = Query(None),
    tratamiento_antihipertensivo: Optional[int] = Query(None),
    fuma:                       Optional[int]   = Query(None),
) -> UploadOutput:
    if len(archivos) > 5:
        raise HTTPException(status_code=422, detail='Se permiten máximo 5 PDFs por solicitud.')

    pdfs_bytes = []
    for archivo in archivos:
        _validar_tamano(archivo, settings.max_upload_size_mb)
        _validar_extension(archivo.filename, ['.pdf'])
        pdfs_bytes.append(await archivo.read())

    campos = extraer_de_pdfs(pdfs_bytes)

    manuales = {}
    for campo in OBLIGATORIOS_QUERY + OPCIONALES_QUERY:
        valor = locals().get(campo)
        if valor is not None:
            manuales[campo] = valor

    if manuales:
        campos = _aplicar_campos_manuales(campos, manuales)

    result = predecir_desde_extraccion(campos)
    return result


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validar_tamano(archivo: UploadFile, max_mb: int) -> None:
    if archivo.size and archivo.size > max_mb * BYTES_POR_MB:
        raise HTTPException(
            status_code=413,
            detail=f"El archivo '{archivo.filename}' supera el límite de {max_mb} MB.",
        )


def _validar_extension(nombre: str, extensiones: list[str]) -> None:
    if not any(nombre.lower().endswith(ext) for ext in extensiones):
        raise HTTPException(
            status_code=422,
            detail=f'Formato no válido. Se esperaba: {", ".join(extensiones)}.',
        )