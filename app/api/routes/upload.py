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
    """Rellena los campos faltantes con los valores ingresados manualmente."""
    for clave, valor in manuales.items():
        if valor is not None and campos.get(clave) is None:
            try:
                campos[clave] = int(valor) if '.' not in str(valor) else float(valor)
            except (ValueError, TypeError):
                pass
    # Recalcular campos_faltantes tras completar
    from app.services.json_extractor import CAMPOS_REQUERIDOS
    campos['campos_faltantes'] = [
        {'campo': k, 'descripcion': CAMPOS_REQUERIDOS[k]}
        for k in CAMPOS_REQUERIDOS
        if campos.get(k) is None
    ]
    return campos


# ── JSON ──────────────────────────────────────────────────────────────────────

@router.post(
    '/upload',
    response_model=UploadOutput,
    summary='Predicción desde historia clínica en JSON',
)
async def predict_upload_json(
    archivo: UploadFile = File(...),
    ap_lo:       Optional[float] = Query(None),
    ap_hi:       Optional[float] = Query(None),
    age_days:    Optional[int]   = Query(None),
    gender:      Optional[int]   = Query(None),
    height:      Optional[int]   = Query(None),
    weight:      Optional[float] = Query(None),
    cholesterol: Optional[int]   = Query(None),
    gluc:        Optional[int]   = Query(None),
    smoke:       Optional[int]   = Query(None),
    alco:        Optional[int]   = Query(None),
    active:      Optional[int]   = Query(None),
) -> UploadOutput:
    _validar_tamano(archivo, settings.max_upload_size_mb)
    _validar_extension(archivo.filename, ['.json'])

    contenido = await archivo.read()
    try:
        datos = json.loads(contenido)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f'El archivo no es un JSON válido: {e}')

    campos = extraer_de_json(datos)

    manuales = {k: v for k, v in {
        'ap_lo': ap_lo, 'ap_hi': ap_hi, 'age_days': age_days,
        'gender': gender, 'height': height, 'weight': weight,
        'cholesterol': cholesterol, 'gluc': gluc,
        'smoke': smoke, 'alco': alco, 'active': active,
    }.items() if v is not None}

    if manuales:
        campos = _aplicar_campos_manuales(campos, manuales)

    return predecir_desde_extraccion(campos)


# ── PDF ───────────────────────────────────────────────────────────────────────

@router.post(
    '/upload/pdf',
    response_model=UploadOutput,
    summary='Predicción desde historia clínica en PDF',
)
async def predict_upload_pdf(
    archivos:    list[UploadFile] = File(...),
    ap_lo:       Optional[float] = Query(None),
    ap_hi:       Optional[float] = Query(None),
    age_days:    Optional[int]   = Query(None),
    gender:      Optional[int]   = Query(None),
    height:      Optional[int]   = Query(None),
    weight:      Optional[float] = Query(None),
    cholesterol: Optional[int]   = Query(None),
    gluc:        Optional[int]   = Query(None),
    smoke:       Optional[int]   = Query(None),
    alco:        Optional[int]   = Query(None),
    active:      Optional[int]   = Query(None),
) -> UploadOutput:
    if len(archivos) > 5:
        raise HTTPException(status_code=422, detail='Se permiten máximo 5 PDFs por solicitud.')

    pdfs_bytes = []
    for archivo in archivos:
        _validar_tamano(archivo, settings.max_upload_size_mb)
        _validar_extension(archivo.filename, ['.pdf'])
        pdfs_bytes.append(await archivo.read())

    campos = extraer_de_pdfs(pdfs_bytes)

    manuales = {k: v for k, v in {
        'ap_lo': ap_lo, 'ap_hi': ap_hi, 'age_days': age_days,
        'gender': gender, 'height': height, 'weight': weight,
        'cholesterol': cholesterol, 'gluc': gluc,
        'smoke': smoke, 'alco': alco, 'active': active,
    }.items() if v is not None}

    if manuales:
        campos = _aplicar_campos_manuales(campos, manuales)

    return predecir_desde_extraccion(campos)


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
