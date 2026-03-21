import json
from fastapi import APIRouter, UploadFile, File, HTTPException

from app.schemas.output_schema import UploadOutput
from app.services.prediction_service import predecir_desde_extraccion
from app.services.json_extractor import extraer_de_json
from app.services.pdf_extractor  import extraer_de_pdfs
from app.core.config import settings

router = APIRouter()

BYTES_POR_MB = 1_048_576


# ── JSON ──────────────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=UploadOutput,
    summary="Predicción desde historia clínica en JSON",
)
async def predict_upload_json(archivo: UploadFile = File(...)) -> UploadOutput:
    """
    Recibe un archivo JSON de historia clínica, extrae los campos del modelo
    y retorna la predicción. Si algún campo falta, indica cuáles completar.
    """
    _validar_tamano(archivo, settings.max_upload_size_mb)
    _validar_extension(archivo.filename, [".json"])

    contenido = await archivo.read()

    try:
        datos = json.loads(contenido)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=422, detail=f"El archivo no es un JSON válido: {e}")

    campos = extraer_de_json(datos)
    return predecir_desde_extraccion(campos)


# ── PDF ───────────────────────────────────────────────────────────────────────

@router.post(
    "/upload/pdf",
    response_model=UploadOutput,
    summary="Predicción desde historia clínica en PDF (uno o varios del mismo paciente)",
)
async def predict_upload_pdf(
    archivos: list[UploadFile] = File(..., description="Uno o más PDFs del mismo paciente"),
) -> UploadOutput:
    """
    Recibe uno o varios archivos PDF del mismo paciente, detecta automáticamente
    el tipo de cada PDF (texto, tablas o escaneado), extrae los campos del modelo,
    fusiona los resultados y retorna la predicción. Si algún campo falta tras
    revisar todos los PDFs, indica cuáles completar manualmente.
    """
    if len(archivos) > 5:
        raise HTTPException(
            status_code=422,
            detail="Se permiten máximo 5 PDFs por solicitud.",
        )

    pdfs_bytes = []
    for archivo in archivos:
        _validar_tamano(archivo, settings.max_upload_size_mb)
        _validar_extension(archivo.filename, [".pdf"])
        pdfs_bytes.append(await archivo.read())

    campos = extraer_de_pdfs(pdfs_bytes)
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
            detail=f"Formato no válido. Se esperaba: {', '.join(extensiones)}.",
        )
