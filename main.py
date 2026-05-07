import warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.staticfiles import StaticFiles   # <-- añadido
from pydantic import ValidationError
import os                                      # <-- añadido

from app.core.config import settings
from app.api.router import router

app = FastAPI(
    title="CardioPredict API",
    description="API para predicción de riesgo cardiovascular mediante clustering con inteligencia artificial.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


# ── Manejadores de errores ────────────────────────────────────────────────────
# (Todos los tuyos se mantienen sin cambios)

@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    errores = []
    for error in exc.errors():
        campo = " → ".join(str(loc) for loc in error["loc"] if loc != "body")
        errores.append({
            "campo":   campo,
            "mensaje": error["msg"],
            "valor":   error.get("input"),
        })
    return JSONResponse(
        status_code=422,
        content={
            "error":   "Datos de entrada inválidos",
            "detalle": errores,
        },
    )

@app.exception_handler(ValidationError)
async def pydantic_error_handler(request: Request, exc: ValidationError):
    errores = []
    for error in exc.errors():
        campo = " → ".join(str(loc) for loc in error["loc"])
        errores.append({
            "campo":   campo,
            "mensaje": error["msg"],
            "valor":   error.get("input"),
        })
    return JSONResponse(
        status_code=422,
        content={
            "error":   "Los valores extraídos del archivo no son válidos para el modelo",
            "detalle": errores,
            "sugerencia": (
                "Verifique que los valores estén dentro de los rangos clínicos aceptados "
                "para cada una de las 22 variables (creatinina, glucosa, presión arterial, etc.)."
            ),
        },
    )

@app.exception_handler(FileNotFoundError)
async def model_not_found_handler(request: Request, exc: FileNotFoundError):
    return JSONResponse(
        status_code=503,
        content={
            "error":      "Modelo o artefactos no disponibles",
            "detalle":    str(exc),
            "sugerencia": (
                "Verifique que los archivos random_forest.pkl, scaler.pkl y pca.pkl "
                "estén en la carpeta models/ y que las rutas en .env sean correctas."
            ),
        },
    )

@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "error":   "Error interno del servidor",
            "detalle": str(exc),
        },
    )


@app.get("/api/health", tags=["Health"])
def health_check():
    return {"status": "ok", "modelo": "RandomForest con clustering PCA"}


# ── Servir el frontend (React) ────────────────────────────────────────────────
frontend_path = os.path.join(os.path.dirname(__file__), "frontend_dist")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="static")