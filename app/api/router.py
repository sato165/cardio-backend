from fastapi import APIRouter

from app.api.routes import predict, upload, explain

router = APIRouter()

router.include_router(predict.router, prefix="/predict", tags=["Predicción"])
router.include_router(upload.router, tags=["Carga de historia clínica"])
router.include_router(explain.router, prefix="/predict", tags=["Explicabilidad"])   