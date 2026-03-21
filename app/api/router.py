from fastapi import APIRouter

from app.api.routes import predict, upload

router = APIRouter()

router.include_router(predict.router, prefix="/predict", tags=["Predicción"])
router.include_router(upload.router, prefix="/predict", tags=["Carga de historia clínica"])
