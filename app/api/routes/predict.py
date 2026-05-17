from fastapi import APIRouter
from app.schemas.input_schema  import PredictionInput
from app.schemas.output_schema import PredictionOutput
from app.services.prediction_service import predecir_desde_formulario

router = APIRouter()


@router.post(
    "/",
    response_model=PredictionOutput,
    summary="Predicción desde formulario manual",
    description=(
        "Recibe las variables clínicas del paciente y retorna su perfil de riesgo "
        "cardiovascular. El modelo asigna al paciente a uno de 4 clusters: "
        "Cardiovascular, Bajo riesgo, Cardiometabólico o Cardiorrenal, "
        "junto con las probabilidades de pertenencia a cada uno."
    ),
)
def predict_manual(datos: PredictionInput) -> PredictionOutput:
    return predecir_desde_formulario(datos)