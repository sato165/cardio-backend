from fastapi import APIRouter
from app.schemas.input_schema  import PredictionInput
from app.schemas.output_schema import PredictionOutput
from app.services.prediction_service import predecir_desde_formulario

router = APIRouter()


@router.post("/", response_model=PredictionOutput, summary="Predicción desde formulario manual")
def predict_manual(datos: PredictionInput) -> PredictionOutput:
    """
    Recibe los datos del paciente (22 variables) y retorna
    la predicción del cluster de riesgo cardiovascular con sus probabilidades.
    """
    return predecir_desde_formulario(datos)