from fastapi import APIRouter
from app.schemas.input_schema  import CardiovascularInput
from app.schemas.output_schema import PredictionOutput
from app.services.prediction_service import predecir_desde_formulario

router = APIRouter()


@router.post("/", response_model=PredictionOutput, summary="Predicción desde formulario manual")
def predict_manual(datos: CardiovascularInput) -> PredictionOutput:
    """
    Recibe los datos del paciente ingresados manualmente y retorna
    la predicción de riesgo cardiovascular con explicabilidad clínica.
    """
    return predecir_desde_formulario(datos)
