from pydantic import BaseModel, Field
from typing import Optional


class FactorExplicacion(BaseModel):
    factor:      str   = Field(..., description="Nombre clínico del factor")
    impacto:     float = Field(..., description="Valor SHAP: positivo aumenta el riesgo, negativo lo reduce")
    descripcion: str   = Field(..., description="Frase en lenguaje clínico para el médico")
    nivel:       str   = Field(..., description="'crítico' | 'moderado' | 'leve'")
    advertencia: Optional[str] = Field(None, description="Advertencia de sesgo si aplica (smoke, alco)")


class PredictionOutput(BaseModel):
    riesgo_cardiovascular: int   = Field(..., description="0 = bajo riesgo · 1 = alto riesgo")
    probabilidad:          float = Field(..., description="Probabilidad de riesgo entre 0 y 1")
    nivel_riesgo:          str   = Field(..., description="'Alto' | 'Moderado' | 'Bajo'")
    explicabilidad:        list[FactorExplicacion] = Field(..., description="Factores ordenados por impacto absoluto")


class CampoFaltante(BaseModel):
    campo:       str = Field(..., description="Nombre técnico del campo faltante")
    descripcion: str = Field(..., description="Descripción legible para el médico")


class UploadOutput(BaseModel):
    """
    Respuesta del endpoint de carga de historia clínica JSON.
    Si hay campos faltantes, la predicción es None y se listan los campos
    que el médico debe completar manualmente antes de reintentar.
    """
    campos_faltantes:      list[CampoFaltante]         = Field(default_factory=list)
    prediccion:            Optional[PredictionOutput]  = Field(None)
    mensaje:               str                         = Field(..., description="Mensaje informativo para el médico")
