# cardio-backend/app/schemas/output_schema.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict


class ClusterProb(BaseModel):
    cluster_id:   int   = Field(..., description="Identificador del cluster (0,1,2,3)")
    cluster_name: str   = Field(..., description="Nombre clínico del perfil")
    probability:  float = Field(..., description="Probabilidad de pertenencia (0-1)")


class RiesgoComparativo(BaseModel):
    """Resultados de Framingham y SCC."""
    framingham_porcentaje:          Optional[float] = None
    framingham_nivel:               Optional[str]   = None
    scc_porcentaje:                 Optional[float] = None
    scc_nivel:                      Optional[str]   = None
    datos_suficientes:              bool             = False
    campos_faltantes_framingham:    List[str]        = Field(default_factory=list)


class PredictionOutput(BaseModel):
    """Respuesta de la predicción principal — k=4 clusters."""
    predicted_cluster:  int               = Field(..., description="Cluster predicho (0,1,2,3)")
    cluster_name:       str               = Field(..., description="Nombre del perfil clínico")
    description:        str               = Field(..., description="Interpretación y recomendaciones")
    probabilities:      List[ClusterProb] = Field(..., description="Probabilidad por cluster (4 elementos)")
    riesgo_comparativo: Optional[RiesgoComparativo] = Field(
        None, description="Resultado de Framingham/SCC si se calculó"
    )


class CampoFaltante(BaseModel):
    campo:       str
    descripcion: str


class DatosPaciente(BaseModel):
    """Campos extraídos en la carga de archivos — variables del modelo final."""
    # Laboratorio
    c_total:        Optional[float] = None
    creatinina:     Optional[float] = None
    glucosa:        Optional[float] = None
    hdl:            Optional[float] = None
    hemoglobina:    Optional[float] = None
    ldl:            Optional[float] = None
    leucocitos:     Optional[float] = None
    plaquetas:      Optional[float] = None
    trigliceridos:  Optional[float] = None
    # Demográficos
    edad:            Optional[int]  = None
    sexo:            Optional[int]  = None
    zona:            Optional[int]  = None
    ap_hipertension: Optional[int]  = None
    # Signos vitales y antropometría
    ta_sistolica:   Optional[float] = None
    ta_diastolica:  Optional[float] = None
    peso:           Optional[float] = None
    talla:          Optional[float] = None   # en cm
    imc:            Optional[float] = None
    TFG:            Optional[float] = None
    # Extras Framingham
    colesterol_total_mgdl:        Optional[float] = None
    diabetes:                     Optional[int]   = None
    tratamiento_antihipertensivo: Optional[int]   = None
    fuma:                         Optional[int]   = None


class UploadOutput(BaseModel):
    campos_faltantes:    List[CampoFaltante]        = Field(default_factory=list)
    prediccion:          Optional[PredictionOutput] = None
    mensaje:             str                        = ""
    datos_paciente:      Optional[DatosPaciente]    = None
    framingham_faltante: Optional[List[CampoFaltante]] = None


# ──────────────────────────────────────────────────────────
# ESQUEMAS PARA EXPLICABILIDAD (SHAP)
# ──────────────────────────────────────────────────────────

class FeatureSHAP(BaseModel):
    feature:       str            = Field(..., description="Nombre de la variable clínica")
    shap_value:    float          = Field(..., description="Contribución SHAP para el cluster predicho")
    feature_value: Optional[float] = Field(None, description="Valor real del paciente (winsorizado)")


class ExplainabilityOutput(BaseModel):
    predicted_cluster: int                       = Field(..., description="Cluster predicho (0,1,2,3)")
    cluster_name:      str                       = Field(..., description="Nombre del perfil clínico")
    shap_values:       Dict[str, List[FeatureSHAP]] = Field(
        ..., description="Nombre de cluster → lista de contribuciones SHAP por feature"
    )
    base_values:       Dict[str, float]          = Field(
        ..., description="Valor base (expected value) del modelo para cada cluster"
    )