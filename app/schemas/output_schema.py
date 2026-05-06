from pydantic import BaseModel, Field
from typing import Optional, List


class ClusterProb(BaseModel):
    cluster_id:   int   = Field(..., description="Identificador del cluster (0,1,2)")
    cluster_name: str   = Field(..., description="Nombre clínico del perfil")
    probability:  float = Field(..., description="Probabilidad de pertenencia (0-1)")


class RiesgoComparativo(BaseModel):
    """Resultados de Framingham y SCC (se mantienen igual que antes)."""
    framingham_porcentaje: Optional[float] = None
    framingham_nivel: Optional[str] = None
    scc_porcentaje: Optional[float] = None
    scc_nivel: Optional[str] = None
    datos_suficientes: bool = False
    campos_faltantes_framingham: List[str] = Field(default_factory=list)


class PredictionOutput(BaseModel):
    """Respuesta de la predicción principal (modelo de clusters)."""
    predicted_cluster: int                  = Field(..., description="Cluster predicho (0,1,2)")
    cluster_name:      str                  = Field(..., description="Nombre del perfil clínico")
    description:       str                  = Field(..., description="Interpretación y recomendaciones")
    probabilities:     List[ClusterProb]    = Field(..., description="Probabilidad por cluster")
    riesgo_comparativo: Optional[RiesgoComparativo] = Field(
        None, description="Resultado de Framingham/SCC si se calculó"
    )


class CampoFaltante(BaseModel):
    campo:       str
    descripcion: str


class DatosPaciente(BaseModel):
    """Campos extraídos en la carga de archivos (22 variables)"""
    creatinina:      Optional[float] = None
    celulas_medias:  Optional[float] = None
    glucosa:         Optional[float] = None
    granulocitos:    Optional[float] = None
    hdl:             Optional[float] = None
    hematocrito:     Optional[float] = None
    hemoglobina:     Optional[float] = None
    ldl:             Optional[float] = None
    leucocitos:      Optional[float] = None
    linfocitos:      Optional[float] = None
    plaquetas:       Optional[float] = None
    trigliceridos:   Optional[float] = None
    edad:            Optional[int]   = None
    sexo:            Optional[int]   = None
    zona:            Optional[int]   = None
    ap_hipertension: Optional[int]   = None
    ta_sistolica:    Optional[float] = None
    ta_diastolica:   Optional[float] = None
    peso:            Optional[float] = None
    talla:           Optional[float] = None
    imc:             Optional[float] = None
    TFG:             Optional[float] = None
    # extras Framingham
    colesterol_total_mgdl: Optional[float] = None
    diabetes: Optional[int] = None
    tratamiento_antihipertensivo: Optional[int] = None
    fuma: Optional[int] = None


class UploadOutput(BaseModel):
    campos_faltantes: List[CampoFaltante] = Field(default_factory=list)
    prediccion:       Optional[PredictionOutput] = None
    mensaje:          str = ""
    datos_paciente:   Optional[DatosPaciente] = None
    framingham_faltante: Optional[List[CampoFaltante]] = None