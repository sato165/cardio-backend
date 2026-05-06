from pydantic import BaseModel, Field, model_validator
from typing import Optional


class PredictionInput(BaseModel):
    """22 campos obligatorios para el modelo de clusters (nuevo dataset real)"""

    # Laboratorio
    creatinina:      float = Field(..., ge=0, description="Creatinina sérica (mg/dL)")
    celulas_medias:  float = Field(..., ge=0, description="Volumen corpuscular medio (fL)")
    glucosa:         float = Field(..., ge=0, description="Glucosa en ayunas (mg/dL)")
    granulocitos:    float = Field(..., ge=0, le=100, description="Granulocitos (%)")
    hdl:             float = Field(..., ge=0, description="Colesterol HDL (mg/dL)")
    hematocrito:     float = Field(..., ge=0, description="Hematocrito (%)")
    hemoglobina:     float = Field(..., ge=0, description="Hemoglobina (g/dL)")
    ldl:             float = Field(..., ge=0, description="Colesterol LDL (mg/dL)")
    leucocitos:      float = Field(..., ge=0, description="Leucocitos (10³/µL)")
    linfocitos:      float = Field(..., ge=0, le=100, description="Linfocitos (%)")
    plaquetas:       float = Field(..., ge=0, description="Plaquetas (10³/µL)")
    trigliceridos:   float = Field(..., ge=0, description="Triglicéridos (mg/dL)")

    # Demográficos / antecedentes
    edad:            int   = Field(..., ge=6, le=110, description="Edad en años")
    sexo:            int   = Field(..., ge=0, le=1, description="0 = mujer, 1 = hombre")
    zona:            int   = Field(..., ge=0, le=1, description="0 = rural, 1 = urbana")
    ap_hipertension: int   = Field(..., ge=0, le=1, description="Antecedente personal de hipertensión")

    # Signos vitales y antropometría
    ta_sistolica:    float = Field(..., ge=60.5, le=220.0, description="Presión sistólica (mmHg)")
    ta_diastolica:   float = Field(..., ge=40.0, le=120.0, description="Presión diastólica (mmHg)")
    peso:            float = Field(..., ge=9.0, le=170.0, description="Peso (kg)")
    talla:           float = Field(..., ge=1.27, le=1.97, description="Talla (metros)")
    imc:             float = Field(..., ge=4.51, le=60.0, description="Índice de masa corporal (kg/m²)")
    TFG:             float = Field(..., ge=11.47, le=197.39, description="Tasa de filtración glomerular (mL/min/1.73m²)")

    # --- Opcionales para Framingham / SCC (se mantienen como extra) ---
    colesterol_total_mgdl: Optional[float] = Field(
        None, ge=50, le=500, description="Colesterol total (mg/dL) – para cálculo Framingham"
    )
    # hdl_mgdl ya está arriba como 'hdl', pero si se necesita por separado se puede ignorar
    diabetes: Optional[int] = Field(
        None, ge=0, le=1, description="0 no diabético, 1 diabético – para Framingham"
    )
    tratamiento_antihipertensivo: Optional[int] = Field(
        None, ge=0, le=1, description="¿Recibe tratamiento antihipertensivo? 0 no, 1 sí"
    )
    fuma: Optional[int] = Field(
        None, ge=0, le=1, description="¿Fuma actualmente? 0 no, 1 sí"
    )

    @model_validator(mode="after")
    def validar_presion(self) -> "PredictionInput":
        """La presión diastólica debe ser menor que la sistólica."""
        if self.ta_diastolica >= self.ta_sistolica:
            raise ValueError(
                f"Presión diastólica ({self.ta_diastolica}) debe ser menor que la sistólica ({self.ta_sistolica})"
            )
        return self