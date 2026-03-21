from pydantic import BaseModel, Field, model_validator


class CardiovascularInput(BaseModel):
    age_days:    int   = Field(..., ge=10000, le=40000, description="Edad en días")
    gender:      int   = Field(..., ge=1, le=2,         description="1 = mujer · 2 = hombre")
    height:      int   = Field(..., ge=140, le=220,     description="Altura en cm")
    weight:      float = Field(..., ge=30, le=180,      description="Peso en kg")
    ap_hi:       int   = Field(..., ge=60, le=250,      description="Presión sistólica (mmHg)")
    ap_lo:       int   = Field(..., ge=40, le=200,      description="Presión diastólica (mmHg)")
    cholesterol: int   = Field(..., ge=1, le=3,         description="1 normal · 2 alto · 3 muy alto")
    gluc:        int   = Field(..., ge=1, le=3,         description="1 normal · 2 alto · 3 muy alto")
    smoke:       int   = Field(..., ge=0, le=1,         description="0 no fuma · 1 fuma")
    alco:        int   = Field(..., ge=0, le=1,         description="0 no consume · 1 consume alcohol")
    active:      int   = Field(..., ge=0, le=1,         description="0 no activo · 1 activo físicamente")

    @model_validator(mode="after")
    def validar_presion(self) -> "CardiovascularInput":
        """La presión diastólica siempre debe ser menor que la sistólica."""
        if self.ap_lo >= self.ap_hi:
            raise ValueError(
                f"La presión diastólica ({self.ap_lo}) debe ser menor "
                f"que la sistólica ({self.ap_hi})."
            )
        return self
