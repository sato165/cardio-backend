import pytest
from app.ml.framingham_calculator import calcular_framingham, campos_faltantes_framingham
from app.ml.scc_calculator import calcular_scc


class TestFramingham:
    def test_hombre_bajo_riesgo(self):
        # 45 años, col 180, HDL 55, PAS 120, no fuma, no diabetes, sin tto.
        result = calcular_framingham(
            edad=45, sexo=2,
            colesterol_total=180, hdl=55,
            presion_sistolica=120,
            tratamiento_antihipertensivo=False,
            fuma=False, diabetes=False
        )
        assert result["porcentaje"] is not None
        # con los coeficientes corregidos debe dar alrededor de 3-6%
        assert 3 <= result["porcentaje"] <= 6
        assert result["nivel"] == "Bajo"

    def test_mujer_alto_riesgo(self):
        result = calcular_framingham(
            edad=65, sexo=1,
            colesterol_total=260, hdl=40,
            presion_sistolica=160,
            tratamiento_antihipertensivo=True,
            fuma=True, diabetes=True
        )
        assert result["porcentaje"] > 20
        assert result["nivel"] == "Alto"

    def test_scc_ajuste(self):
        fram = calcular_framingham(
            edad=55, sexo=2,
            colesterol_total=200, hdl=45,
            presion_sistolica=140,
            tratamiento_antihipertensivo=False,
            fuma=False, diabetes=False
        )
        scc = calcular_scc(
            edad=55, sexo=2,
            colesterol_total=200, hdl=45,
            presion_sistolica=140,
            tratamiento_antihipertensivo=False,
            fuma=False, diabetes=False
        )
        assert scc["porcentaje_scc"] == round(fram["porcentaje"] * 0.75, 1)

    def test_datos_faltantes(self):
        datos = {
            "age_days": 15000, "gender": 1,
            "ap_hi": 130, "smoke": 0
            # sin colesterol_total_mgdl ni hdl_mgdl
        }
        faltantes = campos_faltantes_framingham(datos)
        assert len(faltantes) >= 2
        nombres = [f["campo"] for f in faltantes]
        assert "colesterol_total_mgdl" in nombres
        assert "hdl_mgdl" in nombres