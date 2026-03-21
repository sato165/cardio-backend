"""
test_preprocessing.py
Pruebas unitarias para app/ml/preprocessing.py
"""

import pytest
import pandas as pd
from app.schemas.input_schema import CardiovascularInput
from app.ml.preprocessing import (
    preparar_features,
    FEATURE_ORDER,
    _calcular_age_range,
    _calcular_bp_category,
    _calcular_metabolic_risk,
)


@pytest.fixture
def datos_validos():
    return CardiovascularInput(
        age_days=21061, gender=2, height=172, weight=88.5,
        ap_hi=145, ap_lo=90, cholesterol=2, gluc=2,
        smoke=0, alco=0, active=1,
    )


class TestPrepararFeatures:

    def test_retorna_dataframe(self, datos_validos):
        assert isinstance(preparar_features(datos_validos), pd.DataFrame)

    def test_una_sola_fila(self, datos_validos):
        assert len(preparar_features(datos_validos)) == 1

    def test_columnas_en_orden_correcto(self, datos_validos):
        assert list(preparar_features(datos_validos).columns) == FEATURE_ORDER

    def test_sin_valores_nulos(self, datos_validos):
        assert preparar_features(datos_validos).isnull().sum().sum() == 0

    def test_edad_convertida_a_anos(self, datos_validos):
        resultado = preparar_features(datos_validos)
        assert resultado["age"].iloc[0] == round(21061 / 365.25, 1)

    def test_bmi_calculado_correctamente(self, datos_validos):
        resultado = preparar_features(datos_validos)
        assert resultado["bmi"].iloc[0] == round(88.5 / (172 / 100) ** 2, 1)

    def test_pulse_pressure_calculado(self, datos_validos):
        assert preparar_features(datos_validos)["pulse_pressure"].iloc[0] == 55

    def test_valores_originales_preservados(self, datos_validos):
        resultado = preparar_features(datos_validos)
        assert resultado["ap_hi"].iloc[0] == 145
        assert resultado["ap_lo"].iloc[0] == 90
        assert resultado["gender"].iloc[0] == 2
        assert resultado["cholesterol"].iloc[0] == 2


class TestCalcularAgeRange:

    def test_menor_40(self):
        assert _calcular_age_range(35.0) == 1

    def test_limite_inferior_40(self):
        assert _calcular_age_range(40.0) == 2

    def test_rango_40_49(self):
        assert _calcular_age_range(45.5) == 2

    def test_limite_inferior_50(self):
        assert _calcular_age_range(50.0) == 3

    def test_rango_50_59(self):
        assert _calcular_age_range(57.7) == 3

    def test_limite_inferior_60(self):
        assert _calcular_age_range(60.0) == 4

    def test_mayor_60(self):
        assert _calcular_age_range(64.9) == 4


class TestCalcularBpCategory:

    def test_normal(self):
        assert _calcular_bp_category(115, 75) == 1

    def test_elevada(self):
        assert _calcular_bp_category(125, 75) == 2

    def test_hta_grado_1_por_sistolica(self):
        assert _calcular_bp_category(135, 85) == 3

    def test_hta_grado_1_por_diastolica(self):
        # ap_hi < 140 (entra en elif) → grado 1 aunque ap_lo >= 90
        assert _calcular_bp_category(125, 85) == 3

    def test_hta_grado_2_por_sistolica(self):
        # ap_hi >= 140 → ningún elif aplica → grado 2
        assert _calcular_bp_category(145, 90) == 4

    def test_hta_grado_2_ambas_elevadas(self):
        # ap_hi=142 >= 140 y ap_lo=95 >= 90 → grado 2
        assert _calcular_bp_category(142, 95) == 4

    def test_hta_grado_1_sistolica_130_139_diastolica_alta(self):
        # ap_hi=130 < 140 → entra en elif → grado 1 (clasificación AHA)
        assert _calcular_bp_category(130, 95) == 3

    def test_paciente_ejemplo(self):
        # Carlos Andres: ap_hi=145, ap_lo=90 → HTA grado 2
        assert _calcular_bp_category(145, 90) == 4


class TestCalcularMetabolicRisk:

    def test_score_cero(self):
        assert _calcular_metabolic_risk(1, 1, 22.0) == 0

    def test_score_uno_colesterol(self):
        assert _calcular_metabolic_risk(2, 1, 22.0) == 1

    def test_score_uno_gluc(self):
        assert _calcular_metabolic_risk(1, 2, 22.0) == 1

    def test_score_uno_obesidad(self):
        assert _calcular_metabolic_risk(1, 1, 31.0) == 1

    def test_score_dos(self):
        assert _calcular_metabolic_risk(2, 2, 22.0) == 2

    def test_score_tres(self):
        assert _calcular_metabolic_risk(2, 2, 31.0) == 3

    def test_paciente_ejemplo(self):
        bmi = round(88.5 / (172 / 100) ** 2, 1)
        assert _calcular_metabolic_risk(2, 2, bmi) == 2
