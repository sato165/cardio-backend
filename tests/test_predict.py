"""
test_predict.py
Pruebas de integración para POST /api/predict

Verifica el flujo completo: validación de entrada → preprocesamiento →
modelo → explicabilidad → respuesta estructurada.
"""

import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# ── Datos de prueba ───────────────────────────────────────────────────────────

PACIENTE_ALTO_RIESGO = {
    "age_days": 21061,
    "gender": 2,
    "height": 172,
    "weight": 88.5,
    "ap_hi": 145,
    "ap_lo": 90,
    "cholesterol": 2,
    "gluc": 2,
    "smoke": 0,
    "alco": 0,
    "active": 1,
}

PACIENTE_BAJO_RIESGO = {
    "age_days": 12000,
    "gender": 1,
    "height": 165,
    "weight": 60.0,
    "ap_hi": 110,
    "ap_lo": 70,
    "cholesterol": 1,
    "gluc": 1,
    "smoke": 0,
    "alco": 0,
    "active": 1,
}


# ── Respuesta exitosa ─────────────────────────────────────────────────────────

class TestPredictExitoso:

    def test_status_200(self):
        r = client.post("/api/predict/", json=PACIENTE_ALTO_RIESGO)
        assert r.status_code == 200

    def test_estructura_respuesta(self):
        r = client.post("/api/predict/", json=PACIENTE_ALTO_RIESGO)
        data = r.json()
        assert "riesgo_cardiovascular" in data
        assert "probabilidad" in data
        assert "nivel_riesgo" in data
        assert "explicabilidad" in data

    def test_riesgo_binario(self):
        r = client.post("/api/predict/", json=PACIENTE_ALTO_RIESGO)
        assert r.json()["riesgo_cardiovascular"] in (0, 1)

    def test_probabilidad_entre_0_y_1(self):
        r = client.post("/api/predict/", json=PACIENTE_ALTO_RIESGO)
        prob = r.json()["probabilidad"]
        assert 0.0 <= prob <= 1.0

    def test_nivel_riesgo_valido(self):
        r = client.post("/api/predict/", json=PACIENTE_ALTO_RIESGO)
        assert r.json()["nivel_riesgo"] in ("Alto", "Moderado", "Bajo")

    def test_explicabilidad_tiene_16_factores(self):
        r = client.post("/api/predict/", json=PACIENTE_ALTO_RIESGO)
        assert len(r.json()["explicabilidad"]) == 16

    def test_estructura_factor_explicacion(self):
        r = client.post("/api/predict/", json=PACIENTE_ALTO_RIESGO)
        factor = r.json()["explicabilidad"][0]
        assert "factor" in factor
        assert "impacto" in factor
        assert "descripcion" in factor
        assert "nivel" in factor

    def test_nivel_factor_valido(self):
        r = client.post("/api/predict/", json=PACIENTE_ALTO_RIESGO)
        for factor in r.json()["explicabilidad"]:
            assert factor["nivel"] in ("crítico", "moderado", "leve")

    def test_explicabilidad_ordenada_por_impacto(self):
        r = client.post("/api/predict/", json=PACIENTE_ALTO_RIESGO)
        factores = r.json()["explicabilidad"]
        impactos = [abs(f["impacto"]) for f in factores]
        assert impactos == sorted(impactos, reverse=True)

    def test_smoke_tiene_advertencia(self):
        r = client.post("/api/predict/", json=PACIENTE_ALTO_RIESGO)
        factores = r.json()["explicabilidad"]
        smoke = next((f for f in factores if "Tabaquismo" in f["factor"]), None)
        assert smoke is not None
        assert smoke["advertencia"] is not None
        assert "subregistrado" in smoke["advertencia"]

    def test_alco_tiene_advertencia(self):
        r = client.post("/api/predict/", json=PACIENTE_ALTO_RIESGO)
        factores = r.json()["explicabilidad"]
        alco = next((f for f in factores if "alcohol" in f["factor"].lower()), None)
        assert alco is not None
        assert alco["advertencia"] is not None

    def test_paciente_bajo_riesgo_retorna_bajo(self):
        r = client.post("/api/predict/", json=PACIENTE_BAJO_RIESGO)
        assert r.status_code == 200
        assert r.json()["nivel_riesgo"] in ("Bajo", "Moderado")


# ── Validaciones de entrada ───────────────────────────────────────────────────

class TestPredictValidaciones:

    def test_ap_hi_fuera_de_rango(self):
        datos = {**PACIENTE_ALTO_RIESGO, "ap_hi": 300}
        r = client.post("/api/predict/", json=datos)
        assert r.status_code == 422
        assert r.json()["error"] == "Datos de entrada inválidos"

    def test_ap_lo_fuera_de_rango(self):
        datos = {**PACIENTE_ALTO_RIESGO, "ap_lo": 5}
        r = client.post("/api/predict/", json=datos)
        assert r.status_code == 422

    def test_ap_lo_mayor_que_ap_hi(self):
        datos = {**PACIENTE_ALTO_RIESGO, "ap_hi": 100, "ap_lo": 110}
        r = client.post("/api/predict/", json=datos)
        assert r.status_code == 422

    def test_ap_lo_igual_a_ap_hi(self):
        datos = {**PACIENTE_ALTO_RIESGO, "ap_hi": 120, "ap_lo": 120}
        r = client.post("/api/predict/", json=datos)
        assert r.status_code == 422

    def test_gender_invalido(self):
        datos = {**PACIENTE_ALTO_RIESGO, "gender": 3}
        r = client.post("/api/predict/", json=datos)
        assert r.status_code == 422

    def test_cholesterol_invalido(self):
        datos = {**PACIENTE_ALTO_RIESGO, "cholesterol": 0}
        r = client.post("/api/predict/", json=datos)
        assert r.status_code == 422

    def test_campo_faltante(self):
        datos = {k: v for k, v in PACIENTE_ALTO_RIESGO.items() if k != "ap_lo"}
        r = client.post("/api/predict/", json=datos)
        assert r.status_code == 422

    def test_body_vacio(self):
        r = client.post("/api/predict/", json={})
        assert r.status_code == 422

    def test_error_tiene_campo_detalle(self):
        datos = {**PACIENTE_ALTO_RIESGO, "ap_hi": 300}
        r = client.post("/api/predict/", json=datos)
        data = r.json()
        assert "detalle" in data
        assert isinstance(data["detalle"], list)
        assert data["detalle"][0]["campo"] == "ap_hi"
