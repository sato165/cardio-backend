"""
test_upload.py
Pruebas de integración para POST /api/predict/upload y /api/predict/upload/pdf
"""

import json
import os
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

PDF_PATH = r"C:\Users\Sebastian Torres\AppData\Local\Temp\historia_clinica_ejemplo.pdf"


@pytest.fixture
def json_completo():
    return {
        "identificacion_paciente": {"edad_dias": 21061, "genero_codigo": 2},
        "datos_antropometricos":   {"altura_cm": 172, "peso_kg": 88.5},
        "signos_vitales":          {"presion_sistolica_mmhg": 145, "presion_diastolica_mmhg": 90},
        "examenes_laboratorio":    {"colesterol_codigo_modelo": 2, "glucosa_codigo_modelo": 2},
        "habitos_vida":            {"fuma_actualmente": False, "consume_alcohol": False, "actividad_fisica": True},
    }


@pytest.fixture
def json_ap_lo_faltante():
    return {
        "identificacion_paciente": {"edad_dias": 21061, "genero_codigo": 2},
        "datos_antropometricos":   {"altura_cm": 172, "peso_kg": 88.5},
        "signos_vitales":          {"presion_sistolica_mmhg": 145, "presion_diastolica_mmhg": None},
        "examenes_laboratorio":    {"colesterol_codigo_modelo": 2, "glucosa_codigo_modelo": 2},
        "habitos_vida":            {"fuma_actualmente": False, "consume_alcohol": False, "actividad_fisica": True},
    }


# ── JSON exitoso ──────────────────────────────────────────────────────────────

class TestUploadJsonExitoso:

    def test_status_200(self, json_completo):
        r = client.post(
            "/api/predict/upload",
            files={"archivo": ("hc.json", json.dumps(json_completo), "application/json")},
        )
        assert r.status_code == 200

    def test_prediccion_presente_cuando_completo(self, json_completo):
        r = client.post(
            "/api/predict/upload",
            files={"archivo": ("hc.json", json.dumps(json_completo), "application/json")},
        )
        data = r.json()
        assert data["prediccion"] is not None
        assert data["campos_faltantes"] == []

    def test_estructura_prediccion(self, json_completo):
        r = client.post(
            "/api/predict/upload",
            files={"archivo": ("hc.json", json.dumps(json_completo), "application/json")},
        )
        pred = r.json()["prediccion"]
        assert "riesgo_cardiovascular" in pred
        assert "probabilidad" in pred
        assert "nivel_riesgo" in pred
        assert "explicabilidad" in pred

    def test_mensaje_exitoso(self, json_completo):
        r = client.post(
            "/api/predict/upload",
            files={"archivo": ("hc.json", json.dumps(json_completo), "application/json")},
        )
        assert "exitosamente" in r.json()["mensaje"].lower()


# ── JSON campos faltantes ─────────────────────────────────────────────────────

class TestUploadJsonFaltantes:

    def test_detecta_ap_lo_faltante(self, json_ap_lo_faltante):
        r = client.post(
            "/api/predict/upload",
            files={"archivo": ("hc.json", json.dumps(json_ap_lo_faltante), "application/json")},
        )
        assert r.status_code == 200
        campos = [f["campo"] for f in r.json()["campos_faltantes"]]
        assert "ap_lo" in campos

    def test_prediccion_null_cuando_falta_campo(self, json_ap_lo_faltante):
        r = client.post(
            "/api/predict/upload",
            files={"archivo": ("hc.json", json.dumps(json_ap_lo_faltante), "application/json")},
        )
        assert r.json()["prediccion"] is None

    def test_mensaje_indica_campo_faltante(self, json_ap_lo_faltante):
        r = client.post(
            "/api/predict/upload",
            files={"archivo": ("hc.json", json.dumps(json_ap_lo_faltante), "application/json")},
        )
        assert "ap_lo" in r.json()["mensaje"]

    def test_estructura_campo_faltante(self, json_ap_lo_faltante):
        r = client.post(
            "/api/predict/upload",
            files={"archivo": ("hc.json", json.dumps(json_ap_lo_faltante), "application/json")},
        )
        faltante = r.json()["campos_faltantes"][0]
        assert "campo" in faltante
        assert "descripcion" in faltante


# ── JSON errores de formato ───────────────────────────────────────────────────

class TestUploadJsonErrores:

    def test_json_malformado(self):
        r = client.post(
            "/api/predict/upload",
            files={"archivo": ("hc.json", b"esto no es json {{{", "application/json")},
        )
        assert r.status_code == 422

    def test_extension_incorrecta(self, json_completo):
        r = client.post(
            "/api/predict/upload",
            files={"archivo": ("hc.txt", json.dumps(json_completo), "text/plain")},
        )
        assert r.status_code == 422


# ── PDF ───────────────────────────────────────────────────────────────────────

class TestUploadPdf:

    def test_pdf_retorna_200(self):
        if not os.path.exists(PDF_PATH):
            pytest.skip("PDF de prueba no disponible")
        with open(PDF_PATH, "rb") as f:
            r = client.post(
                "/api/predict/upload/pdf",
                files={"archivos": ("historia.pdf", f, "application/pdf")},
            )
        assert r.status_code == 200

    def test_pdf_detecta_ap_lo_faltante(self):
        if not os.path.exists(PDF_PATH):
            pytest.skip("PDF de prueba no disponible")
        with open(PDF_PATH, "rb") as f:
            r = client.post(
                "/api/predict/upload/pdf",
                files={"archivos": ("historia.pdf", f, "application/pdf")},
            )
        data = r.json()
        campos = [f["campo"] for f in data["campos_faltantes"]]
        assert "ap_lo" in campos
        assert data["prediccion"] is None

    def test_extension_incorrecta_pdf(self, json_completo):
        r = client.post(
            "/api/predict/upload/pdf",
            files={"archivos": ("hc.json", json.dumps(json_completo), "application/json")},
        )
        assert r.status_code == 422

    def test_mas_de_5_pdfs_rechazado(self):
        if not os.path.exists(PDF_PATH):
            pytest.skip("PDF de prueba no disponible")
        files = []
        with open(PDF_PATH, "rb") as f:
            contenido = f.read()
        for i in range(6):
            files.append(("archivos", (f"p{i}.pdf", contenido, "application/pdf")))
        r = client.post("/api/predict/upload/pdf", files=files)
        assert r.status_code == 422
        assert "máximo 5" in r.json()["detail"]


# ── Health check ──────────────────────────────────────────────────────────────

def test_health_check():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert "model_activo" in r.json()
