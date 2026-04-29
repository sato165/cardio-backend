"""
framingham_calculator.py
Implementa la ecuación de Framingham D'Agostino 2008.

Referencia: D'Agostino RB Sr, Vasan RS, Pencina MJ, et al.
"General Cardiovascular Risk Profile for Use in Primary Care."
Circulation. 2008;117:743-753. Table 2.

Calcula el riesgo de desarrollar cualquier evento cardiovascular
(coronario, cerebrovascular, insuficiencia cardíaca, enfermedad
arterial periférica) en los próximos 10 años.

Variables requeridas:
  - edad (años): 30–74
  - sexo: 'M' (hombre) o 'F' (mujer)
  - colesterol_total (mg/dL)
  - hdl (mg/dL)
  - presion_sistolica (mmHg)
  - tratamiento_antihipertensivo (bool)
  - fuma (bool)
  - diabetes (bool)
"""

import math
from typing import Optional


# ── Coeficientes D'Agostino 2008 — Tabla 2 ───────────────────────────────────

# Hombres
_COEF_H = {
    'ln_age':    3.06117,
    'ln_chol':   1.12370,
    'ln_hdl':   -0.93263,
    'ln_sbp_tx': 1.93303,   # presión sistólica con tratamiento
    'ln_sbp_no': 1.99881,   # presión sistólica sin tratamiento
    'smoke':     0.65451,
    'diabetes':  0.57367,
    'G':        23.9802,
    'S0':        0.88936,
}

# Mujeres — coeficiente de edad validado con caso de referencia del paper
_COEF_M = {
    'ln_age':    2.32888,
    'ln_chol':   1.20904,
    'ln_hdl':   -0.70833,
    'ln_sbp_tx': 2.82263,
    'ln_sbp_no': 2.76157,
    'smoke':     0.52873,
    'diabetes':  0.69154,
    'G':        26.1931,
    'S0':        0.94833,
}


# ── Función principal ─────────────────────────────────────────────────────────

def calcular_framingham(
    edad: int,
    sexo: int,                        # 1 = mujer, 2 = hombre (mismo código que el dataset)
    colesterol_total: float,          # mg/dL
    hdl: float,                       # mg/dL
    presion_sistolica: int,           # mmHg
    tratamiento_antihipertensivo: bool,
    fuma: bool,
    diabetes: bool,
) -> dict:
    """
    Calcula el riesgo cardiovascular a 10 años según Framingham D'Agostino 2008.

    Retorna:
        porcentaje: float — riesgo a 10 años (0–100)
        nivel: str        — 'Bajo' | 'Moderado' | 'Alto'
        descripcion: str  — interpretación clínica
    """
    # Validar rango de edad del modelo
    if not (30 <= edad <= 74):
        return _resultado_no_aplicable(
            f"Framingham D'Agostino 2008 está validado para edades entre "
            f"30 y 74 años. La edad del paciente es {edad} años."
        )

    coef = _COEF_H if sexo == 2 else _COEF_M

    ln_sbp = math.log(presion_sistolica)
    coef_sbp = coef['ln_sbp_tx'] if tratamiento_antihipertensivo else coef['ln_sbp_no']

    L = (coef['ln_age']  * math.log(edad)
       + coef['ln_chol'] * math.log(colesterol_total)
       + coef['ln_hdl']  * math.log(hdl)
       + coef_sbp        * ln_sbp
       + coef['smoke']   * (1 if fuma else 0)
       + coef['diabetes']* (1 if diabetes else 0))

    B         = math.exp(L - coef['G'])
    porcentaje = round((1 - coef['S0'] ** B) * 100, 1)
    porcentaje = max(0.1, min(99.9, porcentaje))  # limitar al rango válido

    nivel, descripcion = _clasificar(porcentaje)

    return {
        'porcentaje':  porcentaje,
        'nivel':       nivel,
        'descripcion': descripcion,
        'aplicable':   True,
        'referencia':  "D'Agostino et al. Circulation 2008;117:743-753",
    }


def _clasificar(porcentaje: float) -> tuple[str, str]:
    """
    Umbrales estándar Framingham:
    < 10%  → Bajo
    10–20% → Moderado
    > 20%  → Alto
    """
    if porcentaje < 10:
        return (
            'Bajo',
            f"Riesgo cardiovascular bajo a 10 años ({porcentaje}%). "
            f"Probabilidad de evento cardiovascular menor al 10%."
        )
    elif porcentaje <= 20:
        return (
            'Moderado',
            f"Riesgo cardiovascular moderado a 10 años ({porcentaje}%). "
            f"Se recomienda control de factores de riesgo modificables."
        )
    return (
        'Alto',
        f"Riesgo cardiovascular alto a 10 años ({porcentaje}%). "
        f"Se recomienda evaluación clínica y manejo intensivo de factores de riesgo."
    )


def _resultado_no_aplicable(razon: str) -> dict:
    return {
        'porcentaje':  None,
        'nivel':       None,
        'descripcion': razon,
        'aplicable':   False,
        'referencia':  "D'Agostino et al. Circulation 2008;117:743-753",
    }


# ── Campos requeridos para mostrar en el frontend ────────────────────────────

CAMPOS_REQUERIDOS_FRAMINGHAM = {
    'colesterol_total_mgdl': 'Colesterol total (mg/dL)',
    'hdl_mgdl':              'Colesterol HDL (mg/dL)',
    'diabetes':              'Diabetes mellitus (sí / no)',
    'tratamiento_hta':       'Tratamiento antihipertensivo actual (sí / no)',
}


def campos_faltantes_framingham(datos: dict) -> list[dict]:
    """
    Recibe el dict de datos del paciente y retorna la lista de campos
    de Framingham que no están presentes o son None.
    """
    return [
        {'campo': k, 'descripcion': v}
        for k, v in CAMPOS_REQUERIDOS_FRAMINGHAM.items()
        if datos.get(k) is None
    ]