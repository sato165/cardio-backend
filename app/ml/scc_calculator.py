"""
scc_calculator.py
Implementa el ajuste de Framingham para población colombiana recomendado
por la Sociedad Colombiana de Cardiología (SCC).

Referencia:
  Muñoz OM, Rodríguez NI, Ruiz A, Rondón M.
  "Validación de los modelos de predicción de Framingham y PROCAM como
  estimadores del riesgo cardiovascular en una población colombiana."
  Rev Col Cardiol. 2014;21:202-210.

  Guía de práctica clínica para la prevención, detección temprana,
  diagnóstico, tratamiento y seguimiento de las dislipidemias en Colombia.
  Ministerio de Salud y Protección Social, 2014.

La SCC recomienda aplicar un factor de corrección de 0.75 al resultado
de Framingham para ajustar la sobreestimación del modelo en poblaciones
latinoamericanas de menor incidencia cardiovascular que la cohorte
original de Massachusetts.

  Riesgo SCC Colombia = Riesgo Framingham × 0.75
"""

from app.ml.framingham_calculator import calcular_framingham, campos_faltantes_framingham


FACTOR_AJUSTE_COLOMBIA = 0.75


def calcular_scc(
    edad: int,
    sexo: int,
    colesterol_total: float,
    hdl: float,
    presion_sistolica: int,
    tratamiento_antihipertensivo: bool,
    fuma: bool,
    diabetes: bool,
) -> dict:
    """
    Calcula el riesgo cardiovascular a 10 años ajustado para Colombia.
    Primero calcula Framingham y luego aplica el factor 0.75.

    Retorna:
        porcentaje_framingham: float — riesgo Framingham original
        porcentaje_scc: float        — riesgo ajustado Colombia
        nivel: str                   — 'Bajo' | 'Moderado' | 'Alto'
        descripcion: str             — interpretación clínica
        factor_ajuste: float         — 0.75 (documentado explícitamente)
    """
    framingham = calcular_framingham(
        edad=edad,
        sexo=sexo,
        colesterol_total=colesterol_total,
        hdl=hdl,
        presion_sistolica=presion_sistolica,
        tratamiento_antihipertensivo=tratamiento_antihipertensivo,
        fuma=fuma,
        diabetes=diabetes,
    )

    if not framingham['aplicable']:
        return {
            'porcentaje_framingham': None,
            'porcentaje_scc':        None,
            'nivel':                 None,
            'descripcion':           framingham['descripcion'],
            'factor_ajuste':         FACTOR_AJUSTE_COLOMBIA,
            'aplicable':             False,
            'referencia':            _referencia(),
        }

    pct_framingham = framingham['porcentaje']
    pct_scc        = round(pct_framingham * FACTOR_AJUSTE_COLOMBIA, 1)
    pct_scc        = max(0.1, min(99.9, pct_scc))

    nivel, descripcion = _clasificar(pct_scc)

    return {
        'porcentaje_framingham': pct_framingham,
        'porcentaje_scc':        pct_scc,
        'nivel':                 nivel,
        'descripcion':           descripcion,
        'factor_ajuste':         FACTOR_AJUSTE_COLOMBIA,
        'aplicable':             True,
        'referencia':            _referencia(),
    }


def _clasificar(porcentaje: float) -> tuple[str, str]:
    """
    Mismos umbrales que Framingham aplicados al valor ajustado.
    """
    if porcentaje < 10:
        return (
            'Bajo',
            f"Riesgo cardiovascular bajo ajustado para Colombia ({porcentaje}%). "
            f"Calculado con factor de corrección 0.75 sobre Framingham."
        )
    elif porcentaje <= 20:
        return (
            'Moderado',
            f"Riesgo cardiovascular moderado ajustado para Colombia ({porcentaje}%). "
            f"Se recomienda control de factores de riesgo modificables."
        )
    return (
        'Alto',
        f"Riesgo cardiovascular alto ajustado para Colombia ({porcentaje}%). "
        f"Se recomienda evaluación clínica y manejo intensivo de factores de riesgo."
    )


def _referencia() -> str:
    return (
        "Muñoz et al. Rev Col Cardiol 2014;21:202-210. "
        "Factor de ajuste 0.75 recomendado por la Guía colombiana de dislipidemias."
    )