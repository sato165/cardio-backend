"""
json_extractor.py
Extrae los campos del modelo desde una historia clínica en formato JSON.

El JSON puede venir de cualquier sistema HIS con estructura variable.
El extractor busca los campos por múltiples nombres posibles y por rutas
anidadas. El resultado tiene el mismo formato que pdf_extractor.py.
"""

from typing import Any


# ---------------------------------------------------------------------------
# Campos requeridos y sus descripciones para el médico
# ---------------------------------------------------------------------------

CAMPOS_REQUERIDOS = {
    "age_days":    "Edad (en años o días)",
    "gender":      "Género (masculino / femenino)",
    "height":      "Altura (cm)",
    "weight":      "Peso (kg)",
    "ap_hi":       "Presión sistólica (mmHg)",
    "ap_lo":       "Presión diastólica (mmHg)",
    "cholesterol": "Colesterol (normal / alto / muy alto)",
    "gluc":        "Glucosa (normal / alta / muy alta)",
    "smoke":       "Tabaquismo (sí / no)",
    "alco":        "Consumo de alcohol (sí / no)",
    "active":      "Actividad física (sí / no)",
}

# Nombres alternativos por los que puede venir cada campo en el JSON
_ALIAS: dict[str, list[str]] = {
    "age_days":    ["age_days", "edad_dias", "edad_en_dias", "age"],
    "gender":      ["gender", "genero_codigo", "gender_code", "sexo_codigo"],
    "height":      ["height", "height_cm", "altura_cm", "talla_cm"],
    "weight":      ["weight", "weight_kg", "peso_kg"],
    "ap_hi":       ["ap_hi", "presion_sistolica_mmhg", "systolic", "sistolica"],
    "ap_lo":       ["ap_lo", "presion_diastolica_mmhg", "diastolic", "diastolica"],
    "cholesterol": ["cholesterol", "colesterol_codigo_modelo", "cholesterol_code"],
    "gluc":        ["gluc", "glucosa_codigo_modelo", "glucose_code"],
    "smoke":       ["smoke", "fuma_actualmente", "smoking", "tabaquismo"],
    "alco":        ["alco", "consume_alcohol", "alcohol", "drinking"],
    "active":      ["active", "actividad_fisica", "physically_active", "ejercicio"],
}

# Rutas anidadas donde suelen vivir los campos en JSONs de sistemas HIS
_RUTAS_ANIDADAS = [
    "campos_modelo_ia",
    "identificacion_paciente",
    "signos_vitales",
    "datos_antropometricos",
    "examenes_laboratorio",
    "habitos_vida",
]


# ---------------------------------------------------------------------------
# Punto de entrada público
# ---------------------------------------------------------------------------

def extraer_de_json(datos: dict) -> dict[str, Any]:
    """
    Recibe el dict del JSON de historia clínica y retorna los campos
    del modelo. Los campos no encontrados quedan en None.
    """
    campos: dict[str, Any] = {campo: None for campo in CAMPOS_REQUERIDOS}

    # 1. Buscar en la raíz y en las secciones anidadas conocidas
    fuentes = [datos] + [
        datos[seccion]
        for seccion in _RUTAS_ANIDADAS
        if isinstance(datos.get(seccion), dict)
    ]

    for campo, alias_list in _ALIAS.items():
        for fuente in fuentes:
            valor = _buscar_alias(fuente, alias_list)
            if valor is not None:
                campos[campo] = _normalizar(campo, valor)
                break

    # 2. Intentar derivar edad_days desde fecha_nacimiento si no se encontró
    if campos["age_days"] is None:
        campos["age_days"] = _derivar_edad_desde_fecha(datos)

    # 3. Listar campos faltantes
    campos["campos_faltantes"] = [
        {"campo": k, "descripcion": CAMPOS_REQUERIDOS[k]}
        for k in CAMPOS_REQUERIDOS
        if campos.get(k) is None
    ]

    return campos


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _buscar_alias(fuente: dict, alias_list: list[str]) -> Any:
    """Busca el primer alias que exista en el dict y tenga valor no nulo."""
    for alias in alias_list:
        if alias in fuente and fuente[alias] is not None:
            return fuente[alias]
    return None


def _normalizar(campo: str, valor: Any) -> Any:
    """
    Convierte el valor al tipo esperado por el modelo.
    Los campos binarios aceptan bool, string o int.
    """
    binarios = {"smoke", "alco", "active"}

    if campo in binarios:
        if isinstance(valor, bool):
            return 1 if valor else 0
        if isinstance(valor, str):
            return 1 if valor.lower() in {"true", "sí", "si", "yes", "1", "activo"} else 0
        return int(bool(valor))

    if campo in {"age_days", "gender", "height", "cholesterol", "gluc"}:
        return int(valor) if valor is not None else None

    if campo in {"weight"}:
        return float(valor) if valor is not None else None

    if campo in {"ap_hi", "ap_lo"}:
        return int(valor) if valor is not None else None

    return valor


def _derivar_edad_desde_fecha(datos: dict) -> int | None:
    """
    Intenta calcular age_days desde fecha_nacimiento si está presente
    en el JSON (formato ISO: YYYY-MM-DD).
    """
    from datetime import date

    claves_fecha = ["fecha_nacimiento", "birth_date", "date_of_birth", "dob"]
    fuentes = [datos] + [
        datos[s] for s in _RUTAS_ANIDADAS
        if isinstance(datos.get(s), dict)
    ]

    for fuente in fuentes:
        for clave in claves_fecha:
            if clave in fuente and fuente[clave]:
                try:
                    nacimiento = date.fromisoformat(str(fuente[clave]))
                    dias = (date.today() - nacimiento).days
                    if 6000 <= dias <= 40000:  # rango razonable 16-110 años
                        return dias
                except (ValueError, TypeError):
                    continue
    return None
