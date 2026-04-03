import joblib
from pathlib import Path
from app.core.config import settings

# El modelo se carga una sola vez cuando el servidor arranca.
# Todas las peticiones comparten esta misma instancia en memoria.
_model = None


def get_model():
    """Retorna el modelo activo. Lo carga la primera vez que se llama."""
    global _model
    if _model is None:
        _model = _load_model()
    return _model


def _load_model():
    """Selecciona y carga el .pkl según BEST_MODEL en .env."""
    rutas = {
        "random_forest": settings.model_rf_path,
        "xgboost":       settings.model_xgb_path,
    }

    nombre = settings.best_model.lower()
    if nombre not in rutas:
        raise ValueError(
            f"BEST_MODEL='{nombre}' no es válido. "
            f"Opciones: {list(rutas.keys())}"
        )

    ruta = Path(rutas[nombre])
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en '{ruta}'. "
            f"Asegúrate de copiar los .pkl a la carpeta models/."
        )

    model = joblib.load(ruta)
    print(f"✓ Modelo cargado: {nombre} ({ruta})")
    return model