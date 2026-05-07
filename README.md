# CardioPredict API

Backend del sistema de predicción de riesgo cardiovascular con perfilamiento clínico mediante clustering e inteligencia artificial para médicos.

Desarrollado por Sebastián Torres Ortega, Mayerlis Acosta Peralta y Christian Rivera Dibasto como proyecto integrador de Ingeniería de Sistemas e Ingeniería Biomédica.

---

## ¿Qué hace este sistema?

Recibe datos de un paciente (manualmente o desde una historia clínica en JSON o PDF), los procesa mediante un pipeline de clustering (winsorización → StandardScaler → PCA de 11 componentes → Random Forest), asigna al paciente a uno de tres perfiles clínicos — **Cardio‑renal**, **Cardiovascular Inflamatorio** o **Bajo Riesgo** — y retorna las probabilidades de pertenencia a cada perfil junto con una explicación clínica. Opcionalmente calcula los índices de riesgo **Framingham 2008** y el ajuste de la **Sociedad Colombiana de Cardiología (SCC)** si se proporcionan datos adicionales.

---

## Stack tecnológico

| Capa | Tecnología |
|---|---|
| Framework web | FastAPI + Uvicorn |
| Modelo principal | Random Forest con clustering PCA |
| Explicabilidad | SHAP TreeExplainer |
| Validación de datos | Pydantic v2 |
| Extracción PDF | PyMuPDF · pdfplumber · pytesseract |
| Empaquetado escritorio | PyInstaller 6.x |
| Pruebas | pytest + httpx |
| Entrenamiento | Google Colab (notebooks en `notebooks/`) |

---

## Estructura del proyecto

```
cardio-backend/
├── app/
│   ├── api/
│   │   ├── router.py
│   │   └── routes/
│   │       ├── explain.py
│   │       ├── predict.py
│   │       └── upload.py
│   ├── core/
│   │   └── config.py
│   ├── ml/
│   │   ├── explainability.py
│   │   ├── framingham_calculator.py
│   │   ├── model_loader.py
│   │   ├── predictor.py
│   │   ├── preprocessing.py
│   │   └── scc_calculator.py
│   ├── schemas/
│   │   ├── input_schema.py
│   │   └── output_schema.py
│   └── services/
│       ├── json_extractor.py
│       ├── pdf_extractor.py
│       └── prediction_service.py
├── data/
│   ├── content/
│   ├── processed/
│   └── raw/
├── dist/
│   └── CardioPredictor.exe        ← ejecutable de escritorio generado
├── frontend_dist/                 ← build del frontend (copiado desde cardio-frontend/dist/)
├── models/
│   ├── kmeans.pkl
│   ├── pca.pkl
│   ├── random_forest.pkl
│   ├── scaler.pkl
│   └── xgboost_model.pkl
├── notebooks/
├── tests/
├── cardio_app.py                  ← punto de entrada para el ejecutable
├── hook-numpy.py                  ← hook personalizado de PyInstaller para numpy
├── main.py
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Instalación (modo desarrollo)

### Requisitos previos

- Python 3.12
- Git

### Pasos

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd cardio-backend

# 2. Crear y activar entorno virtual
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / macOS

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
copy .env.example .env       # Windows
cp .env.example .env         # Linux / macOS

# 5. Verificar que los artefactos estén en models/
# models/random_forest.pkl · models/scaler.pkl · models/pca.pkl
# models/kmeans.pkl · models/xgboost_model.pkl

# 6. Arrancar el servidor
uvicorn main:app --reload
```

El servidor queda disponible en `http://localhost:8000`.  
La documentación interactiva Swagger está en `http://localhost:8000/docs`.

---

## Variables de entorno

Copiar `.env.example` como `.env` y completar:

```env
MODEL_PATH=models/random_forest.pkl
SCALER_PATH=models/scaler.pkl
PCA_PATH=models/pca.pkl
DEBUG=True
ALLOWED_ORIGINS=http://localhost:5173
MAX_UPLOAD_SIZE_MB=5
```

---

## Endpoints

### `POST /api/predict`
Predicción desde formulario manual con 22 variables clínicas.

### `POST /api/upload`
Predicción desde historia clínica en JSON. Detecta campos faltantes automáticamente.

### `POST /api/predict/explain`
Retorna las probabilidades por perfil clínico para la explicabilidad.

### `GET /api/health`
Verifica que el servidor y los artefactos estén cargados.

```json
{ "status": "ok", "modelo": "RandomForest con clustering PCA" }
```

---

## Pruebas

```bash
pytest tests/ -v
```

---

## Despliegue como aplicación de escritorio (Windows)

El sistema puede empaquetarse como un único ejecutable `.exe` que levanta el servidor FastAPI y abre el frontend automáticamente en el navegador. No requiere Python instalado en la máquina del usuario.

### Cómo funciona

`cardio_app.py` es el punto de entrada del ejecutable. Arranca Uvicorn en un hilo separado, espera 2 segundos y abre `http://127.0.0.1:8000` en el navegador por defecto. El frontend React (compilado) se sirve como archivos estáticos desde `frontend_dist/` mediante `FastAPI.mount`. Las rutas de los artefactos `.pkl` se resuelven automáticamente usando `sys._MEIPASS` dentro del ejecutable.

### Preparar el frontend antes del build

El build del frontend debe copiarse a `frontend_dist/` antes de empaquetar:

```bash
# Desde la raíz del proyecto (cardio-backend/)
xcopy ..\cardio-frontend\dist frontend_dist /E /I /Y
```

### Parche requerido en scipy

PyInstaller 6.x con scipy 1.11.x presenta un bug conocido (`NameError: name 'obj' is not defined`) en `scipy/stats/_distn_infrastructure.py`. Antes de cualquier build, aplicar este parche en el archivo del venv:

Localizar el archivo:
```bash
python -c "import scipy.stats._distn_infrastructure as m; print(m.__file__)"
```

Buscar las líneas:
```python
for obj in [s for s in dir() if s.startswith('_doc_')]:
    exec('del ' + obj)
del obj
```

Reemplazarlas por:
```python
for obj in [s for s in dir() if s.startswith('_doc_')]:
    exec('del ' + obj)
try:
    del obj
except NameError:
    pass
```

> **Nota:** este parche se aplica dentro del venv y no afecta scipy globalmente. Si se recrea el venv o se reinstala scipy, debe reaplicarse.

### Comando de build

Desde el directorio `cardio-backend/` con el venv activado:

```bash
rmdir /s /q build dist & del CardioPredictor.spec & pyinstaller --onefile --name CardioPredictor --add-data "models;models" --add-data "frontend_dist;frontend_dist" --hidden-import main --hidden-import numpy._core --hidden-import numpy._core._multiarray_umath --hidden-import numpy._core.multiarray --hidden-import joblib.externals.loky.backend.managers --collect-all numpy --collect-all scipy --collect-all joblib --collect-all shap --collect-all sklearn --copy-metadata numpy --copy-metadata scipy --copy-metadata joblib --copy-metadata scikit-learn --additional-hooks-dir . cardio_app.py
```

El ejecutable queda en `dist/CardioPredictor.exe`.

### Actualizar y reconstruir

**Solo cambios en el backend:**
1. Modificar el código Python
2. Ejecutar el comando de build

**Solo cambios en el frontend:**
1. `npm run build` en `cardio-frontend/`
2. `xcopy ..\cardio-frontend\dist frontend_dist /E /I /Y`
3. Ejecutar el comando de build

**Cambios en ambos:** completar los pasos del frontend primero, luego ejecutar el build una sola vez.

### Versiones del entorno de build

| Herramienta | Versión |
|---|---|
| Python | 3.12 |
| PyInstaller | 6.20.0 |
| numpy | 1.26.4 |
| scipy | 1.11.4 |
| joblib | 1.4.2 |

---

## Cómo extender el sistema

**Agregar un nuevo modelo:**
1. Crear archivo en `app/ml/`
2. Agregar la ruta en `config.py` y `.env.example`
3. Actualizar `model_loader.py`

**Agregar un nuevo endpoint:**
1. Crear archivo en `app/api/routes/`
2. Registrarlo en `app/api/router.py`
3. No tocar `main.py`

---

## Notas

- El archivo `.env` nunca se sube a Git. Usar `.env.example` como plantilla.

- El hook `hook-numpy.py` en la raíz del proyecto es necesario para el build y no debe eliminarse.
