# CardioPredict API

Backend del sistema de predicción de riesgo cardiovascular con perfilamiento clínico mediante clustering e inteligencia artificial para médicos.

Desarrollado por Sebastián Torres Ortega, Mayerlis Acosta Peralta y Christian Rivera Dibasto como proyecto integrador de Ingeniería de Sistemas e Ingeniería Biomédica.

---

## ¿Qué hace este sistema?

Recibe datos de un paciente (manualmente o desde una historia clínica en JSON o PDF), los procesa mediante un pipeline de clustering (winsorización con límites clínicos expertos → imputación KNN → StandardScaler → PCA de 11 componentes → Random Forest), asigna al paciente a uno de cuatro perfiles clínicos — **Cardiovascular**, **Bajo riesgo**, **Cardiometabólico** o **Cardiorrenal** — y retorna las probabilidades de pertenencia a cada perfil junto con una descripción clínica. Opcionalmente calcula los índices de riesgo **Framingham 2008** y el ajuste de la **Sociedad Colombiana de Cardiología (SCC)**. También ofrece explicabilidad mediante valores SHAP en waterfall por cada uno de los 4 perfiles.

El modelo fue entrenado con el dataset real de pacientes colombianos (notebook_proy_final) con k=4 clusters. Las 19 variables de entrada son: `c_total`, `creatinina`, `glucosa`, `hdl`, `hemoglobina`, `ldl`, `leucocitos`, `plaquetas`, `trigliceridos`, `edad`, `sexo`, `zona`, `ap_hipertension`, `ta_sistolica`, `ta_diastolica`, `peso`, `talla`, `imc`, `TFG`. La talla se recibe en cm y se convierte a metros internamente.

---

## Stack tecnológico

| Capa | Tecnología |
|---|---|
| Framework web | FastAPI + Uvicorn |
| Modelo principal | Random Forest (k=4 clusters vía KMeans + PCA) |
| Imputación | KNNImputer (sklearn, k=5) |
| Explicabilidad | SHAP TreeExplainer |
| Validación de datos | Pydantic v2 |
| Extracción PDF | PyMuPDF · pdfplumber · pytesseract |
| Empaquetado escritorio | PyInstaller 6.x |
| Pruebas | pytest + httpx |
| Entrenamiento | Google Colab (notebook_proy_final) |

---

## Estructura del proyecto

cardio-backend/
├── app/
│   ├── api/
│   │   ├── router.py
│   │   └── routes/
│   │       ├── explain.py          ← SHAP waterfall por los 4 perfiles
│   │       ├── predict.py          ← Predicción desde formulario manual
│   │       └── upload.py           ← Predicción desde JSON o PDF
│   ├── core/
│   │   └── config.py               ← Rutas de artefactos y configuración
│   ├── ml/
│   │   ├── explainability.py       ← SHAP en espacio PCA → features originales
│   │   ├── framingham_calculator.py
│   │   ├── model_loader.py         ← Carga de los 6 artefactos .pkl
│   │   ├── predictor.py            ← Pipeline: scaler → PCA → RF → k=4 clusters
│   │   ├── preprocessing.py        ← Winsorización + imputación KNN + conversión talla
│   │   └── scc_calculator.py
│   ├── schemas/
│   │   ├── input_schema.py         ← 19 variables obligatorias + 4 opcionales Framingham
│   │   └── output_schema.py        ← Respuesta con 4 ClusterProb + SHAP
│   └── services/
│       ├── json_extractor.py       ← Mapeo de aliases JSON → columnas_modelo.pkl
│       ├── pdf_extractor.py        ← Extracción de texto/tablas desde PDF
│       └── prediction_service.py   ← Orquestación del pipeline completo
├── data/
│   ├── content/
│   ├── processed/
│   └── raw/
├── dist/
│   └── CardioPredictor.exe         ← ejecutable de escritorio generado
├── frontend_dist/                  ← build del frontend (copiado desde cardio-frontend/dist/)
├── models/
│   ├── columnas_modelo.pkl         ← orden exacto de columnas de entrada
│   ├── imputer.pkl                 ← KNNImputer entrenado
│   ├── kmeans.pkl                  ← referencia de clusters
│   ├── pca.pkl                     ← PCA de 11 componentes
│   ├── random_forest.pkl           ← clasificador principal
│   └── scaler.pkl                  ← StandardScaler entrenado
├── notebooks/
├── tests/
├── cardio_app.py                   ← punto de entrada para el ejecutable
├── hook-numpy.py                   ← hook personalizado de PyInstaller para numpy
├── main.py
├── requirements.txt
├── .env.example
└── .gitignore

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

# 5. Verificar que los 6 artefactos estén en models/
# random_forest.pkl · scaler.pkl · pca.pkl
# kmeans.pkl · imputer.pkl · columnas_modelo.pkl

# 6. Arrancar el servidor
uvicorn main:app --reload
```

El servidor queda disponible en `http://localhost:8000`.
La documentación interactiva Swagger está en `http://localhost:8000/docs`.

---

## Variables de entorno

Copiar `.env.example` como `.env` y completar:

```env
# Rutas a los artefactos del modelo final (Colombia) — k=4 clusters
MODEL_PATH=models/random_forest.pkl
SCALER_PATH=models/scaler.pkl
PCA_PATH=models/pca.pkl
IMPUTER_PATH=models/imputer.pkl
COLUMNAS_PATH=models/columnas_modelo.pkl

DEBUG=True
ALLOWED_ORIGINS=http://localhost:5173
MAX_UPLOAD_SIZE_MB=5
```

---

## Endpoints

### `POST /api/predict/`
Predicción desde formulario manual con 19 variables clínicas obligatorias. Retorna el perfil predicho (0-3), nombre del cluster, descripción clínica y probabilidades de los 4 perfiles.

### `POST /api/upload`
Predicción desde historia clínica en JSON. Detecta campos faltantes automáticamente y permite completarlos vía query params.

### `POST /api/upload/pdf`
Predicción desde uno o varios PDFs (máximo 5). Fusiona los campos extraídos de todos los archivos.

### `POST /api/predict/explain`
Retorna los valores SHAP en unidades originales para cada variable y cada uno de los 4 perfiles, junto con los valores base del modelo. Permite construir el gráfico waterfall en el frontend.

### `GET /api/health`
Verifica que el servidor y los 6 artefactos estén cargados correctamente.

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

**Nuevos artefactos `.pkl`:**
Al reemplazar artefactos del modelo, verificar que los 6 archivos estén presentes en `models/` antes de ejecutar el build.

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

**Agregar un nuevo artefacto de modelo:**
1. Añadir la ruta en `config.py` y `.env.example`
2. Cargar el artefacto en `model_loader.py`
3. Usarlo en `preprocessing.py` o `predictor.py`

**Agregar un nuevo endpoint:**
1. Crear archivo en `app/api/routes/`
2. Registrarlo en `app/api/router.py`
3. No tocar `main.py`

---

## Notas

- El archivo `.env` nunca se sube a Git. Usar `.env.example` como plantilla.
- El hook `hook-numpy.py` en la raíz del proyecto es necesario para el build y no debe eliminarse.
- `xgboost_model.pkl` permanece en `models/` como modelo alternativo evaluado durante el desarrollo pero no usado en producción (el modelo seleccionado es Random Forest).
- Las columnas de entrada se determinan dinámicamente desde `columnas_modelo.pkl`. No modificar el orden manualmente.