# CardioPredict API

Backend del sistema de predicción de riesgo cardiovascular con explicabilidad clínica para médicos.

Desarrollado por Sebastián Torres Ortega, Mayerlis Acosta Peralta y Christian Rivera Dibasto como proyecto integrador de Ingeniería de Sistemas e Ingeniería Biomédica.

---

## ¿Qué hace este sistema?

Recibe datos de un paciente (manualmente o desde una historia clínica en JSON o PDF), predice el riesgo de enfermedad cardiovascular usando un modelo XGBoost entrenado sobre 68 515 registros, y retorna la predicción junto con una explicación en lenguaje clínico que indica qué factores aumentaron o redujeron el riesgo y en qué proporción.

---

## Stack tecnológico

| Capa | Tecnología |
|---|---|
| Framework web | FastAPI + Uvicorn |
| Modelo principal | XGBoost (AUC-ROC 0.799) |
| Modelo alternativo | Random Forest (AUC-ROC 0.798) |
| Explicabilidad | SHAP + LIME |
| Validación de datos | Pydantic v2 |
| Extracción PDF | PyMuPDF · pdfplumber · pytesseract |
| Pruebas | pytest + httpx |
| Entrenamiento | Google Colab (notebooks en `notebooks/`) |

---

## Estructura del proyecto

```
cardio-backend/
│
├── app/
│   ├── api/
│   │   ├── router.py              # Registra todos los routers
│   │   └── routes/
│   │       ├── predict.py         # POST /api/predict
│   │       └── upload.py          # POST /api/predict/upload y /upload/pdf
│   │
│   ├── core/
│   │   └── config.py              # Configuración desde .env
│   │
│   ├── ml/
│   │   ├── model_loader.py        # Carga el .pkl una sola vez al arrancar
│   │   ├── preprocessing.py       # Limpieza y feature engineering por predicción
│   │   ├── predictor.py           # Inferencia del modelo
│   │   └── explainer.py           # SHAP y LIME → texto clínico legible
│   │
│   ├── schemas/
│   │   ├── input_schema.py        # Validación de entrada con Pydantic
│   │   └── output_schema.py       # Estructura de la respuesta
│   │
│   └── services/
│       ├── prediction_service.py  # Orquesta el flujo completo
│       ├── json_extractor.py      # Extrae campos de historia clínica JSON
│       └── pdf_extractor.py       # Extrae campos de historia clínica PDF
│
├── data/
│   ├── raw/cardio_train.csv       # Dataset original (70 000 registros)
│   └── processed/
│       ├── cardio_clean.csv       # Tras limpieza (68 515 registros)
│       └── cardio_features.csv    # Con features derivados (17 columnas)
│
├── models/
│   ├── random_forest.pkl
│   └── xgboost_model.pkl          # Modelo activo por defecto
│
├── notebooks/                     # Ejecutar en Google Colab
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_model_training.ipynb
│
├── tests/
│   ├── test_predict.py
│   ├── test_preprocessing.py
│   └── test_upload.py
│
├── main.py
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Instalación

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
# Editar .env con las rutas y configuración deseada

# 5. Verificar que los modelos estén en models/
# models/random_forest.pkl
# models/xgboost_model.pkl

# 6. Arrancar el servidor
uvicorn main:app --reload
```

El servidor queda disponible en `http://localhost:8000`.
La documentación interactiva Swagger está en `http://localhost:8000/docs`.

---

## Variables de entorno

Copiar `.env.example` como `.env` y completar:

```env
BEST_MODEL=xgboost              # 'xgboost' o 'random_forest'
MODEL_RF_PATH=models/random_forest.pkl
MODEL_XGB_PATH=models/xgboost_model.pkl
DEBUG=True
ALLOWED_ORIGINS=http://localhost:5173
MAX_UPLOAD_SIZE_MB=5
```

---

## Endpoints

### `POST /api/predict/`
Predicción desde formulario manual.

**Body:**
```json
{
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
  "active": 1
}
```

**Respuesta:**
```json
{
  "riesgo_cardiovascular": 1,
  "probabilidad": 0.8178,
  "nivel_riesgo": "Alto",
  "explicabilidad": [
    {
      "factor": "Presión sistólica",
      "impacto": 1.0557,
      "descripcion": "Su presión sistólica de 145 mmHg aumenta el riesgo en un 105.6%.",
      "nivel": "crítico",
      "advertencia": null
    }
  ]
}
```

---

### `POST /api/predict/upload`
Predicción desde historia clínica en JSON. Detecta campos faltantes automáticamente.

**Form data:** archivo `.json`

---

### `POST /api/predict/upload/pdf`
Predicción desde historia clínica en PDF. Acepta hasta 5 PDFs del mismo paciente y fusiona los campos encontrados en cada uno.

Detecta automáticamente el tipo de PDF:
- **Texto nativo** → PyMuPDF
- **Con tablas** → pdfplumber
- **Escaneado** → pytesseract (requiere Tesseract instalado en el sistema)

**Form data:** uno o más archivos `.pdf`

---

### `GET /api/health`
Verifica que el servidor y el modelo estén cargados.

```json
{"status": "ok", "model_activo": "xgboost"}
```

---

## Campos del modelo

| Campo | Tipo | Descripción | Rango válido |
|---|---|---|---|
| `age_days` | int | Edad en días | 10 000 – 40 000 |
| `gender` | int | 1 = mujer · 2 = hombre | 1 – 2 |
| `height` | int | Altura en cm | 140 – 220 |
| `weight` | float | Peso en kg | 30 – 180 |
| `ap_hi` | int | Presión sistólica (mmHg) | 60 – 250 |
| `ap_lo` | int | Presión diastólica (mmHg) | 40 – 200 |
| `cholesterol` | int | 1 normal · 2 alto · 3 muy alto | 1 – 3 |
| `gluc` | int | 1 normal · 2 alto · 3 muy alto | 1 – 3 |
| `smoke` | int | 0 no fuma · 1 fuma | 0 – 1 |
| `alco` | int | 0 no consume · 1 consume alcohol | 0 – 1 |
| `active` | int | 0 sedentario · 1 activo | 0 – 1 |

**Nota clínica:** las variables `smoke` y `alco` presentan sesgo de subregistro en el dataset de entrenamiento (8.8% y 5.4% de positivos autorreportados respectivamente). Su impacto en la explicabilidad puede subestimar el riesgo real del tabaquismo y el consumo de alcohol. Esto se documenta en la respuesta con el campo `advertencia` de cada factor.

---

## Modelo y rendimiento

Los modelos fueron entrenados sobre el dataset [Cardiovascular Disease](https://www.kaggle.com/code/abdelrahmanabdelalem/cardiovascular-disease-83-accuracy/notebook) con 68 515 registros tras limpieza.

| Modelo | Accuracy | F1 | AUC-ROC |
|---|---|---|---|
| Random Forest (tuned) | 0.731 | 0.715 | 0.798 |
| **XGBoost (tuned)** | **0.733** | **0.719** | **0.799** |

Hiperparámetros seleccionados con GridSearchCV y validación cruzada estratificada de 5 folds. Ver `notebooks/04_model_training.ipynb` para detalles.

Features usados: `age`, `gender`, `height`, `weight`, `ap_hi`, `ap_lo`, `cholesterol`, `gluc`, `smoke`, `alco`, `active`, `bmi`, `age_range`, `bp_category`, `pulse_pressure`, `metabolic_risk`.

---

## Pruebas

```bash
pytest tests/ -v
```

66 pruebas que cubren preprocesamiento, endpoints de predicción, carga de JSON y PDF, validaciones de entrada y manejo de errores.

---

## Agregar un nuevo modelo

1. Crear archivo en `app/ml/` con la lógica de inferencia
2. Agregar la ruta en `config.py` y `.env.example`
3. Actualizar `model_loader.py` para cargarlo según el nombre en `BEST_MODEL`
4. No tocar rutas ni servicios

## Agregar un nuevo endpoint

1. Crear archivo en `app/api/routes/`
2. Registrarlo en `app/api/router.py`
3. No tocar `main.py`

---

## Notas de despliegue

- **Vercel**: no soporta Tesseract OCR. Los PDFs de texto y tablas funcionan sin él. Para soporte completo de PDFs escaneados usar Railway o Render donde se puede instalar el paquete del sistema `tesseract-ocr`.
- Los archivos `.pkl` no se suben a Git (están en `.gitignore`). Deben copiarse manualmente a `models/` tras clonar el repositorio.
- El archivo `.env` nunca se sube a Git. Usar `.env.example` como plantilla.
