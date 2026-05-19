# Sistema Agéntico para Análisis Predictivo — Premier League

**Proyecto Final — Programación para Analítica de Datos 2026-I**  
Universidad de Manizales

## Integrantes
- Juan C. V. Murillo
- [Tu nombre aquí]

---

## Descripción del Proyecto

Sistema agéntico que implementa un pipeline completo de ciencia de datos
sobre partidos de la Premier League inglesa, siguiendo una arquitectura
modular con un agente orquestador y tres skills diferenciadas.

**Objetivo de predicción:** clasificar el resultado de cada partido en
`Local`, `Empate` o `Visitante` usando únicamente features históricas
calculadas **antes** del partido (sin data leakage).

---

## Arquitectura del Sistema

```
ProyectoAgentico_BDFutbol/
│
├── data/
│   └── premier_league.db          # Base de datos SQLite con partidos
│
├── graphs/                        # Gráficas generadas automáticamente
│   ├── 01_distribucion_resultados.png
│   ├── 02_promedio_goles_equipos.png
│   ├── 03_heatmap_correlacion.png
│   ├── 04_comparacion_modelos.png
│   ├── 05_matriz_confusion.png
│   └── 06_importancia_features.png
│
├── notebooks/
│   └── pipeline_agentico.ipynb    # Notebook ejecutable principal
│
├── scripts/
│   ├── orchestrator.py            # Agente orquestador principal
│   └── skills/
│       ├── skill_preparacion.py   # Skill 1: Preparación de datos
│       ├── skill_eda.py           # Skill 2: Análisis exploratorio
│       └── skill_modelado.py      # Skill 3: Modelado predictivo
│
├── prompts_usados.md              # Prompts usados con IA generativa
└── README.md
```

---

## Requisito Arquitectónico

El sistema cumple con los requisitos del proyecto final:

| Requisito | Cumplimiento |
|-----------|-------------|
| Un único agente orquestador | `orchestrator.py` |
| Tres skills diferenciadas | `skill_preparacion`, `skill_eda`, `skill_modelado` |
| Separación clara de responsabilidades | Cada skill tiene una función orquestadora única |
| Flujo explícito de ejecución | `ejecutar_pipeline()` coordina las 3 skills |
| El agente NO contiene lógica analítica | Solo invoca skills y consolida resultados |
| No hay código monolítico | Cada skill es un módulo independiente |

---

## Skills del Sistema

### Skill 1 — Preparación de Datos (`skill_preparacion.py`)
**Función principal:** `preparar_datos(db_path) → DataFrame`

- Carga datos desde SQLite (`tabla: partidos`)
- Limpieza: duplicados, tipos de datos, valores nulos, goles negativos
- Feature engineering histórico (ventana de 5 partidos):
  - `promedio_goles_local` / `promedio_goles_visitante`
  - `racha_local` / `racha_visitante`
  - `ventaja_local` (indicador binario)
- Crea variable objetivo `resultado`: Local / Empate / Visitante

### Skill 2 — Análisis Exploratorio (`skill_eda.py`)
**Función principal:** `ejecutar_eda(df) → Dict`

- Estadísticas descriptivas generales
- Distribución de la variable objetivo
- Top 5 equipos con más victorias (local y visitante)
- Matriz de correlación entre features numéricas
- 3 gráficas: distribución de resultados, promedios por equipo, heatmap

### Skill 3 — Modelado Predictivo (`skill_modelado.py`)
**Función principal:** `ejecutar_modelado(df) → Dict`

- Selección de features sin data leakage
- División train/test 80/20 estratificada
- Entrenamiento de 3 modelos: Regresión Logística, Árbol de Decisión, Random Forest
- Evaluación: Accuracy, F1 Macro, Validación Cruzada 5-fold
- Selección del mejor modelo (Score = 0.6×F1_macro + 0.4×CV_mean)
- 3 gráficas: comparación de modelos, matriz de confusión, importancia de features

---

## Agente Orquestador (`orchestrator.py`)
**Función principal:** `ejecutar_pipeline(db_path) → Dict`

Coordina las tres skills en secuencia, pasa los datos entre ellas
y consolida un resumen ejecutivo final. No contiene lógica analítica.

```
ejecutar_pipeline()
    │
    ├── [Paso 1] preparar_datos()      → df
    ├── [Paso 2] ejecutar_eda(df)      → reporte_eda
    ├── [Paso 3] ejecutar_modelado(df) → reporte_modelado
    └── [Paso 4] consolidar_resultados() → resumen_ejecutivo
```

---

## Instalación y Ejecución

### Requisitos
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Ejecución desde el Notebook (recomendado)
Abrir `notebooks/pipeline_agentico.ipynb` y ejecutar todas las celdas.

### Ejecución desde Python
```python
from scripts.orchestrator import ejecutar_pipeline

resumen = ejecutar_pipeline('data/premier_league.db')
print(resumen['modelo_ganador'])
print(f"Accuracy: {resumen['accuracy_test']:.4f}")
```

### Ejecución desde terminal
```bash
cd ProyectoAgentico_BDFutbol
python -m scripts.orchestrator data/premier_league.db
```

---

## Resultados del Pipeline

El sistema produce como mínimo:
- **Dataset preparado** con features históricas listas para modelado
- **Reporte EDA** con 5 hallazgos clave sobre la distribución de datos
- **3 modelos entrenados** y evaluados con métricas comparables
- **Modelo final seleccionado** con justificación explícita
- **6 gráficas** guardadas en `graphs/`

---

## Fuente de Datos

Los datos provienen de una API de fútbol y están almacenados en SQLite.
La tabla `partidos` contiene: `id`, `fecha`, `equipo_local`,
`equipo_visitante`, `goles_local`, `goles_visitante`.

---

## Tecnologías Utilizadas

- **Python 3.x**
- **pandas** — Manipulación de datos
- **numpy** — Operaciones numéricas
- **scikit-learn** — Modelos de ML y métricas
- **matplotlib / seaborn** — Visualizaciones
- **SQLite** — Almacenamiento de datos
- **Jupyter Notebook** — Entorno de ejecución interactivo
