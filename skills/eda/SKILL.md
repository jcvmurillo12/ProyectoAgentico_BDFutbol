---
name: analisis-exploratorio
description: Ejecuta análisis exploratorio sobre el dataset preparado de Premier League, genera estadísticas descriptivas, distribución de resultados, top equipos y visualizaciones guardadas como imágenes PNG.
---

# Skill: Análisis Exploratorio de Datos (EDA)

## Cuándo usar esta skill

Usar cuando se necesite:
- Explorar el dataset preparado de partidos
- Generar estadísticas descriptivas y análisis de distribuciones
- Identificar equipos dominantes (locales y visitantes)
- Calcular correlaciones entre features
- Crear visualizaciones para comprensión del dataset

Esta skill es el **segundo paso** del pipeline y requiere que primero se haya ejecutado `preparacion-datos`.

## Descripción técnica

### Funcionalidad
La skill ejecuta análisis exploratorio completo:

1. **Estadísticas descriptivas**: Resumen general del dataset
2. **Análisis de distribución**: Variable objetivo (Local/Empate/Visitante)
3. **Equipos dominantes**: Top 5 equipos como local y visitante
4. **Matriz de correlación**: Relaciones entre features numéricas
5. **Visualizaciones**: 3 gráficas PNG de alta calidad (300 DPI)
6. **Reporte textual**: Hallazgos consolidados en JSON

### Análisis realizados

#### 1. Estadísticas Descriptivas
- Total de partidos (380)
- Rango de fechas
- Cantidad de equipos únicos
- Promedio de goles (general, local, visitante)

#### 2. Distribución de Resultados
- Conteo por clase (Local/Empate/Visitante)
- Porcentaje de cada resultado
- Detección de desbalance de clases

#### 3. Equipos Dominantes
- Top 5 equipos con más victorias como local
- Top 5 equipos con más victorias como visitante
- Tasa de victoria por equipo

#### 4. Correlaciones
- Matriz de correlación de Pearson
- Features numéricas analizadas:
  - `goles_local`, `goles_visitante`
  - `promedio_goles_local`, `promedio_goles_visitante`
  - `racha_local`, `racha_visitante`
  - `ventaja_local`

### Visualizaciones generadas
1. **Distribución de resultados**: Countplot con colores distintivos (Verde/Naranja/Rojo)
2. **Promedio de goles por equipo**: Gráfica horizontal comparando local vs visitante (Top 10)
3. **Heatmap de correlación**: Matriz con escala coolwarm (-1 a +1)

## Cómo ejecutar

### Desde línea de comandos
```bash
python skills/eda/eda.py
```

### Desde Python
```python
from skills.eda.eda import ejecutar_eda
import pandas as pd

df = pd.read_csv('data/dataset_preparado.csv')
reporte_eda = ejecutar_eda(df)
```

## Input

**Requerido**:
- `data/dataset_preparado.csv`: Dataset preparado por skill anterior
  - Generado por `preparacion-datos`
  - Debe contener 380 partidos × 12 columnas

**Ubicación esperada**:
```
data/dataset_preparado.csv
```

**Estructura esperada del DataFrame**:
```
- id (int): ID único del partido
- fecha (datetime): Fecha del partido
- equipo_local (str): Nombre del equipo local
- equipo_visitante (str): Nombre del equipo visitante
- goles_local (int): Goles del equipo local
- goles_visitante (int): Goles del equipo visitante
- promedio_goles_local (float): Feature histórica
- promedio_goles_visitante (float): Feature histórica
- racha_local (int): Feature histórica
- racha_visitante (int): Feature histórica
- ventaja_local (int): Feature de campo
- resultado (str): Variable objetivo (Local/Empate/Visitante)
```

## Output

### Archivos generados
- `data/reporte_eda.json`: Reporte con hallazgos principales

### Estructura del reporte JSON
```json
{
  "estadisticas_generales": "El dataset contiene 380 partidos...",
  "distribucion_resultados": "La variable objetivo 'resultado' muestra...",
  "equipos_dominantes": "El equipo más ganador como local es...",
  "correlaciones_principales": "Se detectan correlaciones relevantes...",
  "conclusion": "El análisis exploratorio revela..."
}
```

### Gráficas generadas
- `graphs/01_distribucion_resultados.png`: Countplot (10×6)
- `graphs/02_promedio_goles_equipos.png`: Gráfica horizontal (14×7)
- `graphs/03_heatmap_correlacion.png`: Heatmap (10×8)

**Formato**: PNG, 300 DPI, optimizado para impresión

### Ejemplo de salida
```
======================================================================
INICIANDO ANÁLISIS EXPLORATORIO DE DATOS (EDA)
======================================================================

[1/6] Calculando estadísticas descriptivas...
📊 ESTADÍSTICAS DESCRIPTIVAS GENERALES
----------------------------------------------------------------------
Total de partidos: 380
Rango de fechas: 2023-08-12 a 2024-05-19
Cantidad de equipos únicos: 20
Promedio de goles por partido: 2.68
Promedio de goles local: 1.51
Promedio de goles visitante: 1.17

[2/6] Analizando distribución de resultados...
⚽ DISTRIBUCIÓN DE LA VARIABLE OBJETIVO
----------------------------------------------------------------------
Local    : 118 partidos ( 31.1%)
Empate   :  84 partidos ( 22.1%)
Visitante: 178 partidos ( 46.8%)
```

## Comportamiento esperado

### Tiempo de ejecución
- Típico: 5-15 segundos (dataset de 380 partidos)
- Generación de gráficas: ~10 segundos

### Logs de consola
```
[3/6] Analizando equipos dominantes como local...
🏆 TOP 5 EQUIPOS CON MÁS VICTORIAS COMO LOCAL
...
[4/6] Analizando equipos dominantes como visitante...
🚀 TOP 5 EQUIPOS CON MÁS VICTORIAS COMO VISITANTE
...
[5/6] Calculando matriz de correlación...
📈 CORRELACIONES ENTRE FEATURES NUMÉRICAS
[6/6] Generando visualizaciones...
----------------------------------------------------------------------
✓ Gráfica 1 guardada: graphs/01_distribucion_resultados.png
✓ Gráfica 2 guardada: graphs/02_promedio_goles_equipos.png
✓ Gráfica 3 guardada: graphs/03_heatmap_correlacion.png
----------------------------------------------------------------------
```

## Dependencias

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
from datetime import datetime
import os
import json
```

Instalar con:
```bash
pip install pandas numpy matplotlib seaborn
```

## Visualización de resultados

### En Jupyter Notebook
```python
from IPython.display import Image, display

# Mostrar gráfica
display(Image('graphs/01_distribucion_resultados.png'))
```

### Con matplotlib
```python
from PIL import Image
import matplotlib.pyplot as plt

img = Image.open('graphs/01_distribucion_resultados.png')
plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis('off')
plt.show()
```

## Interpretación de resultados

### Estadísticas Clave
- **Ventaja de campo**: Los resultados con victoria local superan los de visitante (~31% vs ~47%)
  - Nota: El dataset muestra más victorias de visitantes, indicando equilibrio competitivo
- **Promedio de goles**: 2.68 por partido indica liga con buen desempeño ofensivo
- **Diferencia local-visitante**: 0.34 goles sugiere equilibrio competitivo

### Correlaciones Relevantes
- Features históricas muestran correlación moderada (0.5-0.8)
- Racha y promedio_goles suelen correlacionar positivamente
- Variables relativamente independientes facilitan modelado

### Equipos Dominantes
- Top equipos locales: Generalmente tienen >70% de victorias en casa
- Top equipos visitantes: Suelen tener 35-50% de victorias de visitante

## Notas importantes

1. **Reproducibilidad**: Mismo dataset genera siempre el mismo reporte
2. **Actualización de gráficas**: Se sobrescriben si se ejecuta nuevamente
3. **Rutas relativas**: Las gráficas se guardan en carpeta `graphs/` relativa al directorio de ejecución
4. **Resolución**: Todas las gráficas se generan a 300 DPI para uso en reportes

## Troubleshooting

### Error: FileNotFoundError
```
ERROR: data/dataset_preparado.csv no encontrado
```
**Solución**: Ejecutar primero `preparacion-datos` para generar el CSV

### Error: Columna no encontrada
```
KeyError: 'resultado'
```
**Solución**: Verificar que el CSV contiene todas las columnas esperadas

### Gráficas en blanco
**Solución**: Verificar que matplotlib backend está configurado correctamente en Jupyter

## Referencias

- **Estándar**: Anthropic Agent Skills
- **Tipo**: Data Analysis Skill
- **Dependencia**: `preparacion-datos` (debe ejecutarse primero)
- **Próxima skill**: `modelado-predictivo`
- **Prerequisito previo**: Ninguno (usa output de preparacion-datos)
