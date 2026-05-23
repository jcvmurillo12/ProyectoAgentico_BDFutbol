---
name: modelado-predictivo
description: Entrena LogisticRegression y RandomForestClassifier para predecir resultados de partidos (Local/Empate/Visitante), evalúa con accuracy/precision/recall, selecciona el mejor modelo y genera reporte.
---

# Skill: Modelado Predictivo

## Cuándo usar esta skill

Usar cuando se necesite:
- Entrenar modelos de machine learning para predicción de resultados
- Comparar desempeño de múltiples algoritmos
- Evaluar modelos con métricas multiclass
- Generar visualizaciones de desempeño
- Seleccionar el mejor modelo para producción

Esta skill es el **tercer y final paso** del pipeline y requiere que se haya ejecutado `preparacion-datos`.

## Descripción técnica

### Funcionalidad
La skill ejecuta un pipeline completo de modelado:

1. **Preparación de features**: Selecciona 5 features predictoras
2. **División de datos**: Train/test 80/20 con estratificación
3. **Escalado**: StandardScaler para features numéricas
4. **Entrenamiento**: Dos modelos en paralelo
5. **Evaluación**: Métricas multiclass (accuracy, precision, recall)
6. **Selección**: Mejor modelo basado en accuracy
7. **Visualizaciones**: 3 gráficas PNG de desempeño

### Modelos entrenados

#### LogisticRegression
- Tipo: Modelo lineal multinomial
- Solver: LBFGS (optimización)
- Regularización: L2 (default)
- Max iterations: 1000
- Multi-class: Multinomial (para 3 clases)

#### RandomForestClassifier
- Tipo: Ensemble de árboles de decisión
- N estimators: 100 árboles
- Max depth: 10 (control de complejidad)
- Min samples split: 5
- Min samples leaf: 2
- Random state: 42 (reproducibilidad)
- N jobs: -1 (usa todos los procesadores)

### Features utilizadas
1. `promedio_goles_local`: Promedio histórico de goles del equipo local
2. `promedio_goles_visitante`: Promedio histórico de goles del equipo visitante
3. `racha_local`: Victorias recientes del equipo local
4. `racha_visitante`: Victorias recientes del equipo visitante
5. `ventaja_local`: Indicador de ventaja de campo (siempre 1)

### Clases objetivo
- `Local`: Victoria del equipo local (goles_local > goles_visitante)
- `Empate`: Resultado igualado (goles_local == goles_visitante)
- `Visitante`: Victoria del equipo visitante (goles_local < goles_visitante)

### Métricas de evaluación
- **Accuracy**: Proporción de predicciones correctas (métrica principal)
- **Precision (macro)**: Promedio de precisión por clase
- **Recall (macro)**: Promedio de recall por clase
- **Matriz de confusión**: Distribución de errores por clase

## Cómo ejecutar

### Desde línea de comandos
```bash
python skills/modelado/modelado.py
```

### Desde Python
```python
from skills.modelado.modelado import ejecutar_modelado
import pandas as pd

df = pd.read_csv('data/dataset_preparado.csv')
resultados = ejecutar_modelado(df)
print(f"Mejor modelo: {resultados['mejor_modelo']}")
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

**Estructura esperada**:
```
- id (int)
- fecha (datetime)
- equipo_local (str)
- equipo_visitante (str)
- goles_local (int)
- goles_visitante (int)
- promedio_goles_local (float)
- promedio_goles_visitante (float)
- racha_local (int)
- racha_visitante (int)
- ventaja_local (int)
- resultado (str): Local/Empate/Visitante
```

## Output

### Archivos generados
- `data/reporte_modelado.json`: Reporte con resultados del modelado

### Estructura del reporte JSON
```json
{
  "mejor_modelo": "RandomForestClassifier",
  "metricas": {
    "accuracy": 0.6105,
    "precision": 0.5847,
    "recall": 0.6105
  },
  "reporte_modelado": {
    "rendimiento_general": "Se entrenaron dos modelos...",
    "metricas_detalladas": "El modelo ganador...",
    "justificacion": "RandomForest superó...",
    "recomendaciones": "Se recomienda utilizar..."
  }
}
```

### Gráficas generadas
- `graphs/04_matriz_confusion_logisticregression.png`: Matriz de confusión LR
- `graphs/04_matriz_confusion_randomforestclassifier.png`: Matriz de confusión RF
- `graphs/05_comparacion_modelos.png`: Comparación de métricas (8×6 pulgadas, 300 DPI)

### Ejemplo de salida
```
======================================================================
INICIANDO PIPELINE DE MODELADO PREDICTIVO
======================================================================

[1/9] Preparando datos...
✓ Features seleccionados: ['promedio_goles_local', 'promedio_goles_visitante', 'racha_local', 'racha_visitante', 'ventaja_local']
✓ Dimensiones: X=(380, 5), y=(380,)
✓ Clases: ['Empate' 'Local' 'Visitante']

[2/9] Dividiendo datos en entrenamiento/prueba...
✓ Datos divididos:
  - Entrenamiento: 304 samples (80%)
  - Prueba: 76 samples (20%)

[3/9] Escalando features...
✓ Features escalados usando StandardScaler

[4/9] Entrenando LogisticRegression...
✓ Modelo LogisticRegression entrenado

[5/9] Entrenando RandomForestClassifier...
✓ Modelo RandomForestClassifier entrenado

[6/9] Evaluando LogisticRegression...
📊 MÉTRICAS - LogisticRegression
----------------------------------------------------------------------
Accuracy:  0.6053
Precision: 0.5894
Recall:    0.6053

[7/9] Evaluando RandomForestClassifier...
📊 MÉTRICAS - RandomForestClassifier
----------------------------------------------------------------------
Accuracy:  0.6105
Precision: 0.5847
Recall:    0.6105

[8/9] Seleccionando mejor modelo...
======================================================================
🏆 MODELO GANADOR: RandomForestClassifier
======================================================================
Accuracy:  0.6105
Precision: 0.5847
Recall:    0.6105
======================================================================
```

## Comportamiento esperado

### Tiempo de ejecución
- LogisticRegression: 1-2 segundos
- RandomForest: 3-5 segundos
- Evaluación + Gráficas: 5-10 segundos
- **Total**: ~15-20 segundos

### División de datos
- Train: 304 samples (80%)
- Test: 76 samples (20%)
- Estratificado por clase para mantener proporciones

### Métricas esperadas
- **Accuracy**: 0.50-0.65 (predicción de 3 clases es desafiante)
- **Precision/Recall**: Similar a accuracy ±0.05
- Mejor rendimiento típicamente en clases con más muestras

## Dependencias

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from typing import Dict, Tuple, Any
import json
```

Instalar con:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Interpretación de resultados

### Matrices de confusión
- Diagonal: Predicciones correctas
- Fuera de diagonal: Errores de clasificación
- Clases con mejor desempeño tendrán diagonales más oscuras

### Comparación de modelos
- **LogisticRegression**: Más interpretable, generalmente más rápido
- **RandomForest**: Mejor con relaciones no-lineales, más robusto a outliers

### Selección de modelo
- Criterio: **Máximo Accuracy** en datos de test
- Desempate: Considera Precision y Recall

## Mejoras futuras

1. **Cross-Validation**: K-fold para estimación más robusta de desempeño
2. **Hyperparameter Tuning**: GridSearchCV para optimizar parámetros
3. **Ensemble Methods**: VotingClassifier para combinar predicciones
4. **Feature Importance**: Análisis de importancia de features (RandomForest)
5. **Calibración**: Probabilidades calibradas para predicciones probabilísticas
6. **Validación en nuevos datos**: Evaluación en futuras temporadas

## Notas importantes

1. **Reproducibilidad**: Random state fijado en 42 para resultados consistentes
2. **Estratificación**: Division respeta proporción de clases
3. **Escalado**: Solo aplicado a LogisticRegression (RandomForest no requiere)
4. **Multiclass**: Ambos modelos configurados para 3 clases (Local/Empate/Visitante)
5. **Gráficas**: Siempre 300 DPI para uso en reportes

## Troubleshooting

### Error: FileNotFoundError
```
ERROR: data/dataset_preparado.csv no encontrado
```
**Solución**: Ejecutar primero `preparacion-datos` para generar el CSV

### Error: Clase faltante
```
ValueError: y contains previously unseen labels
```
**Solución**: Verificar que el dataset contiene las 3 clases (Local/Empate/Visitante)

### Accuracy muy bajo (<0.33)
**Posible causa**: Problema en features o dataset  
**Solución**: Revisar distribución de datos, features nulas, o valores anómalos

### Memorias insuficiente (RandomForest)
**Solución**: Reducir n_estimators a 50, max_depth a 8, o usar n_jobs=-1 menos agresivamente

## Referencias

- **Estándar**: Anthropic Agent Skills
- **Tipo**: Machine Learning Model Training Skill
- **Dependencia**: `preparacion-datos` (debe ejecutarse primero)
- **Próxima skill**: Ninguna (última del pipeline)
- **Algoritmos**: scikit-learn 1.0+
