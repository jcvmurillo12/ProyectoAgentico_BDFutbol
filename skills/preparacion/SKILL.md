---
name: preparacion-datos
description: Carga datos de partidos de Premier League desde SQLite, limpia el dataset y calcula features históricas como promedio de goles y racha de victorias para preparar los datos para modelado predictivo.
---

# Skill: Preparación de Datos

## Cuándo usar esta skill

Usar cuando se necesite:
- Cargar datos de partidos desde `premier_league.db` (base de datos SQLite)
- Limpiar y validar el dataset
- Calcular features históricas para modelado
- Crear variable objetivo de resultado (Local/Empate/Visitante)

Esta skill es el **primer paso** del pipeline y es prerequisito para ejecutar las otras skills.

## Descripción técnica

### Funcionalidad
La skill ejecuta un pipeline de preparación completo:

1. **Carga de datos**: Extrae 380 partidos completados desde SQLite
2. **Limpieza**: Elimina duplicados, valida tipos de datos, detecta anomalías
3. **Enriquecimiento**: Calcula features históricas basadas en últimos 5 partidos
4. **Variable objetivo**: Crea columna `resultado` (Local/Empate/Visitante)

### Features calculadas
- `promedio_goles_local`: Promedio de goles como local (últimos 5 partidos)
- `promedio_goles_visitante`: Promedio de goles como visitante (últimos 5 partidos)
- `racha_local`: Victorias como local (últimos 5 partidos)
- `racha_visitante`: Victorias como visitante (últimos 5 partidos)
- `ventaja_local`: Indicador de ventaja de campo (siempre 1)

### Validaciones realizadas
- Detección y eliminación de duplicados
- Conversión correcta de tipos de datos
- Verificación de valores nulos en columnas críticas
- Detección de goles negativos
- Ordenamiento cronológico

## Cómo ejecutar

### Desde línea de comandos
```bash
python skills/preparacion/preparacion.py <db_path>
```

### Ejemplo
```bash
python skills/preparacion/preparacion.py "C:\Users\juanc\Documents\Universidad\Programacion Analitica De Datos\ProyectoAgentico_BDFutbol\data\premier_league.db"
```

### Desde Python
```python
from skills.preparacion.preparacion import preparar_datos
df = preparar_datos("path/to/premier_league.db")
```

## Input

**Requerido**:
- `premier_league.db`: Base de datos SQLite con tabla `fixtures`
  - Columnas esperadas: `fixture.id`, `fixture.date`, `teams.home.name`, `teams.away.name`, `goals.home`, `goals.away`, `fixture.status.short`
  - Registros: Mínimo 380 partidos con estatus 'FT' (Full Time)

**Ubicación esperada**:
```
data/premier_league.db
```

## Output

### Archivos generados
- `data/dataset_preparado.csv`: Dataset enriquecido en formato CSV

### Estructura del DataFrame de salida
```
Columnas: 12
- id (int): ID único del partido
- fecha (datetime): Fecha del partido
- equipo_local (str): Nombre del equipo local
- equipo_visitante (str): Nombre del equipo visitante
- goles_local (int): Goles anotados por equipo local
- goles_visitante (int): Goles anotados por equipo visitante
- promedio_goles_local (float): Feature histórica
- promedio_goles_visitante (float): Feature histórica
- racha_local (int): Feature histórica
- racha_visitante (int): Feature histórica
- ventaja_local (int): Feature de campo (siempre 1)
- resultado (str): Variable objetivo (Local/Empate/Visitante)

Dimensión: 380 rows × 12 columns
```

### Ejemplo de salida
```csv
id,fecha,equipo_local,equipo_visitante,goles_local,goles_visitante,promedio_goles_local,promedio_goles_visitante,racha_local,racha_visitante,ventaja_local,resultado
1,2023-08-12,Arsenal,Nottingham Forest,2,1,0.0,0.0,0,0,1,Local
2,2023-08-12,Aston Villa,West Ham,1,0,0.0,0.0,0,0,1,Local
3,2023-08-12,Bournemouth,Fulham,1,0,0.0,0.0,0,0,1,Local
...
```

## Comportamiento esperado

### Tiempo de ejecución
- Típico: 30-60 segundos para 380 partidos
- Cálculo de features: ~45 segundos (iteración por partido)

### Logs de consola
```
======================================================================
INICIANDO PIPELINE DE PREPARACIÓN DE DATOS
======================================================================

[1/5] Cargando datos...

[2/5] Limpiando datos...
✓ Duplicados eliminados: 0
✓ Tipos de datos convertidos correctamente
✓ Datos ordenados por fecha. Rango: 2023-08-12 a 2024-05-19
[3/5] Creando features históricas...
Calculando features históricas...
  → Procesados 76/380 partidos...
  → Procesados 152/380 partidos...
  → Procesados 228/380 partidos...
  → Procesados 304/380 partidos...
✓ Features históricas calculadas para 380 partidos

[4/5] Creando variable objetivo...
✓ Variable objetivo 'resultado' creada:
  - Local: 118 (31.1%)
  - Empate: 84 (22.1%)
  - Visitante: 178 (46.8%)

[5/5] Generando resumen final...
...
======================================================================
✓ PIPELINE COMPLETADO EXITOSAMENTE
======================================================================
```

## Dependencias

```python
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, List
```

Instalar con:
```bash
pip install pandas numpy
```

## Notas importantes

1. **Primeros 5 partidos sin features**: Los primeros 5 partidos de cada equipo tendrán features = 0 ya que no hay historial previo
2. **Orden cronológico**: El DataFrame está ordenado por fecha para facilitar cálculo de features
3. **Frecuencia de actualización**: Si se agrega un nuevo partido a la DB, se debe re-ejecutar esta skill
4. **CSV separador**: El archivo CSV usa coma (,) como separador

## Troubleshooting

### Error: FileNotFoundError
```
ERROR: Base de datos no encontrada
```
**Solución**: Verificar ruta de `premier_league.db` y permisos de lectura

### Error: sqlite3.OperationalError
```
ERROR: no such table: fixtures
```
**Solución**: Verificar que la tabla `fixtures` exista en la base de datos

### Warning: Valores nulos detectados
```
⚠ Valores nulos detectados:
```
**Solución**: La skill automáticamente elimina filas con nulos en columnas críticas

## Referencias

- **Estándar**: Anthropic Agent Skills
- **Tipo**: Data Preparation Skill
- **Dependencia**: Ninguna (primera skill del pipeline)
- **Próxima skill**: `analisis-exploratorio`
