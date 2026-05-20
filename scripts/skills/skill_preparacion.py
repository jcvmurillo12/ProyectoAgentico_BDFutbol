"""
Skill de Preparación de Datos para Sistema Agéntico de Premier League.

Este módulo contiene la lógica de carga, limpieza y enriquecimiento de datos
de partidos de la Premier League, incluyendo la ingeniería de features históricas
y creación de la variable objetivo.

Funciones principales:
    - preparar_datos(db_path): Función orquestadora que ejecuta todo el pipeline.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, List


def cargar_datos_sqlite(db_path: str) -> pd.DataFrame:
    """
    Carga datos de la tabla 'fixtures' desde una base de datos SQLite.
    
    Parámetros:
        db_path (str): Ruta al archivo SQLite (ej: 'premier_league.db')
    
    Retorna:
        pd.DataFrame: DataFrame con las columnas normalizadas:
            - fecha: fecha del partido
            - equipo_local: nombre del equipo que juega en casa
            - equipo_visitante: nombre del equipo visitante
            - goles_local: goles anotados por el equipo local
            - goles_visitante: goles anotados por el equipo visitante
    
    Excepciones:
        FileNotFoundError: Si el archivo SQLite no existe.
        sqlite3.OperationalError: Si la tabla 'fixtures' no existe.
    """
    try:
        conexion = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM fixtures", conexion)
        conexion.close()
        
        # Renombrar columnas para consistencia interna
        df = df.rename(columns={
            'date': 'fecha',
            'home_team': 'equipo_local',
            'away_team': 'equipo_visitante',
            'home_goals': 'goles_local',
            'away_goals': 'goles_visitante'
        })
        
        print(f"✓ Datos cargados correctamente: {len(df)} partidos desde tabla 'fixtures'")
        return df
    
    except FileNotFoundError:
        raise FileNotFoundError(f"El archivo SQLite no existe en: {db_path}")
    except sqlite3.OperationalError as e:
        raise sqlite3.OperationalError(f"Error al acceder a la tabla 'fixtures': {str(e)}")


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza la limpieza y validación de datos del dataset de partidos.
    
    Operaciones realizadas:
        1. Elimina duplicados exactos.
        2. Valida y convierte tipos de datos.
        3. Detecta y maneja valores nulos.
        4. Valida rangos de goles (no negativos).
        5. Ordena por fecha.
    
    Parámetros:
        df (pd.DataFrame): DataFrame bruto de partidos.
    
    Retorna:
        pd.DataFrame: DataFrame limpio y validado.
    """
    df = df.copy()
    
    # 1. Eliminar duplicados exactos
    duplicados_iniciales = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"✓ Duplicados eliminados: {duplicados_iniciales}")
    
    # 2. Convertir tipos de datos
    try:
        df['fecha'] = pd.to_datetime(df['fecha'])
        df['goles_local'] = df['goles_local'].astype(int)
        df['goles_visitante'] = df['goles_visitante'].astype(int)
        df['equipo_local'] = df['equipo_local'].astype(str).str.strip()
        df['equipo_visitante'] = df['equipo_visitante'].astype(str).str.strip()
        print("✓ Tipos de datos convertidos correctamente")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Error al convertir tipos de datos: {str(e)}")
    
    # 3. Detectar valores nulos en columnas críticas
    columnas_criticas = ['fecha', 'equipo_local', 'equipo_visitante', 'goles_local', 'goles_visitante']
    nulos = df[columnas_criticas].isnull().sum()
    if nulos.any():
        print(f"⚠ Valores nulos detectados:\n{nulos[nulos > 0]}")
        df = df.dropna(subset=columnas_criticas)
        print(f"✓ Filas con valores nulos eliminadas. Dataset: {len(df)} filas")
    
    # 4. Validar rangos de goles
    goles_negativos = ((df['goles_local'] < 0) | (df['goles_visitante'] < 0)).sum()
    if goles_negativos > 0:
        print(f"⚠ Goles negativos detectados: {goles_negativos}")
        df = df[(df['goles_local'] >= 0) & (df['goles_visitante'] >= 0)]
    
    # 5. Ordenar por fecha
    df = df.sort_values('fecha').reset_index(drop=True)
    print(f"✓ Datos ordenados por fecha. Rango: {df['fecha'].min().date()} a {df['fecha'].max().date()}")
    
    return df


def calcular_promedio_goles_historico(
    df: pd.DataFrame, 
    equipo: str, 
    es_local: bool, 
    indice_actual: int, 
    ventana: int = 5
) -> float:
    """
    Calcula el promedio de goles anotados por un equipo en sus últimos N partidos.
    
    Parámetros:
        df (pd.DataFrame): DataFrame con todos los partidos (debe estar ordenado por fecha).
        equipo (str): Nombre del equipo.
        es_local (bool): True si se calculan goles como local, False como visitante.
        indice_actual (int): Índice del partido actual (no se incluye en el cálculo).
        ventana (int): Número de últimos partidos a considerar (default: 5).
    
    Retorna:
        float: Promedio de goles en los últimos N partidos. 
               Retorna 0.0 si no hay partidos previos.
    """
    if indice_actual == 0:
        return 0.0
    
    # Obtener partidos anteriores
    partidos_previos = df.iloc[:indice_actual].copy()
    
    # Filtrar partidos del equipo (local o visitante)
    if es_local:
        partidos_equipo = partidos_previos[partidos_previos['equipo_local'] == equipo]
        goles = partidos_equipo['goles_local'].values
    else:
        partidos_equipo = partidos_previos[partidos_previos['equipo_visitante'] == equipo]
        goles = partidos_equipo['goles_visitante'].values
    
    # Tomar los últimos N partidos
    goles_ventana = goles[-ventana:]
    
    return float(np.mean(goles_ventana)) if len(goles_ventana) > 0 else 0.0


def calcular_racha_victorias(
    df: pd.DataFrame, 
    equipo: str, 
    es_local: bool, 
    indice_actual: int
) -> int:
    """
    Calcula la racha actual de victorias consecutivas de un equipo.
    
    Una victoria se contabiliza como:
        - Local: goles_local > goles_visitante
        - Visitante: goles_visitante > goles_local
    
    Parámetros:
        df (pd.DataFrame): DataFrame con todos los partidos (debe estar ordenado por fecha).
        equipo (str): Nombre del equipo.
        es_local (bool): True si se calcula racha como local, False como visitante.
        indice_actual (int): Índice del partido actual (no se incluye en el cálculo).
    
    Retorna:
        int: Número de victorias consecutivas (0 si no hay racha o no hay partidos previos).
    """
    if indice_actual == 0:
        return 0
    
    # Obtener partidos anteriores
    partidos_previos = df.iloc[:indice_actual].copy()
    
    # Filtrar partidos del equipo (local o visitante)
    if es_local:
        partidos_equipo = partidos_previos[partidos_previos['equipo_local'] == equipo].copy()
        # Calcular si ganó (goles_local > goles_visitante)
        partidos_equipo['gano'] = partidos_equipo['goles_local'] > partidos_equipo['goles_visitante']
    else:
        partidos_equipo = partidos_previos[partidos_previos['equipo_visitante'] == equipo].copy()
        # Calcular si ganó (goles_visitante > goles_local)
        partidos_equipo['gano'] = partidos_equipo['goles_visitante'] > partidos_equipo['goles_local']
    
    if len(partidos_equipo) == 0:
        return 0
    
    # Iterar desde el último partido hacia atrás contando victorias
    racha = 0
    for idx in range(len(partidos_equipo) - 1, -1, -1):
        if partidos_equipo.iloc[idx]['gano']:
            racha += 1
        else:
            break
    
    return racha


def crear_features_historicas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features históricas (engineering) basadas en el desempeño previo de los equipos.
    
    Features creadas:
        - promedio_goles_local: Promedio de goles del equipo local en sus últimos 5 partidos como local.
        - promedio_goles_visitante: Promedio de goles del equipo visitante en sus últimos 5 partidos como visitante.
        - racha_local: Victorias consecutivas del equipo local como local.
        - racha_visitante: Victorias consecutivas del equipo visitante como visitante.
        - ventaja_local: Indicador binario (siempre 1) representando la ventaja de jugar en casa.
    
    Parámetros:
        df (pd.DataFrame): DataFrame limpio de partidos (debe estar ordenado por fecha).
    
    Retorna:
        pd.DataFrame: DataFrame con las nuevas features añadidas.
    """
    df = df.copy()
    
    # Inicializar columnas
    df['promedio_goles_local'] = 0.0
    df['promedio_goles_visitante'] = 0.0
    df['racha_local'] = 0
    df['racha_visitante'] = 0
    df['ventaja_local'] = 1
    
    print("Calculando features históricas...")
    
    # Calcular features para cada partido
    for idx in range(len(df)):
        # Promedio de goles local
        df.loc[idx, 'promedio_goles_local'] = calcular_promedio_goles_historico(
            df, 
            df.loc[idx, 'equipo_local'], 
            es_local=True, 
            indice_actual=idx, 
            ventana=5
        )
        
        # Promedio de goles visitante
        df.loc[idx, 'promedio_goles_visitante'] = calcular_promedio_goles_historico(
            df, 
            df.loc[idx, 'equipo_visitante'], 
            es_local=False, 
            indice_actual=idx, 
            ventana=5
        )
        
        # Racha de victorias local
        df.loc[idx, 'racha_local'] = calcular_racha_victorias(
            df, 
            df.loc[idx, 'equipo_local'], 
            es_local=True, 
            indice_actual=idx
        )
        
        # Racha de victorias visitante
        df.loc[idx, 'racha_visitante'] = calcular_racha_victorias(
            df, 
            df.loc[idx, 'equipo_visitante'], 
            es_local=False, 
            indice_actual=idx
        )
        
        if (idx + 1) % max(1, len(df) // 10) == 0:
            print(f"  → Procesados {idx + 1}/{len(df)} partidos...")
    
    print(f"✓ Features históricas calculadas para {len(df)} partidos")
    return df


def crear_variable_objetivo(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la variable objetivo "resultado" basada en el resultado del partido.
    
    Clasificación:
        - "Local": Si goles_local > goles_visitante
        - "Empate": Si goles_local == goles_visitante
        - "Visitante": Si goles_local < goles_visitante
    
    Parámetros:
        df (pd.DataFrame): DataFrame con columnas goles_local y goles_visitante.
    
    Retorna:
        pd.DataFrame: DataFrame con la nueva columna "resultado".
    """
    df = df.copy()
    
    def determinar_resultado(fila):
        if fila['goles_local'] > fila['goles_visitante']:
            return "Local"
        elif fila['goles_local'] == fila['goles_visitante']:
            return "Empate"
        else:
            return "Visitante"
    
    df['resultado'] = df.apply(determinar_resultado, axis=1)
    
    # Estadísticas de la variable objetivo
    conteos = df['resultado'].value_counts()
    print(f"✓ Variable objetivo 'resultado' creada:")
    for clase, conteo in conteos.items():
        pct = (conteo / len(df)) * 100
        print(f"  - {clase}: {conteo} ({pct:.1f}%)")
    
    return df


def preparar_datos(db_path: str) -> pd.DataFrame:
    """
    Función orquestadora principal que ejecuta todo el pipeline de preparación de datos.
    
    Pipeline:
        1. Carga datos desde SQLite (tabla fixtures).
        2. Limpia y valida datos (nulos, tipos, duplicados).
        3. Crea features históricas basadas en el desempeño previo.
        4. Crea la variable objetivo.
        5. Retorna el dataset enriquecido y listo para modelado.
    
    Parámetros:
        db_path (str): Ruta al archivo SQLite (ej: 'premier_league.db').
    
    Retorna:
        pd.DataFrame: Dataset limpio y enriquecido con columnas:
            - fecha, equipo_local, equipo_visitante
            - goles_local, goles_visitante
            - promedio_goles_local, promedio_goles_visitante
            - racha_local, racha_visitante
            - ventaja_local
            - resultado (variable objetivo)
    
    Ejemplo:
        >>> df_preparado = preparar_datos('premier_league.db')
        >>> print(df_preparado.shape)
        >>> print(df_preparado.head())
    """
    print("=" * 70)
    print("INICIANDO PIPELINE DE PREPARACIÓN DE DATOS")
    print("=" * 70)
    print()
    
    # 1. Cargar datos
    print("[1/5] Cargando datos...")
    df = cargar_datos_sqlite(db_path)
    print()
    
    # 2. Limpiar datos
    print("[2/5] Limpiando datos...")
    df = limpiar_datos(df)
    print()
    
    # 3. Crear features históricas
    print("[3/5] Creando features históricas...")
    df = crear_features_historicas(df)
    print()
    
    # 4. Crear variable objetivo
    print("[4/5] Creando variable objetivo...")
    df = crear_variable_objetivo(df)
    print()
    
    # 5. Resumen final
    print("[5/5] Generando resumen final...")
    print("-" * 70)
    print(f"Dataset final: {len(df)} partidos × {len(df.columns)} columnas")
    print(f"\nColumnas del dataset:")
    for col in df.columns:
        print(f"  - {col}: {df[col].dtype}")
    print("-" * 70)
    print("✓ PIPELINE COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    
    return df


if __name__ == "__main__":
    # Ejemplo de uso
    df_preparado = preparar_datos('premier_league.db')
    print("\n📊 Primeras 5 filas del dataset preparado:")
    print(df_preparado.head())
    print("\n📈 Información estadística:")
    print(df_preparado.describe())
