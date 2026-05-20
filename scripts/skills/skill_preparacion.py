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


def cargar_datos(db_path):
    import sqlite3
    import pandas as pd
    
    query = """
    SELECT 
        "fixture.id",
        "fixture.date",
        "teams.home.name",
        "teams.away.name",
        "goals.home",
        "goals.away",
        "fixture.status.short"
    FROM fixtures
    WHERE "fixture.status.short" = 'FT'
    """
    
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df = df.rename(columns={
        'fixture.id': 'id',
        'fixture.date': 'fecha',
        'teams.home.name': 'equipo_local',
        'teams.away.name': 'equipo_visitante',
        'goals.home': 'goles_local',
        'goals.away': 'goles_visitante'
    })
    
    return df


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    duplicados_iniciales = df.duplicated().sum()
    df = df.drop_duplicates()
    print(f"✓ Duplicados eliminados: {duplicados_iniciales}")
    
    try:
        df['fecha'] = pd.to_datetime(df['fecha'])
        df['goles_local'] = df['goles_local'].astype(int)
        df['goles_visitante'] = df['goles_visitante'].astype(int)
        df['equipo_local'] = df['equipo_local'].astype(str).str.strip()
        df['equipo_visitante'] = df['equipo_visitante'].astype(str).str.strip()
        df['id'] = df['id'].astype(int)
        print("✓ Tipos de datos convertidos correctamente")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Error al convertir tipos de datos: {str(e)}")
    
    columnas_criticas = ['fecha', 'equipo_local', 'equipo_visitante', 'goles_local', 'goles_visitante']
    nulos = df[columnas_criticas].isnull().sum()
    if nulos.any():
        print(f"⚠ Valores nulos detectados:\n{nulos[nulos > 0]}")
        df = df.dropna(subset=columnas_criticas)
        print(f"✓ Filas con valores nulos eliminadas. Dataset: {len(df)} filas")
    
    goles_negativos = ((df['goles_local'] < 0) | (df['goles_visitante'] < 0)).sum()
    if goles_negativos > 0:
        print(f"⚠ Goles negativos detectados: {goles_negativos}")
        df = df[(df['goles_local'] >= 0) & (df['goles_visitante'] >= 0)]
    
    df = df.sort_values('fecha').reset_index(drop=True)
    print(f"✓ Datos ordenados por fecha. Rango: {df['fecha'].min().date()} a {df['fecha'].max().date()}")
    
    return df


def calcular_promedio_goles_local(df: pd.DataFrame, equipo: str, fecha) -> float:
    partidos_anteriores = df[
        (df['equipo_local'] == equipo) & 
        (df['fecha'] < fecha)
    ].sort_values('fecha').tail(5)
    if len(partidos_anteriores) == 0:
        return 0.0
    return float(partidos_anteriores['goles_local'].mean())


def calcular_promedio_goles_visitante(df: pd.DataFrame, equipo: str, fecha) -> float:
    partidos_anteriores = df[
        (df['equipo_visitante'] == equipo) & 
        (df['fecha'] < fecha)
    ].sort_values('fecha').tail(5)
    if len(partidos_anteriores) == 0:
        return 0.0
    return float(partidos_anteriores['goles_visitante'].mean())


def calcular_racha_local(df, equipo, fecha):
    partidos_anteriores = df[
        (df['equipo_local'] == equipo) & 
        (df['fecha'] < fecha)
    ].sort_values('fecha').tail(5)
    if len(partidos_anteriores) == 0:
        return 0
    victorias = (partidos_anteriores['goles_local'] > partidos_anteriores['goles_visitante']).sum()
    return int(victorias)


def calcular_racha_visitante(df, equipo, fecha):
    partidos_anteriores = df[
        (df['equipo_visitante'] == equipo) & 
        (df['fecha'] < fecha)
    ].sort_values('fecha').tail(5)
    if len(partidos_anteriores) == 0:
        return 0
    victorias = (partidos_anteriores['goles_visitante'] > partidos_anteriores['goles_local']).sum()
    return int(victorias)


def crear_features_historicas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df['promedio_goles_local'] = 0.0
    df['promedio_goles_visitante'] = 0.0
    df['racha_local'] = 0
    df['racha_visitante'] = 0
    df['ventaja_local'] = 1
    
    print("Calculando features históricas...")
    
    df = df.reset_index(drop=True)
    for idx in df.index:
        fecha = df.at[idx, 'fecha']
        equipo_local = df.at[idx, 'equipo_local']
        equipo_visitante = df.at[idx, 'equipo_visitante']
        
        df.at[idx, 'promedio_goles_local'] = calcular_promedio_goles_local(df, equipo_local, fecha)
        df.at[idx, 'promedio_goles_visitante'] = calcular_promedio_goles_visitante(df, equipo_visitante, fecha)
        df.at[idx, 'racha_local'] = calcular_racha_local(df, equipo_local, fecha)
        df.at[idx, 'racha_visitante'] = calcular_racha_visitante(df, equipo_visitante, fecha)
        
        if (idx + 1) % max(1, len(df) // 10) == 0:
            print(f"  → Procesados {idx + 1}/{len(df)} partidos...")
    
    print(f"✓ Features históricas calculadas para {len(df)} partidos")
    return df


def crear_variable_objetivo(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    def determinar_resultado(fila):
        if fila['goles_local'] > fila['goles_visitante']:
            return "Local"
        elif fila['goles_local'] == fila['goles_visitante']:
            return "Empate"
        else:
            return "Visitante"
    
    df['resultado'] = df.apply(determinar_resultado, axis=1)
    
    conteos = df['resultado'].value_counts()
    print(f"✓ Variable objetivo 'resultado' creada:")
    for clase, conteo in conteos.items():
        pct = (conteo / len(df)) * 100
        print(f"  - {clase}: {conteo} ({pct:.1f}%)")
    
    return df


def preparar_datos(db_path: str) -> pd.DataFrame:
    """
    Función orquestadora principal que ejecuta todo el pipeline de preparación de datos.
    """
    print("=" * 70)
    print("INICIANDO PIPELINE DE PREPARACIÓN DE DATOS")
    print("=" * 70)
    print()
    
    print("[1/5] Cargando datos...")
    df = cargar_datos(db_path)
    print()
    
    print("[2/5] Limpiando datos...")
    df = limpiar_datos(df)
    print()
    
    print("[3/5] Creando features históricas...")
    df = crear_features_historicas(df)
    print()
    
    print("[4/5] Creando variable objetivo...")
    df = crear_variable_objetivo(df)
    print()
    
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
    df_preparado = preparar_datos('premier_league.db')
    print("\n📊 Primeras 5 filas del dataset preparado:")
    print(df_preparado.head())
    print("\n📈 Información estadística:")
    print(df_preparado.describe())