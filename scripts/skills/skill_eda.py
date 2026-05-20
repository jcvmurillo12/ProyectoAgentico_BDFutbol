"""
Skill de Análisis Exploratorio de Datos para Sistema Agéntico de Premier League.

Este módulo contiene la lógica de análisis exploratorio del dataset de partidos
de la Premier League, incluyendo estadísticas descriptivas, análisis de distribuciones,
correlaciones y visualizaciones.

Funciones principales:
    - ejecutar_eda(df): Función orquestadora que ejecuta todo el análisis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple, List
from datetime import datetime
import os


def calcular_estadisticas_descriptivas(df: pd.DataFrame) -> Dict[str, any]:
    """
    Calcula estadísticas descriptivas generales del dataset.
    
    Parámetros:
        df (pd.DataFrame): DataFrame con los datos de partidos.
    
    Retorna:
        Dict: Diccionario con estadísticas:
            - cantidad_partidos: número total de partidos
            - rango_fechas: tupla (fecha_inicio, fecha_fin)
            - cantidad_equipos: número único de equipos
            - promedio_goles_por_partido: promedio de goles totales
    """
    estadisticas = {
        'cantidad_partidos': len(df),
        'rango_fechas': (df['fecha'].min(), df['fecha'].max()),
        'cantidad_equipos': len(set(df['equipo_local'].unique()) | set(df['equipo_visitante'].unique())),
        'promedio_goles_por_partido': ((df['goles_local'] + df['goles_visitante']).mean()),
        'promedio_goles_local': df['goles_local'].mean(),
        'promedio_goles_visitante': df['goles_visitante'].mean()
    }
    
    print("📊 ESTADÍSTICAS DESCRIPTIVAS GENERALES")
    print("-" * 70)
    print(f"Total de partidos: {estadisticas['cantidad_partidos']}")
    print(f"Rango de fechas: {estadisticas['rango_fechas'][0].date()} a {estadisticas['rango_fechas'][1].date()}")
    print(f"Cantidad de equipos únicos: {estadisticas['cantidad_equipos']}")
    print(f"Promedio de goles por partido: {estadisticas['promedio_goles_por_partido']:.2f}")
    print(f"Promedio de goles local: {estadisticas['promedio_goles_local']:.2f}")
    print(f"Promedio de goles visitante: {estadisticas['promedio_goles_visitante']:.2f}")
    print()
    
    return estadisticas


def analizar_distribucion_resultados(df: pd.DataFrame) -> Dict[str, any]:
    """
    Analiza la distribución de la variable objetivo "resultado".
    
    Parámetros:
        df (pd.DataFrame): DataFrame con columna 'resultado'.
    
    Retorna:
        Dict: Diccionario con:
            - conteos: valor_counts de resultado
            - porcentajes: porcentaje de cada clase
            - distribuciones: detalle de la distribución
    """
    conteos = df['resultado'].value_counts()
    total = len(df)
    
    distribucion = {
        'conteos': conteos.to_dict(),
        'porcentajes': {clase: (conteo / total * 100) for clase, conteo in conteos.items()},
        'distribuciones': {}
    }
    
    print("⚽ DISTRIBUCIÓN DE LA VARIABLE OBJETIVO")
    print("-" * 70)
    for clase in ['Local', 'Empate', 'Visitante']:
        if clase in conteos.index:
            conteo = conteos[clase]
            pct = (conteo / total) * 100
            distribucion['distribuciones'][clase] = f"{conteo} partidos ({pct:.1f}%)"
            print(f"{clase:12} : {conteo:5} partidos ({pct:5.1f}%)")
        else:
            distribucion['distribuciones'][clase] = "0 partidos (0.0%)"
            print(f"{clase:12} : 0 partidos (0.0%)")
    print()
    
    return distribucion


def analizar_equipos_victoria_local(df: pd.DataFrame, top_n: int = 5) -> Dict[str, List[Tuple]]:
    """
    Identifica los N equipos con más victorias como local.
    
    Parámetros:
        df (pd.DataFrame): DataFrame con datos de partidos.
        top_n (int): Número de equipos a mostrar (default: 5).
    
    Retorna:
        Dict: Diccionario con:
            - top_equipos_local: lista de tuplas (equipo, victorias, total_locales)
            - top_visitante: lista de tuplas (equipo, victorias, total_visitantes)
    """
    # Victorias como local
    partidos_locales = df[df['goles_local'] > df['goles_visitante']]
    victorias_local = partidos_locales['equipo_local'].value_counts().head(top_n)
    
    # Total de partidos como local para cada equipo
    total_locales = df['equipo_local'].value_counts()
    
    top_equipos_local = [
        (equipo, int(victorias), int(total_locales[equipo]))
        for equipo, victorias in victorias_local.items()
    ]
    
    print(f"🏆 TOP {top_n} EQUIPOS CON MÁS VICTORIAS COMO LOCAL")
    print("-" * 70)
    for i, (equipo, victorias, total) in enumerate(top_equipos_local, 1):
        porcentaje = (victorias / total) * 100 if total > 0 else 0
        print(f"{i}. {equipo:20} : {victorias:2} victorias de {total:2} locales ({porcentaje:5.1f}%)")
    print()
    
    return {'top_equipos_local': top_equipos_local}


def analizar_equipos_victoria_visitante(df: pd.DataFrame, top_n: int = 5) -> Dict[str, List[Tuple]]:
    """
    Identifica los N equipos con más victorias como visitante.
    
    Parámetros:
        df (pd.DataFrame): DataFrame con datos de partidos.
        top_n (int): Número de equipos a mostrar (default: 5).
    
    Retorna:
        Dict: Diccionario con:
            - top_equipos_visitante: lista de tuplas (equipo, victorias, total_visitantes)
    """
    # Victorias como visitante
    partidos_visitantes = df[df['goles_visitante'] > df['goles_local']]
    victorias_visitante = partidos_visitantes['equipo_visitante'].value_counts().head(top_n)
    
    # Total de partidos como visitante para cada equipo
    total_visitantes = df['equipo_visitante'].value_counts()
    
    top_equipos_visitante = [
        (equipo, int(victorias), int(total_visitantes[equipo]))
        for equipo, victorias in victorias_visitante.items()
    ]
    
    print(f"🚀 TOP {top_n} EQUIPOS CON MÁS VICTORIAS COMO VISITANTE")
    print("-" * 70)
    for i, (equipo, victorias, total) in enumerate(top_equipos_visitante, 1):
        porcentaje = (victorias / total) * 100 if total > 0 else 0
        print(f"{i}. {equipo:20} : {victorias:2} victorias de {total:2} visitantes ({porcentaje:5.1f}%)")
    print()
    
    return {'top_equipos_visitante': top_equipos_visitante}


def calcular_matriz_correlacion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la matriz de correlación entre features numéricas.
    
    Features consideradas:
        - goles_local, goles_visitante
        - promedio_goles_local, promedio_goles_visitante
        - racha_local, racha_visitante
        - ventaja_local
    
    Parámetros:
        df (pd.DataFrame): DataFrame con features numéricas.
    
    Retorna:
        pd.DataFrame: Matriz de correlación de Pearson.
    """
    # Seleccionar solo columnas numéricas relevantes
    columnas_numericas = [
        'goles_local', 'goles_visitante',
        'promedio_goles_local', 'promedio_goles_visitante',
        'racha_local', 'racha_visitante',
        'ventaja_local'
    ]
    
    matriz_corr = df[columnas_numericas].corr()
    
    print("📈 CORRELACIONES ENTRE FEATURES NUMÉRICAS")
    print("-" * 70)
    print(matriz_corr.to_string())
    print()
    
    return matriz_corr


def generar_grafica_distribucion_resultados(df: pd.DataFrame, figsize: Tuple[int, int] = (10, 6)) -> None:
    """
    Genera gráfica 1: Distribución de resultados (countplot).
    
    Parámetros:
        df (pd.DataFrame): DataFrame con columna 'resultado'.
        figsize (Tuple): Tamaño de la figura (ancho, alto).
    """
    # Crear carpeta si no existe
    os.makedirs('graphs', exist_ok=True)
    
    plt.figure(figsize=figsize)
    
    # Orden específico para las clases
    orden_clases = ['Local', 'Empate', 'Visitante']
    colores = ['#2ecc71', '#f39c12', '#e74c3c']  # Verde, naranja, rojo
    
    ax = sns.countplot(
        data=df,
        x='resultado',
        order=orden_clases,
        palette=colores,
        edgecolor='black',
        linewidth=1.5
    )
    
    plt.title('Distribución de Resultados de Partidos', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Resultado', fontsize=12, fontweight='bold')
    plt.ylabel('Cantidad de Partidos', fontsize=12, fontweight='bold')
    
    # Añadir valores en las barras
    for container in ax.containers:
        ax.bar_label(container, fontsize=11, fontweight='bold')
    
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('graphs/01_distribucion_resultados.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfica 1 guardada: graphs/01_distribucion_resultados.png")
    plt.close()


def generar_grafica_promedio_goles_por_equipo(df: pd.DataFrame, top_n: int = 10, figsize: Tuple[int, int] = (14, 7)) -> None:
    """
    Genera gráfica 2: Promedio de goles local vs visitante por equipo (top N equipos).
    
    Parámetros:
        df (pd.DataFrame): DataFrame con datos de partidos.
        top_n (int): Número de equipos a mostrar (default: 10).
        figsize (Tuple): Tamaño de la figura (ancho, alto).
    """
    # Calcular promedios por equipo
    goles_local_por_equipo = df.groupby('equipo_local')['goles_local'].mean().sort_values(ascending=False).head(top_n)
    goles_visitante_por_equipo = df.groupby('equipo_visitante')['goles_visitante'].mean().sort_values(ascending=False).head(top_n)
    
    # Obtener los equipos top (unión de ambos)
    equipos_top = list(set(goles_local_por_equipo.index) | set(goles_visitante_por_equipo.index))[:top_n]
    
    # Preparar datos
    datos_plot = []
    for equipo in equipos_top:
        prom_local = df[df['equipo_local'] == equipo]['goles_local'].mean() if equipo in df['equipo_local'].values else 0
        prom_visitante = df[df['equipo_visitante'] == equipo]['goles_visitante'].mean() if equipo in df['equipo_visitante'].values else 0
        datos_plot.append({
            'Equipo': equipo,
            'Promedio Goles Local': prom_local,
            'Promedio Goles Visitante': prom_visitante
        })
    
    df_plot = pd.DataFrame(datos_plot).sort_values('Promedio Goles Local', ascending=True)
    
    # Crear gráfica
    fig, ax = plt.subplots(figsize=figsize)
    
    y_pos = np.arange(len(df_plot))
    width = 0.35
    
    ax.barh(y_pos - width/2, df_plot['Promedio Goles Local'], width, label='Goles Local', color='#3498db', edgecolor='black', linewidth=1)
    ax.barh(y_pos + width/2, df_plot['Promedio Goles Visitante'], width, label='Goles Visitante', color='#e74c3c', edgecolor='black', linewidth=1)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['Equipo'])
    ax.set_xlabel('Promedio de Goles por Partido', fontsize=12, fontweight='bold')
    ax.set_title(f'Promedio de Goles: Local vs Visitante (Top {top_n} Equipos)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('graphs/02_promedio_goles_equipos.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfica 2 guardada: graphs/02_promedio_goles_equipos.png")
    plt.close()


def generar_heatmap_correlacion(df: pd.DataFrame, matriz_corr: pd.DataFrame, figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Genera gráfica 3: Heatmap de correlación entre features numéricas.
    
    Parámetros:
        df (pd.DataFrame): DataFrame original (no usado pero mantenido para consistencia).
        matriz_corr (pd.DataFrame): Matriz de correlación precomputada.
        figsize (Tuple): Tamaño de la figura (ancho, alto).
    """
    plt.figure(figsize=figsize)
    
    # Crear heatmap
    sns.heatmap(
        matriz_corr,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=1,
        cbar_kws={'label': 'Correlación'},
        vmin=-1,
        vmax=1
    )
    
    plt.title('Matriz de Correlación entre Features Numéricas', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig('graphs/03_heatmap_correlacion.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfica 3 guardada: graphs/03_heatmap_correlacion.png")
    plt.close()


def generar_reporte_eda(
    estadisticas: Dict,
    distribucion: Dict,
    equipos_local: Dict,
    equipos_visitante: Dict,
    matriz_corr: pd.DataFrame
) -> Dict[str, str]:
    """
    Genera un reporte textual consolidado con los hallazgos principales del EDA.
    
    Parámetros:
        estadisticas (Dict): Diccionario con estadísticas descriptivas.
        distribucion (Dict): Diccionario con distribución de resultados.
        equipos_local (Dict): Diccionario con top equipos locales.
        equipos_visitante (Dict): Diccionario con top equipos visitantes.
        matriz_corr (pd.DataFrame): Matriz de correlación.
    
    Retorna:
        Dict: Diccionario "reporte_eda" con hallazgos en formato texto.
    """
    reporte = {}
    
    # Hallazgo 1: Estadísticas generales
    reporte['estadisticas_generales'] = (
        f"El dataset contiene {estadisticas['cantidad_partidos']} partidos jugados entre "
        f"{estadisticas['cantidad_equipos']} equipos únicos en el rango "
        f"{estadisticas['rango_fechas'][0].date()} a {estadisticas['rango_fechas'][1].date()}. "
        f"El promedio de goles por partido es {estadisticas['promedio_goles_por_partido']:.2f}, "
        f"con {estadisticas['promedio_goles_local']:.2f} goles en promedio para locales "
        f"y {estadisticas['promedio_goles_visitante']:.2f} para visitantes."
    )
    
    # Hallazgo 2: Distribución de resultados
    dist_str = ""
    for clase in ['Local', 'Empate', 'Visitante']:
        if clase in distribucion['conteos']:
            conteo = distribucion['conteos'][clase]
            pct = distribucion['porcentajes'][clase]
            dist_str += f"{clase}: {conteo} ({pct:.1f}%), "
    dist_str = dist_str.rstrip(", ")
    
    reporte['distribucion_resultados'] = (
        f"La variable objetivo 'resultado' muestra: {dist_str}. "
        f"Los equipos locales tienen una tasa de victoria del "
        f"{distribucion['porcentajes'].get('Local', 0):.1f}%, indicando una ventaja significativa del factor local."
    )
    
    # Hallazgo 3: Equipos dominantes
    top_local = equipos_local['top_equipos_local'][0]
    top_visitante = equipos_visitante['top_equipos_visitante'][0]
    
    reporte['equipos_dominantes'] = (
        f"El equipo más ganador como local es {top_local[0]} con {top_local[1]} victorias "
        f"de {top_local[2]} partidos ({(top_local[1]/top_local[2]*100):.1f}%). "
        f"Como visitante, {top_visitante[0]} lidera con {top_visitante[1]} victorias "
        f"de {top_visitante[2]} partidos ({(top_visitante[1]/top_visitante[2]*100):.1f}%)."
    )
    
    # Hallazgo 4: Correlaciones principales
    # Encontrar las correlaciones más altas (excluyendo diagonal)
    corr_altas = []
    for i in range(len(matriz_corr.columns)):
        for j in range(i+1, len(matriz_corr.columns)):
            valor = matriz_corr.iloc[i, j]
            if abs(valor) > 0.5:  # Correlaciones moderadas/altas
                corr_altas.append((matriz_corr.columns[i], matriz_corr.columns[j], valor))
    
    corr_altas.sort(key=lambda x: abs(x[2]), reverse=True)
    
    if corr_altas:
        corr_texto = ", ".join([f"{f1}-{f2}: {v:.2f}" for f1, f2, v in corr_altas[:3]])
        reporte['correlaciones_principales'] = (
            f"Se detectan correlaciones relevantes entre features: {corr_texto}. "
            f"Esto sugiere que las características históricas están moderadamente relacionadas con el desempeño."
        )
    else:
        reporte['correlaciones_principales'] = (
            "Las correlaciones entre features son generalmente bajas (< 0.5), "
            "indicando que las variables predictoras son relativamente independientes."
        )
    
    # Hallazgo 5: Conclusión
    reporte['conclusion'] = (
        "El análisis exploratorio revela que existe un efecto de campo significativo en la Premier League, "
        "con una clara ventaja para los equipos locales. Las features de desempeño histórico (promedios y rachas) "
        "presentan variabilidad moderada entre equipos, lo que sugiere potencial para predictibilidad en modelos de machine learning."
    )
    
    return reporte


def ejecutar_eda(df: pd.DataFrame) -> Dict[str, str]:
    """
    Función orquestadora principal que ejecuta todo el análisis exploratorio de datos.
    
    Pipeline:
        1. Calcula estadísticas descriptivas generales.
        2. Analiza la distribución de la variable objetivo.
        3. Identifica equipos con más victorias como local y visitante.
        4. Calcula matriz de correlación entre features.
        5. Genera 3 gráficas visuales (distribución, promedios, correlación).
        6. Compila reporte con hallazgos principales.
    
    Parámetros:
        df (pd.DataFrame): DataFrame limpio y enriquecido de la Skill 1 (preparacion),
                          con columnas: id, fecha, equipo_local, equipo_visitante,
                          goles_local, goles_visitante, promedio_goles_local,
                          promedio_goles_visitante, racha_local, racha_visitante,
                          ventaja_local, resultado.
    
    Retorna:
        Dict: Diccionario "reporte_eda" con hallazgos principales en texto:
            - estadisticas_generales: overview del dataset
            - distribucion_resultados: análisis de clases objetivo
            - equipos_dominantes: equipos con mejor desempeño
            - correlaciones_principales: relaciones entre variables
            - conclusion: síntesis del análisis
    
    Archivos generados:
        - graphs/01_distribucion_resultados.png
        - graphs/02_promedio_goles_equipos.png
        - graphs/03_heatmap_correlacion.png
    
    Ejemplo:
        >>> df_preparado = preparar_datos('data/premier_league.db')
        >>> reporte_eda = ejecutar_eda(df_preparado)
        >>> print(reporte_eda['conclusion'])
    """
    print("=" * 70)
    print("INICIANDO ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
    print("=" * 70)
    print()
    
    # 1. Estadísticas descriptivas
    print("[1/6] Calculando estadísticas descriptivas...")
    estadisticas = calcular_estadisticas_descriptivas(df)
    
    # 2. Distribución de resultados
    print("[2/6] Analizando distribución de resultados...")
    distribucion = analizar_distribucion_resultados(df)
    
    # 3. Equipos con más victorias como local
    print("[3/6] Analizando equipos dominantes como local...")
    equipos_local = analizar_equipos_victoria_local(df, top_n=5)
    
    # 4. Equipos con más victorias como visitante
    print("[4/6] Analizando equipos dominantes como visitante...")
    equipos_visitante = analizar_equipos_victoria_visitante(df, top_n=5)
    
    # 5. Matriz de correlación
    print("[5/6] Calculando matriz de correlación...")
    matriz_corr = calcular_matriz_correlacion(df)
    print()
    
    # 6. Generar gráficas
    print("[6/6] Generando visualizaciones...")
    print("-" * 70)
    generar_grafica_distribucion_resultados(df)
    generar_grafica_promedio_goles_por_equipo(df, top_n=10)
    generar_heatmap_correlacion(df, matriz_corr)
    print("-" * 70)
    print()
    
    # 7. Generar reporte
    print("📋 GENERANDO REPORTE EDA...")
    print("-" * 70)
    reporte_eda = generar_reporte_eda(estadisticas, distribucion, equipos_local, equipos_visitante, matriz_corr)
    
    for clave, contenido in reporte_eda.items():
        print(f"\n{clave.upper()}:")
        print(contenido)
    
    print()
    print("=" * 70)
    print("✓ ANÁLISIS EXPLORATORIO COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    
    return reporte_eda


if __name__ == "__main__":
    # Ejemplo de uso (requiere que exista el archivo de datos)
    from scripts.skills.skill_preparacion import preparar_datos
    
    df_preparado = preparar_datos('premier_league.db')
    reporte_eda = ejecutar_eda(df_preparado)
    
    print("\n📄 Reporte EDA generado:")
    for clave, valor in reporte_eda.items():
        print(f"\n{clave}:\n{valor}")
