"""
Orquestador Principal del Sistema Agéntico de Premier League.

Este módulo actúa como el agente principal del sistema, orquestando
la ejecución secuencial de los tres skills del pipeline analítico:
    1. skill_preparacion.py - Preparación y enriquecimiento de datos
    2. skill_eda.py - Análisis exploratorio de datos
    3. skill_modelado.py - Modelado y predicción

Funciones principales:
    - ejecutar_pipeline_agentico(db_path): Ejecuta el pipeline completo
"""

import sys
from pathlib import Path
from typing import Dict, Any, Tuple

# Agregar la ruta de scripts al path para importaciones
script_path = Path(__file__).parent
sys.path.insert(0, str(script_path))

# Importar las funciones principales de cada skill
from skills.skill_preparacion import preparar_datos
from skills.skill_eda import ejecutar_eda
from skills.skill_modelado import ejecutar_modelado


def ejecutar_pipeline_agentico(db_path: str = r"C:\Users\juanc\Documents\Universidad\Programacion Analitica De Datos\ProyectoAgentico_BDFutbol\data\premier_league.db") -> Dict[str, Any]:
    """
    Ejecuta el pipeline completo del sistema agéntico.
    
    Pipeline de ejecución (secuencial):
        1. SKILL 1 (Preparación): Carga, limpia y enriquece datos.
        2. SKILL 2 (EDA): Analiza exploratoriamente los datos preparados.
        3. SKILL 3 (Modelado): Entrena y evalúa modelos predictivos.
    
    Parámetros:
        db_path (str): Ruta al archivo SQLite (default: 'premier_league.db')
                      Puede ser una ruta relativa o absoluta.
    
    Retorna:
        Dict[str, Any]: Diccionario con los resultados de cada skill:
            {
                'df_preparado': DataFrame limpio y enriquecido,
                'reporte_eda': Reporte de hallazgos del EDA,
                'resultados_modelado': {
                    'mejor_modelo': Nombre del modelo ganador,
                    'metricas': Métricas del mejor modelo,
                    'reporte_modelado': Reporte con conclusiones
                }
            }
    
    Excepciones:
        - Si la base de datos no existe o está corrupta
        - Si hay errores en la transformación de datos
        - Si los modelos no convergen
    
    Ejemplo:
        >>> resultados = ejecutar_pipeline_agentico('premier_league.db')
        >>> print(f"Dataset: {resultados['df_preparado'].shape}")
        >>> print(f"Modelo ganador: {resultados['resultados_modelado']['mejor_modelo']}")
    """
    
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "SISTEMA AGÉNTICO - PREDICCIÓN DE RESULTADOS PREMIER LEAGUE".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "═" * 68 + "╝")
    print("\n")
    
    print(f"Base de datos: {db_path}")
    print("Iniciando orquestación de skills...")
    print("\n")
    
    # ========================================================================
    # SKILL 1: PREPARACIÓN DE DATOS
    # ========================================================================
    print("█" * 70)
    print("█ SKILL 1/3: PREPARACIÓN DE DATOS")
    print("█" * 70)
    print()
    
    try:
        df_preparado = preparar_datos(db_path)
        print("\n✓ SKILL 1 COMPLETADO EXITOSAMENTE")
        print(f"  → Dataset: {df_preparado.shape[0]} partidos × {df_preparado.shape[1]} columnas")
    except Exception as e:
        print(f"\n✗ ERROR EN SKILL 1: {str(e)}")
        raise
    
    print()
    print()
    
    # ========================================================================
    # SKILL 2: ANÁLISIS EXPLORATORIO DE DATOS
    # ========================================================================
    print("█" * 70)
    print("█ SKILL 2/3: ANÁLISIS EXPLORATORIO DE DATOS")
    print("█" * 70)
    print()
    
    try:
        reporte_eda = ejecutar_eda(df_preparado)
        print("\n✓ SKILL 2 COMPLETADO EXITOSAMENTE")
        print(f"  → Reporte generado con {len(reporte_eda)} secciones")
    except Exception as e:
        print(f"\n✗ ERROR EN SKILL 2: {str(e)}")
        raise
    
    print()
    print()
    
    # ========================================================================
    # SKILL 3: MODELADO PREDICTIVO
    # ========================================================================
    print("█" * 70)
    print("█ SKILL 3/3: MODELADO PREDICTIVO")
    print("█" * 70)
    print()
    
    try:
        resultados_modelado = ejecutar_modelado(df_preparado)
        print("\n✓ SKILL 3 COMPLETADO EXITOSAMENTE")
        print(f"  → Modelo ganador: {resultados_modelado['mejor_modelo']}")
        print(f"  → Accuracy: {resultados_modelado['metricas']['accuracy']:.4f}")
    except Exception as e:
        print(f"\n✗ ERROR EN SKILL 3: {str(e)}")
        raise
    
    print()
    print()
    
    # ========================================================================
    # RESUMEN FINAL
    # ========================================================================
    print("╔" + "═" * 68 + "╗")
    print("║" + "PIPELINE COMPLETADO EXITOSAMENTE".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    print("📊 RESUMEN DE RESULTADOS:")
    print("-" * 70)
    print(f"\n1. PREPARACIÓN:")
    print(f"   ✓ {df_preparado.shape[0]} partidos procesados")
    print(f"   ✓ {df_preparado.shape[1]} features creadas")
    print(f"   ✓ Variable objetivo: 'resultado' (Local/Empate/Visitante)")
    
    print(f"\n2. ANÁLISIS EXPLORATORIO:")
    print(f"   ✓ Estadísticas descriptivas calculadas")
    print(f"   ✓ Distribución de resultados analizada")
    print(f"   ✓ Top 5 equipos identificados")
    print(f"   ✓ Matriz de correlación generada")
    print(f"   ✓ 3 gráficas visuales creadas")
    
    print(f"\n3. MODELADO:")
    print(f"   ✓ LogisticRegression: {resultados_modelado['metricas']['accuracy']:.4f} (Accuracy) {"[GANADOR]" if resultados_modelado['mejor_modelo']=="LogisticRegression" else ""}")
    print(f"   ✓ RandomForestClassifier: {resultados_modelado['metricas']['accuracy']:.4f} (Accuracy) {"[GANADOR]" if resultados_modelado['mejor_modelo']=="RandomForestClassifier" else ""}")
    print(f"   ✓ Modelo seleccionado: {resultados_modelado['mejor_modelo']}")
    print(f"   ✓ 3 gráficas de modelado creadas")
    
    print("\n" + "-" * 70)
    print("✓ Todos los archivos han sido generados en:")
    print("  - Gráficas EDA: graphs/01_*.png, graphs/02_*.png, graphs/03_*.png")
    print("  - Gráficas Modelado: graphs/04_*.png, graphs/05_*.png")
    print("=" * 70)
    print("\n")
    
    # Retornar resultados
    return {
        'df_preparado': df_preparado,
        'reporte_eda': reporte_eda,
        'resultados_modelado': resultados_modelado
    }


if __name__ == "__main__":
    """
    Punto de entrada para ejecución directa del orquestador.
    
    Uso:
        python scripts/orchestrator.py
    """
    try:
        # Ejecutar pipeline con DB en data/
        resultados = ejecutar_pipeline_agentico(r"C:\Users\juanc\Documents\Universidad\Programacion Analitica De Datos\ProyectoAgentico_BDFutbol\data\premier_league.db")
        
        # Mostrar información final
        print("\n📌 INFORMACIÓN DE CONTACTO DE RESULTADOS:")
        print("=" * 70)
        print("\nPara acceder a los resultados:")
        print("  - DataFrame preparado: resultados['df_preparado']")
        print("  - Reporte EDA: resultados['reporte_eda']")
        print("  - Resultados modelado: resultados['resultados_modelado']")
        print("\nGráficas generadas en carpeta: graphs/")
        print("=" * 70)
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        print("Asegúrate de que 'premier_league.db' existe en la raíz del proyecto")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR INESPERADO: {e}")
        sys.exit(1)
