"""
Agente Orquestador Principal — Sistema Agéntico de Análisis Predictivo
Premier League de Fútbol Inglés.

Este módulo implementa el único agente orquestador del sistema.
Su responsabilidad es coordinar la ejecución de las tres skills
en el orden correcto, pasar los datos entre ellas y consolidar
los resultados finales.

El orquestador NO contiene lógica analítica directamente.
Toda la lógica vive en las skills.

Función principal:
    - ejecutar_pipeline(db_path): Ejecuta el pipeline completo.
"""

import os
import sys
import time
from datetime import datetime
from typing import Dict, Any

# Importar las tres skills
from scripts.skills.skill_preparacion import preparar_datos
from scripts.skills.skill_eda import ejecutar_eda
from scripts.skills.skill_modelado import ejecutar_modelado


# ─────────────────────────────────────────────
# UTILIDADES DEL ORQUESTADOR
# ─────────────────────────────────────────────

def _log(mensaje: str, nivel: str = "INFO") -> None:
    """Registra un mensaje con timestamp."""
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] [{nivel}] {mensaje}")


def _separador(titulo: str = "", ancho: int = 70) -> None:
    """Imprime un separador visual."""
    if titulo:
        espacio = ancho - len(titulo) - 4
        izq = espacio // 2
        der = espacio - izq
        print("\n" + "=" * izq + f"  {titulo}  " + "=" * der)
    else:
        print("=" * ancho)


def _verificar_prerequisitos(db_path: str) -> None:
    """
    Verifica que existan los archivos y carpetas necesarios
    antes de iniciar el pipeline.

    Lanza FileNotFoundError si la base de datos no existe.
    Crea la carpeta 'graphs/' si no existe.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Base de datos no encontrada: '{db_path}'\n"
            "Asegúrate de que el archivo SQLite existe en la ruta indicada."
        )

    os.makedirs('graphs', exist_ok=True)
    _log(f"Base de datos encontrada: {db_path}")
    _log("Carpeta 'graphs/' lista para gráficas.")


# ─────────────────────────────────────────────
# PASOS DEL PIPELINE
# ─────────────────────────────────────────────

def _paso_preparacion(db_path: str) -> Any:
    """
    Paso 1: Invoca la Skill 1 (preparación de datos).

    Parámetros:
        db_path (str): Ruta a la base de datos SQLite.

    Retorna:
        pd.DataFrame: Dataset limpio y enriquecido con features históricas.
    """
    _separador("PASO 1 — PREPARACIÓN DE DATOS")
    _log("Invocando Skill 1: skill_preparacion.preparar_datos()")

    t0 = time.time()
    df = preparar_datos(db_path)
    elapsed = time.time() - t0

    _log(f"Skill 1 completada en {elapsed:.1f}s — {len(df)} partidos, {len(df.columns)} columnas.")
    return df


def _paso_eda(df: Any) -> Dict[str, str]:
    """
    Paso 2: Invoca la Skill 2 (análisis exploratorio de datos).

    Parámetros:
        df (pd.DataFrame): Dataset preparado por la Skill 1.

    Retorna:
        Dict[str, str]: Reporte EDA con hallazgos principales.
    """
    _separador("PASO 2 — ANÁLISIS EXPLORATORIO (EDA)")
    _log("Invocando Skill 2: skill_eda.ejecutar_eda()")

    t0 = time.time()
    reporte_eda = ejecutar_eda(df)
    elapsed = time.time() - t0

    _log(f"Skill 2 completada en {elapsed:.1f}s — {len(reporte_eda)} hallazgos generados.")
    return reporte_eda


def _paso_modelado(df: Any) -> Dict[str, Any]:
    """
    Paso 3: Invoca la Skill 3 (modelado predictivo).

    Parámetros:
        df (pd.DataFrame): Dataset preparado por la Skill 1.

    Retorna:
        Dict[str, Any]: Reporte de modelado con métricas y modelo seleccionado.
    """
    _separador("PASO 3 — MODELADO PREDICTIVO")
    _log("Invocando Skill 3: skill_modelado.ejecutar_modelado()")

    t0 = time.time()
    reporte_modelado = ejecutar_modelado(df)
    elapsed = time.time() - t0

    mejor = reporte_modelado.get('mejor_modelo', {})
    _log(
        f"Skill 3 completada en {elapsed:.1f}s — "
        f"Modelo seleccionado: {mejor.get('nombre', 'N/A')} "
        f"(Accuracy={mejor.get('acc_test', 0):.4f})"
    )
    return reporte_modelado


def _consolidar_resultados(
    reporte_eda: Dict[str, str],
    reporte_modelado: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Consolida los reportes de las tres skills en un resumen ejecutivo final.

    Parámetros:
        reporte_eda (Dict): Reporte de la Skill 2.
        reporte_modelado (Dict): Reporte de la Skill 3.

    Retorna:
        Dict: Resumen ejecutivo del pipeline completo.
    """
    mejor = reporte_modelado.get('mejor_modelo', {})

    resumen = {
        'timestamp':          datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'eda_conclusion':     reporte_eda.get('conclusion', ''),
        'eda_distribucion':   reporte_eda.get('distribucion_resultados', ''),
        'modelo_ganador':     mejor.get('nombre', 'N/A'),
        'accuracy_test':      mejor.get('acc_test', 0),
        'f1_macro':           mejor.get('f1_macro', 0),
        'cv_mean':            mejor.get('cv_mean', 0),
        'modelo_conclusion':  reporte_modelado.get('conclusion', ''),
        'graficas_generadas': [
            'graphs/01_distribucion_resultados.png',
            'graphs/02_promedio_goles_equipos.png',
            'graphs/03_heatmap_correlacion.png',
            'graphs/04_comparacion_modelos.png',
            'graphs/05_matriz_confusion.png',
            'graphs/06_importancia_features.png',
        ]
    }

    return resumen


def _imprimir_resumen_final(resumen: Dict[str, Any]) -> None:
    """Imprime el resumen ejecutivo del pipeline en consola."""
    _separador("RESUMEN EJECUTIVO FINAL")

    print(f"\n  Timestamp      : {resumen['timestamp']}")
    print(f"\n  HALLAZGO EDA")
    print(f"  {resumen['eda_distribucion']}")
    print(f"\n  MODELO SELECCIONADO")
    print(f"  Nombre         : {resumen['modelo_ganador']}")
    print(f"  Accuracy Test  : {resumen['accuracy_test']:.4f}")
    print(f"  F1 Macro       : {resumen['f1_macro']:.4f}")
    print(f"  CV Mean (5-fold): {resumen['cv_mean']:.4f}")
    print(f"\n  CONCLUSION")
    print(f"  {resumen['modelo_conclusion']}")
    print(f"\n  GRAFICAS GENERADAS ({len(resumen['graficas_generadas'])})")
    for g in resumen['graficas_generadas']:
        print(f"    - {g}")

    _separador()


# ─────────────────────────────────────────────
# FUNCIÓN PRINCIPAL DEL ORQUESTADOR
# ─────────────────────────────────────────────

def ejecutar_pipeline(db_path: str = 'data/premier_league.db') -> Dict[str, Any]:
    """
    Ejecuta el pipeline agéntico completo coordinando las tres skills.

    Flujo de ejecución:
        1. Verificar prerequisitos (BD, carpetas).
        2. Skill 1 — Preparación: carga, limpieza y feature engineering.
        3. Skill 2 — EDA: estadísticas, análisis y 3 gráficas.
        4. Skill 3 — Modelado: entrenamiento, evaluación y 3 gráficas.
        5. Consolidar y mostrar resumen ejecutivo final.

    Principio de diseño:
        El orquestador SOLO coordina. No contiene lógica analítica.
        Cada skill es autónoma y recibe/retorna datos bien definidos.

    Parámetros:
        db_path (str): Ruta al archivo SQLite con los datos de partidos.
                       Default: 'data/premier_league.db'

    Retorna:
        Dict: Resumen ejecutivo con métricas finales y rutas de gráficas.

    Ejemplo:
        >>> from scripts.orchestrator import ejecutar_pipeline
        >>> resumen = ejecutar_pipeline('data/premier_league.db')
        >>> print(resumen['modelo_ganador'])
    """
    _separador("SISTEMA AGÉNTICO — PREMIER LEAGUE PREDICTIVE ANALYTICS")
    _log("Agente orquestador iniciado.")
    _log(f"Base de datos objetivo: {db_path}")
    t_inicio = time.time()

    # Verificar prerequisitos
    _verificar_prerequisitos(db_path)

    # Paso 1 — Skill 1: Preparación
    df = _paso_preparacion(db_path)

    # Paso 2 — Skill 2: EDA
    reporte_eda = _paso_eda(df)

    # Paso 3 — Skill 3: Modelado
    reporte_modelado = _paso_modelado(df)

    # Consolidar resultados
    resumen = _consolidar_resultados(reporte_eda, reporte_modelado)

    # Mostrar resumen final
    _imprimir_resumen_final(resumen)

    t_total = time.time() - t_inicio
    _log(f"Pipeline completo ejecutado en {t_total:.1f}s.")
    _log("Agente orquestador finalizado correctamente.")

    return resumen


# ─────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else 'data/premier_league.db'
    resumen = ejecutar_pipeline(db_path)
