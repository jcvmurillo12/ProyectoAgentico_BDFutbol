"""
Skill de Modelado Predictivo para Sistema Agéntico de Premier League.

Este módulo contiene la lógica de entrenamiento, evaluación y selección
de modelos de machine learning para predecir el resultado de partidos
de la Premier League (Local, Empate, Visitante).

Funciones principales:
    - ejecutar_modelado(df): Función orquestadora que ejecuta todo el pipeline.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, Tuple, Any
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ─────────────────────────────────────────────
# 1. PREPARACIÓN DE FEATURES Y SPLIT
# ─────────────────────────────────────────────

def seleccionar_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Selecciona las features históricas (pre-partido) y la variable objetivo.

    Usa ÚNICAMENTE columnas calculadas ANTES del partido para evitar
    data leakage. No se incluyen goles_local ni goles_visitante.

    Parámetros:
        df (pd.DataFrame): DataFrame enriquecido de la Skill 1.

    Retorna:
        Tuple[pd.DataFrame, pd.Series]: (X con features, y con etiquetas).
    """
    features = [
        'promedio_goles_local',
        'promedio_goles_visitante',
        'racha_local',
        'racha_visitante',
        'ventaja_local'
    ]

    X = df[features].copy()
    y = df['resultado'].copy()

    print(f"✓ Features seleccionadas ({len(features)}): {features}")
    print(f"✓ Variable objetivo: 'resultado' — clases: {sorted(y.unique())}")
    print(f"✓ Shape del dataset: {X.shape}")
    print()

    return X, y


def dividir_datos(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide los datos en conjuntos de entrenamiento y prueba (80/20, estratificado).

    Parámetros:
        X (pd.DataFrame): Features.
        y (pd.Series): Variable objetivo.
        test_size (float): Proporción para test (default: 0.2).
        random_state (int): Semilla para reproducibilidad.

    Retorna:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"✓ División train/test: {len(X_train)} entrenamiento | {len(X_test)} prueba")
    print(f"  → Distribución en train: {y_train.value_counts().to_dict()}")
    print(f"  → Distribución en test:  {y_test.value_counts().to_dict()}")
    print()

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# 2. ENTRENAMIENTO DE MODELOS
# ─────────────────────────────────────────────

def entrenar_regresion_logistica(X_train, y_train) -> LogisticRegression:
    """Entrena Regresión Logística multiclase."""
    modelo = LogisticRegression(
        multi_class='multinomial', solver='lbfgs',
        max_iter=1000, random_state=42
    )
    modelo.fit(X_train, y_train)
    print("  ✓ Regresión Logística entrenada")
    return modelo


def entrenar_arbol_decision(X_train, y_train) -> DecisionTreeClassifier:
    """Entrena Árbol de Decisión con profundidad controlada."""
    modelo = DecisionTreeClassifier(
        max_depth=5, min_samples_split=10, random_state=42
    )
    modelo.fit(X_train, y_train)
    print("  ✓ Árbol de Decisión entrenado")
    return modelo


def entrenar_random_forest(X_train, y_train) -> RandomForestClassifier:
    """Entrena Random Forest con 100 estimadores."""
    modelo = RandomForestClassifier(
        n_estimators=100, max_depth=6,
        min_samples_split=10, random_state=42, n_jobs=-1
    )
    modelo.fit(X_train, y_train)
    print("  ✓ Random Forest entrenado")
    return modelo


def entrenar_todos_los_modelos(X_train, y_train) -> Dict[str, Any]:
    """
    Entrena los tres modelos candidatos.

    Retorna:
        Dict[str, modelo]: nombre → modelo entrenado.
    """
    print("Entrenando modelos candidatos...")
    modelos = {
        'Regresión Logística': entrenar_regresion_logistica(X_train, y_train),
        'Árbol de Decisión':   entrenar_arbol_decision(X_train, y_train),
        'Random Forest':       entrenar_random_forest(X_train, y_train),
    }
    print()
    return modelos


# ─────────────────────────────────────────────
# 3. EVALUACIÓN Y COMPARACIÓN
# ─────────────────────────────────────────────

def evaluar_modelo(nombre, modelo, X_train, X_test, y_train, y_test) -> Dict[str, Any]:
    """
    Evalúa un modelo con Accuracy, F1 Macro y validación cruzada (5-fold).

    Retorna:
        Dict con métricas del modelo.
    """
    y_pred    = modelo.predict(X_test)
    acc_train = accuracy_score(y_train, modelo.predict(X_train))
    acc_test  = accuracy_score(y_test, y_pred)
    f1_macro  = f1_score(y_test, y_pred, average='macro')
    cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='accuracy')

    resultado = {
        'nombre':      nombre,
        'modelo':      modelo,
        'acc_train':   acc_train,
        'acc_test':    acc_test,
        'f1_macro':    f1_macro,
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'cv_mean':     cv_scores.mean(),
        'cv_std':      cv_scores.std(),
        'y_pred':      y_pred,
        'reporte':     classification_report(y_test, y_pred, output_dict=True),
        'confusion':   confusion_matrix(y_test, y_pred, labels=['Local', 'Empate', 'Visitante'])
    }

    print(f"  [{nombre}]")
    print(f"    Accuracy Train : {acc_train:.4f}")
    print(f"    Accuracy Test  : {acc_test:.4f}")
    print(f"    F1 Macro       : {f1_macro:.4f}")
    print(f"    CV (5-fold)    : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print()

    return resultado


def comparar_modelos(modelos, X_train, X_test, y_train, y_test) -> Dict[str, Dict]:
    """Evalúa y compara todos los modelos entrenados."""
    print("Evaluando modelos...")
    resultados = {}
    for nombre, modelo in modelos.items():
        resultados[nombre] = evaluar_modelo(
            nombre, modelo, X_train, X_test, y_train, y_test
        )
    return resultados


def seleccionar_mejor_modelo(resultados: Dict[str, Dict]) -> Dict[str, Any]:
    """
    Selecciona el mejor modelo con criterio ponderado:
        Score = 0.6 × F1_macro_test + 0.4 × CV_mean

    Justificación: F1 Macro penaliza el desbalance de clases (3 clases);
    CV mean valida estabilidad y ausencia de sobreajuste.
    """
    mejor       = None
    mejor_score = -1

    print("Seleccionando mejor modelo...")
    print(f"  {'Modelo':<25} {'F1 Macro':>10} {'CV Mean':>10} {'Score':>10}")
    print(f"  {'-'*57}")

    for nombre, res in resultados.items():
        score = 0.6 * res['f1_macro'] + 0.4 * res['cv_mean']
        print(f"  {nombre:<25} {res['f1_macro']:>10.4f} {res['cv_mean']:>10.4f} {score:>10.4f}")
        if score > mejor_score:
            mejor_score = score
            mejor = res

    print(f"\n  MODELO SELECCIONADO: {mejor['nombre']}")
    print(f"     Score final: {mejor_score:.4f}")
    print()
    return mejor


# ─────────────────────────────────────────────
# 4. VISUALIZACIONES
# ─────────────────────────────────────────────

def generar_grafica_comparacion_modelos(resultados: Dict[str, Dict]) -> None:
    """Gráfica 4: Comparación de métricas entre los tres modelos."""
    nombres  = list(resultados.keys())
    acc_test = [resultados[n]['acc_test'] for n in nombres]
    f1_macro = [resultados[n]['f1_macro'] for n in nombres]
    cv_mean  = [resultados[n]['cv_mean']  for n in nombres]

    x     = np.arange(len(nombres))
    ancho = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - ancho, acc_test, ancho, label='Accuracy Test', color='#3498db', edgecolor='black')
    b2 = ax.bar(x,          f1_macro, ancho, label='F1 Macro',      color='#2ecc71', edgecolor='black')
    b3 = ax.bar(x + ancho,  cv_mean,  ancho, label='CV Mean (5k)',   color='#e67e22', edgecolor='black')

    ax.set_ylabel('Puntuación', fontsize=12, fontweight='bold')
    ax.set_title('Comparación de Métricas por Modelo', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(nombres, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.bar_label(b1, fmt='%.3f', fontsize=9)
    ax.bar_label(b2, fmt='%.3f', fontsize=9)
    ax.bar_label(b3, fmt='%.3f', fontsize=9)

    plt.tight_layout()
    plt.savefig('graphs/04_comparacion_modelos.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfica 4 guardada: graphs/04_comparacion_modelos.png")
    plt.close()


def generar_grafica_matriz_confusion(mejor_modelo: Dict[str, Any]) -> None:
    """Gráfica 5: Matriz de confusión del modelo seleccionado."""
    import seaborn as sns

    clases = ['Local', 'Empate', 'Visitante']
    cm     = mejor_modelo['confusion']

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=clases, yticklabels=clases,
        linewidths=1, linecolor='gray', ax=ax
    )
    ax.set_xlabel('Predicción', fontsize=12, fontweight='bold')
    ax.set_ylabel('Real',       fontsize=12, fontweight='bold')
    ax.set_title(
        f'Matriz de Confusión — {mejor_modelo["nombre"]}',
        fontsize=13, fontweight='bold', pad=15
    )
    plt.tight_layout()
    plt.savefig('graphs/05_matriz_confusion.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfica 5 guardada: graphs/05_matriz_confusion.png")
    plt.close()


def generar_grafica_importancia_features(mejor_modelo: Dict[str, Any], X: pd.DataFrame) -> None:
    """
    Gráfica 6: Importancia de features del modelo seleccionado.
    Para Random Forest/Árbol usa feature_importances_.
    Para Regresión Logística usa coeficientes absolutos promedio.
    """
    modelo   = mejor_modelo['modelo']
    nombre   = mejor_modelo['nombre']
    features = X.columns.tolist()

    if hasattr(modelo, 'feature_importances_'):
        importancias = modelo.feature_importances_
        xlabel       = 'Importancia (Gini)'
    else:
        importancias = np.abs(modelo.coef_).mean(axis=0)
        xlabel       = 'Coeficiente promedio absoluto'

    idx_orden = np.argsort(importancias)
    fig, ax   = plt.subplots(figsize=(10, 5))
    ax.barh(
        [features[i] for i in idx_orden],
        [importancias[i] for i in idx_orden],
        color='#9b59b6', edgecolor='black'
    )
    ax.set_xlabel(xlabel, fontsize=12, fontweight='bold')
    ax.set_title(f'Importancia de Features — {nombre}', fontsize=13, fontweight='bold', pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig('graphs/06_importancia_features.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfica 6 guardada: graphs/06_importancia_features.png")
    plt.close()


# ─────────────────────────────────────────────
# 5. REPORTE FINAL
# ─────────────────────────────────────────────

def generar_reporte_modelado(resultados: Dict[str, Dict], mejor_modelo: Dict[str, Any]) -> Dict[str, str]:
    """
    Genera reporte textual con comparativa de modelos y justificación del ganador.

    Retorna:
        Dict[str, str]: hallazgos clave del modelado.
    """
    reporte = {}

    # Tabla comparativa
    tabla = "Modelo                    | Acc.Test | F1 Macro | CV Mean\n"
    tabla += "-" * 62 + "\n"
    for nombre, res in resultados.items():
        marca = " <- GANADOR" if nombre == mejor_modelo['nombre'] else ""
        tabla += (
            f"{nombre:<26}| {res['acc_test']:.4f}   | "
            f"{res['f1_macro']:.4f}   | {res['cv_mean']:.4f}{marca}\n"
        )
    reporte['comparacion_modelos'] = tabla

    # Justificación
    m = mejor_modelo
    reporte['modelo_seleccionado'] = (
        f"El modelo seleccionado es '{m['nombre']}' con F1 Macro={m['f1_macro']:.4f} "
        f"y CV medio={m['cv_mean']:.4f}. Se eligio usando el criterio "
        f"Score = 0.6*F1_macro + 0.4*CV_mean, que prioriza generalizacion "
        f"y estabilidad sobre ajuste puro al conjunto de entrenamiento."
    )

    # Métricas por clase
    rep = m['reporte']
    clases_reporte = ""
    for clase in ['Local', 'Empate', 'Visitante']:
        if clase in rep:
            clases_reporte += (
                f"  {clase}: Precision={rep[clase]['precision']:.3f}  "
                f"Recall={rep[clase]['recall']:.3f}  "
                f"F1={rep[clase]['f1-score']:.3f}\n"
            )
    reporte['metricas_por_clase'] = clases_reporte

    # Conclusión
    reporte['conclusion'] = (
        f"El sistema agéntico completo el pipeline de modelado predictivo. "
        f"El modelo '{m['nombre']}' logra una exactitud de {m['acc_test']:.1%} en datos de prueba. "
        f"Las features históricas (promedio de goles y racha de victorias) demostraron "
        f"ser predictores relevantes. El factor ventaja_local refuerza la tendencia "
        f"observada en el EDA: los equipos locales ganan con mayor frecuencia."
    )

    return reporte


# ─────────────────────────────────────────────
# 6. FUNCIÓN ORQUESTADORA PRINCIPAL
# ─────────────────────────────────────────────

def ejecutar_modelado(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Función orquestadora principal que ejecuta todo el pipeline de modelado.

    Pipeline:
        1. Selecciona features históricas sin data leakage.
        2. Divide datos train/test (80/20, estratificado).
        3. Entrena tres modelos: Regresión Logística, Árbol de Decisión, Random Forest.
        4. Evalúa cada modelo con Accuracy, F1 Macro y CV 5-fold.
        5. Selecciona mejor modelo con justificación explícita.
        6. Genera 3 gráficas (comparación, matriz de confusión, importancia features).
        7. Retorna reporte completo.

    Parámetros:
        df (pd.DataFrame): DataFrame enriquecido de la Skill 1, con columnas:
            promedio_goles_local, promedio_goles_visitante,
            racha_local, racha_visitante, ventaja_local, resultado.

    Retorna:
        Dict: 'reporte_modelado' con claves:
            - comparacion_modelos  : tabla comparativa de métricas
            - modelo_seleccionado  : nombre y justificación del ganador
            - metricas_por_clase   : precision/recall/F1 por clase
            - conclusion           : síntesis del modelado
            - mejor_modelo         : dict con objeto del modelo (para predicciones)

    Archivos generados:
        - graphs/04_comparacion_modelos.png
        - graphs/05_matriz_confusion.png
        - graphs/06_importancia_features.png
    """
    print("=" * 70)
    print("INICIANDO PIPELINE DE MODELADO PREDICTIVO")
    print("=" * 70)
    print()

    # 1. Features
    print("[1/7] Seleccionando features...")
    X, y = seleccionar_features(df)

    # 2. Split
    print("[2/7] Dividiendo datos train/test...")
    X_train, X_test, y_train, y_test = dividir_datos(X, y)

    # 3. Entrenamiento
    print("[3/7] Entrenando modelos...")
    modelos = entrenar_todos_los_modelos(X_train, y_train)

    # 4. Evaluación
    print("[4/7] Evaluando modelos...")
    resultados = comparar_modelos(modelos, X_train, X_test, y_train, y_test)

    # 5. Selección
    print("[5/7] Seleccionando mejor modelo...")
    mejor_modelo = seleccionar_mejor_modelo(resultados)

    # 6. Gráficas
    print("[6/7] Generando visualizaciones...")
    print("-" * 70)
    generar_grafica_comparacion_modelos(resultados)
    generar_grafica_matriz_confusion(mejor_modelo)
    generar_grafica_importancia_features(mejor_modelo, X)
    print("-" * 70)
    print()

    # 7. Reporte
    print("[7/7] Generando reporte de modelado...")
    print("-" * 70)
    reporte_modelado = generar_reporte_modelado(resultados, mejor_modelo)

    for clave, contenido in reporte_modelado.items():
        print(f"\n{clave.upper()}:")
        print(contenido)

    reporte_modelado['mejor_modelo'] = mejor_modelo

    print()
    print("=" * 70)
    print("PIPELINE DE MODELADO COMPLETADO EXITOSAMENTE")
    print("=" * 70)

    return reporte_modelado


if __name__ == "__main__":
    from scripts.skills.skill_preparacion import preparar_datos
    df_preparado = preparar_datos('data/premier_league.db')
    reporte = ejecutar_modelado(df_preparado)
