"""
Skill de Modelado Predictivo para Sistema Agéntico de Premier League.

Este módulo contiene la lógica de entrenamiento, evaluación y selección
de modelos de machine learning para predecir el resultado de partidos.

Funciones principales:
    - ejecutar_modelado(df): Función orquestadora que entrena y evalúa modelos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from typing import Dict, Tuple, Any


def preparar_datos_modelado(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepara los datos para el modelado: selecciona features y target.
    
    Features utilizadas:
        - promedio_goles_local
        - promedio_goles_visitante
        - racha_local
        - racha_visitante
        - ventaja_local
    
    Target:
        - resultado (Local, Empate, Visitante)
    
    Parámetros:
        df (pd.DataFrame): DataFrame completo con todas las características.
    
    Retorna:
        Tuple[np.ndarray, np.ndarray]: (X, y) donde:
            - X: Features array [n_samples, n_features=5]
            - y: Target array [n_samples]
    """
    # Seleccionar features
    features = [
        'promedio_goles_local',
        'promedio_goles_visitante',
        'racha_local',
        'racha_visitante',
        'ventaja_local'
    ]
    
    X = df[features].values
    y = df['resultado'].values
    
    print(f"✓ Features seleccionados: {features}")
    print(f"✓ Dimensiones: X={X.shape}, y={y.shape}")
    print(f"✓ Clases: {np.unique(y)}")
    
    return X, y


def dividir_datos(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Divide los datos en conjunto de entrenamiento y prueba.
    
    Parámetros:
        X (np.ndarray): Features array.
        y (np.ndarray): Target array.
        test_size (float): Proporción de datos para test (default: 0.2).
        random_state (int): Seed para reproducibilidad.
    
    Retorna:
        Tuple: (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y
    )
    
    print(f"✓ Datos divididos:")
    print(f"  - Entrenamiento: {X_train.shape[0]} samples ({100*(1-test_size):.0f}%)")
    print(f"  - Prueba: {X_test.shape[0]} samples ({100*test_size:.0f}%)")
    
    return X_train, X_test, y_train, y_test


def escalar_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Escala los features usando StandardScaler.
    
    Parámetros:
        X_train (np.ndarray): Features de entrenamiento.
        X_test (np.ndarray): Features de prueba.
    
    Retorna:
        Tuple: (X_train_scaled, X_test_scaled, scaler)
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✓ Features escalados usando StandardScaler")
    
    return X_train_scaled, X_test_scaled, scaler


def entrenar_logistic_regression(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    """
    Entrena un modelo de Regresión Logística.
    
    Parámetros:
        X_train (np.ndarray): Features de entrenamiento (escalados).
        y_train (np.ndarray): Target de entrenamiento.
    
    Retorna:
        LogisticRegression: Modelo entrenado.
    """
    modelo = LogisticRegression(
        max_iter=1000,
        multi_class='multinomial',
        solver='lbfgs',
        random_state=42
    )
    modelo.fit(X_train, y_train)
    
    print("✓ Modelo LogisticRegression entrenado")
    
    return modelo


def entrenar_random_forest(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    """
    Entrena un modelo de Random Forest.
    
    Parámetros:
        X_train (np.ndarray): Features de entrenamiento.
        y_train (np.ndarray): Target de entrenamiento.
    
    Retorna:
        RandomForestClassifier: Modelo entrenado.
    """
    modelo = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    modelo.fit(X_train, y_train)
    
    print("✓ Modelo RandomForestClassifier entrenado")
    
    return modelo


def evaluar_modelo(modelo: Any, X_test: np.ndarray, y_test: np.ndarray, nombre_modelo: str) -> Dict[str, Any]:
    """
    Evalúa un modelo y retorna métricas de desempeño.
    
    Métricas calculadas:
        - Accuracy
        - Precision (macro)
        - Recall (macro)
        - Matriz de confusión
        - Reporte de clasificación
    
    Parámetros:
        modelo (Any): Modelo entrenado (sklearn).
        X_test (np.ndarray): Features de prueba.
        y_test (np.ndarray): Target de prueba.
        nombre_modelo (str): Nombre del modelo para reporte.
    
    Retorna:
        Dict: Diccionario con métricas:
            - nombre: nombre del modelo
            - accuracy: precisión general
            - precision: precisión promedio (macro)
            - recall: recall promedio (macro)
            - y_pred: predicciones
            - confusion_matrix: matriz de confusión
            - report: reporte de clasificación
    """
    y_pred = modelo.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=0)
    cm = confusion_matrix(y_test, y_pred, labels=['Local', 'Empate', 'Visitante'])
    report = classification_report(y_test, y_pred, output_dict=False)
    
    metricas = {
        'nombre': nombre_modelo,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'y_pred': y_pred,
        'confusion_matrix': cm,
        'report': report,
        'y_test': y_test
    }
    
    print(f"\n📊 MÉTRICAS - {nombre_modelo}")
    print("-" * 70)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print()
    
    return metricas


def generar_matriz_confusion(metricas_modelo: Dict[str, Any], figsize: Tuple[int, int] = (8, 6)) -> None:
    """
    Genera y guarda visualización de matriz de confusión.
    
    Parámetros:
        metricas_modelo (Dict): Diccionario con métricas del modelo.
        figsize (Tuple): Tamaño de la figura.
    """
    # Crear carpeta si no existe
    os.makedirs('graphs', exist_ok=True)
    
    cm = metricas_modelo['confusion_matrix']
    nombre = metricas_modelo['nombre'].replace(" ", "_").lower()
    
    plt.figure(figsize=figsize)
    
    labels = ['Local', 'Empate', 'Visitante']
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Cantidad'},
        linewidths=1,
        linecolor='black'
    )
    
    plt.title(f'Matriz de Confusión - {metricas_modelo["nombre"]}', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicción', fontsize=12, fontweight='bold')
    plt.ylabel('Valor Real', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    archivo = f'graphs/04_matriz_confusion_{nombre}.png'
    plt.savefig(archivo, dpi=300, bbox_inches='tight')
    print(f"✓ Matriz de confusión guardada: {archivo}")
    plt.close()


def generar_comparacion_modelos(metricas_lr: Dict[str, Any], metricas_rf: Dict[str, Any], figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Genera gráfica comparativa de métricas entre modelos.
    
    Parámetros:
        metricas_lr (Dict): Métricas del modelo LogisticRegression.
        metricas_rf (Dict): Métricas del modelo RandomForest.
        figsize (Tuple): Tamaño de la figura.
    """
    # Crear carpeta si no existe
    os.makedirs('graphs', exist_ok=True)
    
    modelos = [metricas_lr['nombre'], metricas_rf['nombre']]
    accuracy_vals = [metricas_lr['accuracy'], metricas_rf['accuracy']]
    precision_vals = [metricas_lr['precision'], metricas_rf['precision']]
    recall_vals = [metricas_lr['recall'], metricas_rf['recall']]
    
    x = np.arange(len(modelos))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=figsize)
    
    bars1 = ax.bar(x - width, accuracy_vals, width, label='Accuracy', color='#3498db', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x, precision_vals, width, label='Precision', color='#2ecc71', edgecolor='black', linewidth=1.5)
    bars3 = ax.bar(x + width, recall_vals, width, label='Recall', color='#e74c3c', edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Comparación de Métricas entre Modelos', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(modelos, fontsize=11)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Añadir valores en las barras
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('graphs/05_comparacion_modelos.png', dpi=300, bbox_inches='tight')
    print("✓ Gráfica de comparación guardada: graphs/05_comparacion_modelos.png")
    plt.close()


def seleccionar_mejor_modelo(metricas_lr: Dict[str, Any], metricas_rf: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Selecciona el mejor modelo basado en accuracy.
    
    Criterio de selección: Mayor accuracy
    
    Parámetros:
        metricas_lr (Dict): Métricas del modelo LogisticRegression.
        metricas_rf (Dict): Métricas del modelo RandomForest.
    
    Retorna:
        Tuple[str, Dict]: (nombre_mejor_modelo, metricas_mejor_modelo)
    """
    if metricas_rf['accuracy'] > metricas_lr['accuracy']:
        mejor_modelo = metricas_rf['nombre']
        mejores_metricas = metricas_rf
    else:
        mejor_modelo = metricas_lr['nombre']
        mejores_metricas = metricas_lr
    
    print()
    print("=" * 70)
    print(f"🏆 MODELO GANADOR: {mejor_modelo}")
    print("=" * 70)
    print(f"Accuracy:  {mejores_metricas['accuracy']:.4f}")
    print(f"Precision: {mejores_metricas['precision']:.4f}")
    print(f"Recall:    {mejores_metricas['recall']:.4f}")
    print("=" * 70)
    
    return mejor_modelo, mejores_metricas


def generar_reporte_modelado(mejor_modelo: str, metricas_lr: Dict[str, Any], metricas_rf: Dict[str, Any]) -> Dict[str, str]:
    """
    Genera un reporte textual con los hallazgos principales del modelado.
    
    Parámetros:
        mejor_modelo (str): Nombre del modelo ganador.
        metricas_lr (Dict): Métricas del modelo LogisticRegression.
        metricas_rf (Dict): Métricas del modelo RandomForest.
    
    Retorna:
        Dict: Reporte con hallazgos principales.
    """
    reporte = {}
    
    # Hallazgo 1: Rendimiento general
    reporte['rendimiento_general'] = (
        f"Se entrenaron dos modelos: LogisticRegression (Accuracy: {metricas_lr['accuracy']:.4f}) "
        f"y RandomForestClassifier (Accuracy: {metricas_rf['accuracy']:.4f}). "
        f"El modelo ganador es {mejor_modelo} con una precisión del {metricas_lr['accuracy']*100 if mejor_modelo=='LogisticRegression' else metricas_rf['accuracy']*100:.2f}%."
    )
    
    # Hallazgo 2: Análisis de precisión y recall
    if mejor_modelo == 'LogisticRegression':
        metricas = metricas_lr
    else:
        metricas = metricas_rf
    
    reporte['metricas_detalladas'] = (
        f"El modelo ganador ({mejor_modelo}) presenta:\n"
        f"- Accuracy (precisión general): {metricas['accuracy']:.4f}\n"
        f"- Precision (macro): {metricas['precision']:.4f}\n"
        f"- Recall (macro): {metricas['recall']:.4f}\n"
        f"Estas métricas indican un rendimiento equilibrado en la predicción de los tres resultados."
    )
    
    # Hallazgo 3: Justificación del modelo ganador
    if mejor_modelo == 'RandomForestClassifier':
        reporte['justificacion'] = (
            f"RandomForest superó a LogisticRegression porque es capaz de capturar "
            f"relaciones no-lineales entre las features históricas de los equipos y el resultado del partido. "
            f"Su capacidad de ensemble mejora la generalización y robustez de las predicciones."
        )
    else:
        reporte['justificacion'] = (
            f"LogisticRegression demostró un rendimiento superior, sugiriendo que las relaciones "
            f"entre las features y el resultado son predominantemente lineales. Su simplicidad también "
            f"facilita la interpretabilidad del modelo."
        )
    
    # Hallazgo 4: Recomendaciones
    reporte['recomendaciones'] = (
        f"Se recomienda utilizar {mejor_modelo} para predicciones de resultados en futuros partidos. "
        f"Se pueden considerar mejoras futuras como: feature engineering adicional, "
        f"ajuste de hiperparámetros mediante GridSearch, y evaluación en datos de temporadas futuras."
    )
    
    return reporte


def ejecutar_modelado(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Función orquestadora principal que ejecuta todo el pipeline de modelado.
    
    Pipeline:
        1. Preparación de datos (selección de features y target).
        2. División en conjunto de entrenamiento y prueba.
        3. Escalado de features.
        4. Entrenamiento de LogisticRegression.
        5. Entrenamiento de RandomForest.
        6. Evaluación de ambos modelos.
        7. Selección del mejor modelo.
        8. Generación de visualizaciones.
        9. Compilación de reporte.
    
    Parámetros:
        df (pd.DataFrame): DataFrame completo con features y target, proveniente de skill_eda.
    
    Retorna:
        Dict: Diccionario con resultados del modelado:
            - mejor_modelo: nombre del modelo ganador
            - metricas: diccionario con métricas del modelo ganador
            - reporte_modelado: diccionario con hallazgos principales
    
    Archivos generados:
        - graphs/04_matriz_confusion_logisticregression.png
        - graphs/04_matriz_confusion_randomforestclassifier.png
        - graphs/05_comparacion_modelos.png
    
    Ejemplo:
        >>> df_eda = ejecutar_eda(df_preparado)  # (aunque retorna reporte_eda)
        >>> resultados = ejecutar_modelado(df_preparado)  # Recibe df completo
        >>> print(resultados['mejor_modelo'])
    """
    print("=" * 70)
    print("INICIANDO PIPELINE DE MODELADO PREDICTIVO")
    print("=" * 70)
    print()
    
    # 1. Preparar datos
    print("[1/9] Preparando datos...")
    X, y = preparar_datos_modelado(df)
    print()
    
    # 2. Dividir datos
    print("[2/9] Dividiendo datos en entrenamiento/prueba...")
    X_train, X_test, y_train, y_test = dividir_datos(X, y)
    print()
    
    # 3. Escalar features
    print("[3/9] Escalando features...")
    X_train_scaled, X_test_scaled, scaler = escalar_features(X_train, X_test)
    print()
    
    # 4. Entrenar LogisticRegression
    print("[4/9] Entrenando LogisticRegression...")
    modelo_lr = entrenar_logistic_regression(X_train_scaled, y_train)
    print()
    
    # 5. Entrenar RandomForest
    print("[5/9] Entrenando RandomForestClassifier...")
    modelo_rf = entrenar_random_forest(X_train, y_train)  # RandomForest no requiere escalado
    print()
    
    # 6. Evaluar LogisticRegression
    print("[6/9] Evaluando LogisticRegression...")
    metricas_lr = evaluar_modelo(modelo_lr, X_test_scaled, y_test, "LogisticRegression")
    
    # 7. Evaluar RandomForest
    print("[7/9] Evaluando RandomForestClassifier...")
    metricas_rf = evaluar_modelo(modelo_rf, X_test, y_test, "RandomForestClassifier")
    
    # 8. Seleccionar mejor modelo
    print("[8/9] Seleccionando mejor modelo...")
    mejor_modelo, mejores_metricas = seleccionar_mejor_modelo(metricas_lr, metricas_rf)
    print()
    
    # 9. Generar visualizaciones y reporte
    print("[9/9] Generando visualizaciones y reporte...")
    print("-" * 70)
    generar_matriz_confusion(metricas_lr)
    generar_matriz_confusion(metricas_rf)
    generar_comparacion_modelos(metricas_lr, metricas_rf)
    print("-" * 70)
    print()
    
    # Generar reporte
    print("📋 GENERANDO REPORTE DE MODELADO...")
    print("-" * 70)
    reporte_modelado = generar_reporte_modelado(mejor_modelo, metricas_lr, metricas_rf)
    
    for clave, contenido in reporte_modelado.items():
        print(f"\n{clave.upper()}:")
        print(contenido)
    
    print()
    print("=" * 70)
    print("✓ PIPELINE DE MODELADO COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    
    return {
        'mejor_modelo': mejor_modelo,
        'metricas': mejores_metricas,
        'reporte_modelado': reporte_modelado
    }


if __name__ == "__main__":
    # Leer dataset preparado
    df_preparado = pd.read_csv('data/dataset_preparado.csv')
    
    # Ejecutar modelado
    resultados = ejecutar_modelado(df_preparado)
    
    # Preparar reporte para guardar en JSON (convertir valores numpy)
    reporte_para_json = {
        'mejor_modelo': resultados['mejor_modelo'],
        'metricas': {
            'accuracy': float(resultados['metricas']['accuracy']),
            'precision': float(resultados['metricas']['precision']),
            'recall': float(resultados['metricas']['recall'])
        },
        'reporte_modelado': resultados['reporte_modelado']
    }
    
    # Guardar reporte en JSON
    output_path = "data/reporte_modelado.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(reporte_para_json, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Reporte guardado en: {output_path}")
