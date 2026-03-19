# Clase 3 — Regresión y Métricas de Evaluación

## Objetivos
- Entender regresión lineal y logística
- Conocer las métricas principales para evaluar modelos

## Contenido

### 1. Regresión Lineal
- Encontrar la "mejor línea" que ajuste los datos
- Función de costo: MSE (Mean Squared Error)
- Gradient descent para minimizar el error
- Regresión múltiple: más de una feature

### 2. Regresión Logística (Clasificación)
- A pesar del nombre, es para **clasificación** (sí/no, spam/no-spam)
- Función sigmoide: convierte cualquier número en probabilidad (0 a 1)
- Umbral de decisión: generalmente 0.5

### 3. Métricas de Clasificación
- **Accuracy:** % de predicciones correctas (engañosa con clases desbalanceadas)
- **Precision:** de los que predije positivos, ¿cuántos realmente lo son?
- **Recall:** de los que son positivos, ¿cuántos capturé?
- **F1-Score:** media armónica de precision y recall
- **Matriz de confusión:** verdaderos/falsos positivos/negativos

### 4. ¿Cuándo importa más precision vs recall?
- Detector de spam: más precision (no quiero perder emails legítimos)
- Diagnóstico de cáncer: más recall (no quiero dejar pasar un caso real)
- Depende del **costo del error** en cada contexto

## Recursos
- [StatQuest: Linear Regression (YouTube)](https://www.youtube.com/watch?v=nk2CQITm_eo)
- [Scikit-learn: Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
