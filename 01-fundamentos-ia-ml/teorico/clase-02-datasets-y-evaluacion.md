# Clase 2 — Datasets, Features y Evaluación

## Objetivos
- Entender la estructura de un dataset para ML
- Conocer los conceptos de train/test split
- Identificar overfitting y underfitting

## Contenido

### 1. Anatomía de un dataset
- **Filas** = muestras/ejemplos
- **Columnas** = features (características)
- **Label/Target** = lo que queremos predecir
- Datos numéricos vs categóricos

### 2. Preparación de datos
- Datos faltantes (NaN): eliminar, imputar, ignorar
- Normalización y estandarización
- Encoding de variables categóricas (one-hot, label encoding)
- Feature engineering: crear nuevas features a partir de las existentes

### 3. Train / Test Split
- ¿Por qué separar datos? → Evaluar en datos que el modelo nunca vio
- Split típico: 80% train / 20% test
- Validation set y cross-validation
- **Data leakage:** el error más peligroso — cuando info del test se filtra al train

### 4. Overfitting vs Underfitting
- **Overfitting:** el modelo memoriza en vez de generalizar (muy bueno en train, malo en test)
- **Underfitting:** el modelo es demasiado simple (malo en ambos)
- Cómo detectarlo: comparar métricas train vs test
- Soluciones: más datos, regularización, modelos más/menos complejos

## Recursos
- [Kaggle Learn — Intro to ML](https://www.kaggle.com/learn/intro-to-machine-learning)
- [Scikit-learn: train_test_split docs](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)
