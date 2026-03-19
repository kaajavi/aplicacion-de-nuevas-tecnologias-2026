# Clase 4 — Árboles de Decisión y Ensembles

## Objetivos
- Entender árboles de decisión y cómo eligen los splits
- Conocer Random Forests y el concepto de ensemble

## Contenido

### 1. Árboles de Decisión
- Estructura: nodos de decisión → ramas → hojas (predicción)
- ¿Cómo elige los splits? Gini impurity, information gain
- Ventajas: interpretables, no requieren normalización
- Desventaja principal: tienden a overfittear

### 2. Ensembles: la sabiduría de las multitudes
- Idea: combinar muchos modelos débiles = un modelo fuerte
- **Bagging:** entrenar modelos en subsets aleatorios y promediar
- **Boosting:** entrenar modelos secuencialmente, cada uno corrige al anterior

### 3. Random Forest
- = Muchos árboles de decisión + bagging + features aleatorias
- Reduce overfitting drásticamente
- Muy bueno como primer modelo para probar

### 4. Gradient Boosting (XGBoost, LightGBM)
- Estado del arte para datos tabulares
- Gana la mayoría de competencias de Kaggle con datos tabulares
- Más complejo de tunear que Random Forest

### 5. ¿Cuándo usar qué?
- **Datos tabulares:** Random Forest o XGBoost (casi siempre ganan)
- **Imágenes/audio/texto:** Redes neuronales (siguiente unidad)
- **Pocos datos:** Modelos simples (regresión, árboles)
- **Necesito explicar:** Árboles de decisión o regresión logística

## Recursos
- [Visual intro to Decision Trees](https://www.r2d3.us/visual-intro-to-machine-learning-part-1/)
- [StatQuest: Random Forests (YouTube)](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ)
