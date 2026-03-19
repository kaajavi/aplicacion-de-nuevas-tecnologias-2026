# Clase 4: Árboles de Decisión, Ensembles y Gradient Boosting

## Introducción

En la clase anterior vimos modelos lineales (regresión lineal y logística). Son potentes pero tienen una limitación fuerte: asumen relaciones lineales. En la vida real, muchas relaciones son más complejas. Hoy vamos a ver los **árboles de decisión** y sus evoluciones (**Random Forest, XGBoost, LightGBM**), que son los modelos más usados en la industria para datos tabulares.

Si tuvieras que apostar cuál es el tipo de modelo más exitoso en competencias de Kaggle con datos tabulares, la respuesta es **gradient boosting** (que vamos a ver en esta clase).

---

## Árboles de Decisión

### La idea intuitiva

Un árbol de decisión es exactamente lo que suena: **una serie de preguntas con respuestas sí/no que van dividiendo los datos** hasta llegar a una decisión final.

Pensalo como el juego de las 20 preguntas:
- ¿El animal tiene más de 4 patas? → Sí
- ¿Vive en el agua? → No
- ¿Es un insecto? → Sí
- → Es una hormiga.

Para un modelo que predice si un cliente va a cancelar su suscripción:

```
¿Usa el servicio menos de 2 veces por mes?
├── Sí → ¿Lleva más de 12 meses?
│        ├── Sí → CANCELA (85% de probabilidad)
│        └── No → NO CANCELA (60%)
└── No → ¿Tuvo algún reclamo?
         ├── Sí → CANCELA (70%)
         └── No → NO CANCELA (95%)
```

### Estructura de un árbol

- **Nodo raíz (root):** la primera pregunta, la que mejor divide los datos.
- **Nodos internos:** preguntas intermedias.
- **Hojas (leaves):** las decisiones finales (predicciones).
- **Ramas:** las respuestas a cada pregunta.
- **Profundidad (depth):** cuántos niveles de preguntas tiene el árbol.

### ¿Cómo decide el árbol qué pregunta hacer primero?

Acá viene lo interesante. El árbol necesita elegir: ¿cuál feature uso? ¿En qué valor corto? La respuesta: **elige la pregunta que mejor separa los datos en grupos "puros"**.

### Splits (divisiones)

Para features numéricos, el árbol prueba distintos puntos de corte:
- ¿Edad > 25? ¿Edad > 30? ¿Edad > 35?
- Para cada corte, evalúa qué tan bien separa las clases.

Para features categóricos:
- ¿Barrio = Centro? ¿Barrio ∈ {Centro, Alberdi}?

### Índice Gini (Impureza de Gini)

Es la métrica más común para medir qué tan "pura" es una división. Mide la probabilidad de clasificar incorrectamente un elemento si lo clasificás al azar según la distribución del nodo.

```
Gini = 1 - Σ(pᵢ²)
```

Donde pᵢ es la proporción de cada clase en el nodo.

**Ejemplo:**
- Nodo con 50% clase A y 50% clase B → Gini = 1 - (0.5² + 0.5²) = **0.5** (máxima impureza, no separa nada)
- Nodo con 90% clase A y 10% clase B → Gini = 1 - (0.9² + 0.1²) = **0.18** (bastante puro)
- Nodo con 100% clase A → Gini = 1 - (1² + 0²) = **0.0** (perfecto, totalmente puro)

**Cuanto menor el Gini, más puro el nodo.**

### Information Gain (Ganancia de Información)

Otra forma de medir la calidad de un split es usando **entropía** (concepto de teoría de la información):

```
Entropía = -Σ(pᵢ × log₂(pᵢ))
```

- Entropía máxima (= 1 para binario) → datos totalmente mezclados
- Entropía 0 → datos perfectamente separados

**Information Gain = Entropía del nodo padre - Entropía promedio ponderada de los nodos hijos**

El árbol elige el split que maximiza el Information Gain (o minimiza el Gini, según la configuración).

En la práctica, **Gini y entropía dan resultados muy similares**. Scikit-learn usa Gini por defecto porque es un poco más rápido de calcular.

### Implementación en código

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Entrenar
arbol = DecisionTreeClassifier(
    max_depth=4,           # profundidad máxima
    min_samples_split=10,  # mínimo de muestras para dividir un nodo
    min_samples_leaf=5,    # mínimo de muestras en una hoja
    random_state=42
)
arbol.fit(X_train, y_train)

# Predecir
y_pred = arbol.predict(X_test)
print(classification_report(y_test, y_pred))

# Visualizar el árbol (¡una gran ventaja!)
plt.figure(figsize=(20, 10))
plot_tree(arbol, feature_names=X.columns, class_names=["No", "Sí"],
          filled=True, rounded=True, fontsize=10)
plt.title("Árbol de Decisión")
plt.tight_layout()
plt.show()
```

### Ventajas de los árboles de decisión

- **Interpretables:** podés visualizar el árbol y entender exactamente por qué tomó cada decisión.
- **No necesitan escalado:** no les importa si un feature va de 0 a 1 o de 0 a 1.000.000.
- **Manejan datos numéricos y categóricos.**
- **Rápidos de entrenar.**

### Desventajas

- **Overfitting:** sin control, un árbol puede crecer hasta memorizar cada dato de entrenamiento. Un árbol sin límite de profundidad sobre un dataset con 1000 filas puede tener 1000 hojas — una por fila.
- **Inestables:** un pequeño cambio en los datos puede generar un árbol completamente diferente.
- **No son los más precisos:** individualmente, un árbol rara vez es el mejor modelo.

> **Analogía:** Un árbol de decisión es como un empleado que sigue un diagrama de flujo. Es fácil de entender y auditar, pero un solo diagrama puede ser demasiado rígido o demasiado específico.

---

## Ensembles: La Sabiduría de las Multitudes

La solución a las debilidades de un árbol individual es simple y poderosa: **usar muchos árboles juntos**.

La idea detrás de los **ensembles** (conjuntos de modelos) se basa en un principio estadístico: si combinás las predicciones de muchos modelos diversos, el resultado es mejor que cualquier modelo individual.

> **Analogía:** Si le preguntás a una sola persona una trivia, puede acertar o no. Si le preguntás a 100 personas y tomás la respuesta más votada, la probabilidad de acertar es mucho mayor. Esto es literalmente la "sabiduría de las multitudes".

Hay dos estrategias principales para crear ensembles:

### Bagging (Bootstrap Aggregating)

1. Tomás **muestras aleatorias con reemplazo** del dataset de entrenamiento (bootstrap).
2. Entrenás un modelo diferente en cada muestra.
3. Para predecir, combinás las predicciones (voto mayoritario para clasificación, promedio para regresión).

**Clave:** cada modelo ve datos ligeramente diferentes, así que cometen errores diferentes. Al combinarlos, los errores se cancelan.

```
Dataset original: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

Muestra 1: [2, 3, 3, 5, 7, 7, 8, 9, 9, 10] → Árbol 1
Muestra 2: [1, 1, 2, 4, 5, 6, 8, 8, 9, 10] → Árbol 2
Muestra 3: [1, 3, 4, 4, 5, 6, 7, 9, 10, 10] → Árbol 3

Predicción final = voto mayoritario de los 3 árboles
```

### Boosting

A diferencia del bagging (donde los modelos se entrenan en paralelo e independiente), en boosting **los modelos se entrenan secuencialmente**, y cada nuevo modelo intenta corregir los errores del anterior.

1. Entrenás un modelo simple.
2. Identificás los datos donde se equivocó.
3. Entrenás un segundo modelo que **le presta más atención a esos errores**.
4. Repetís: cada modelo se enfoca en lo que los anteriores no pudieron resolver.
5. La predicción final es una combinación ponderada de todos los modelos.

> **Analogía bagging vs boosting:** 
> - **Bagging** es como tener 100 estudiantes resolviendo el mismo examen independientemente y luego promediando sus respuestas.
> - **Boosting** es como tener un estudiante que resuelve el examen, después un tutor que revisa solo las preguntas que falló y le explica, después otro tutor que revisa lo que el anterior no pudo resolver, y así sucesivamente.

---

## Random Forest

Random Forest es la implementación más famosa de **bagging con árboles de decisión**, con un truco adicional.

### ¿Qué lo hace "Random"?

Dos fuentes de aleatoriedad:

1. **Bootstrap de filas:** cada árbol se entrena con una muestra aleatoria del dataset (como en bagging normal).
2. **Subconjunto aleatorio de features:** en cada split de cada árbol, solo se considera un subconjunto random de features (no todas).

Este segundo punto es clave. Si tenés 20 features, en cada split el árbol solo puede elegir entre, digamos, 5 features aleatorias. Esto hace que los árboles sean **más diversos** entre sí, lo cual mejora el ensemble.

### Implementación

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Crear y entrenar
rf = RandomForestClassifier(
    n_estimators=100,       # cantidad de árboles
    max_depth=10,           # profundidad máxima de cada árbol
    min_samples_leaf=5,     # mínimo de muestras en hoja
    max_features="sqrt",    # features a considerar por split: √(total_features)
    random_state=42,
    n_jobs=-1               # usar todos los cores del CPU
)
rf.fit(X_train, y_train)

# Predecir
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))

# Importancia de features
importancias = pd.Series(rf.feature_importances_, index=X.columns)
importancias.sort_values(ascending=False).head(10).plot(kind="barh")
plt.title("Top 10 Features más importantes")
plt.xlabel("Importancia")
plt.show()
```

### Ventajas de Random Forest

- **Muy bueno "out of the box":** funciona bien con parámetros por defecto.
- **Resistente al overfitting** (mucho más que un árbol individual).
- **Maneja bien datos faltantes y outliers.**
- **Te da importancia de features** gratis.
- **Paralelizable:** los árboles se entrenan en paralelo.

### Desventajas

- **Más lento que un solo árbol** (entrenás cientos de árboles).
- **Menos interpretable:** no podés "ver" un solo árbol de decisión claro.
- **En general, no es el mejor posible** — gradient boosting suele ganarle un poco.

---

## Gradient Boosting

Gradient Boosting es la familia de modelos que **domina las competencias de ML con datos tabulares** desde hace años. Es la evolución de boosting aplicada con gradient descent.

### ¿Cómo funciona?

1. Empezás con una predicción simple (por ejemplo, el promedio de todos los valores).
2. Calculás los **residuos** (errores) de esa predicción.
3. Entrenás un árbol para predecir **los residuos** (no los valores originales).
4. Sumás las predicciones del árbol a tu predicción actual.
5. Calculás los nuevos residuos.
6. Repetís: cada árbol nuevo intenta predecir lo que los anteriores no pudieron.

**Formalmente:** cada árbol se entrena en la dirección del **gradiente negativo de la función de pérdida** (de ahí el nombre "gradient" boosting).

```
Predicción final = predicción inicial + η×árbol₁ + η×árbol₂ + ... + η×árbolₙ
```

Donde **η (eta)** es el learning rate: cuánto peso le das a cada árbol nuevo.

### XGBoost (eXtreme Gradient Boosting)

Creado por Tianqi Chen en 2014, XGBoost es la implementación más famosa de gradient boosting. Ganó una cantidad absurda de competencias en Kaggle y es usado en producción por empresas como Airbnb, Uber, y muchas más.

**¿Qué lo hace especial?**
- **Regularización** incorporada (L1 y L2) para evitar overfitting.
- **Paralelización** del proceso de split.
- **Manejo nativo de valores faltantes** (no necesitás imputar NaN).
- **Poda de árboles** más inteligente.
- Soporte para **GPU**.

```python
# Instalar: pip install xgboost
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

xgb = XGBClassifier(
    n_estimators=200,        # cantidad de árboles
    max_depth=6,             # profundidad de cada árbol
    learning_rate=0.1,       # η: cuánto aporta cada árbol
    subsample=0.8,           # proporción de filas por árbol
    colsample_bytree=0.8,    # proporción de features por árbol
    random_state=42,
    eval_metric="logloss"
)

xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
print(classification_report(y_test, y_pred))
```

### LightGBM (Light Gradient Boosting Machine)

Desarrollado por Microsoft en 2017. Hace lo mismo que XGBoost pero es **significativamente más rápido**, especialmente con datasets grandes.

**¿Por qué es más rápido?**
- **Histogram-based splitting:** en vez de probar cada valor posible como punto de corte, agrupa los valores en histogramas y prueba los bordes de los bins. Mucho más rápido con poca pérdida de precisión.
- **Leaf-wise growth:** en vez de crecer nivel por nivel (como XGBoost), crece hoja por hoja, expandiendo siempre la hoja que reduce más el error. Esto produce árboles más asimétricos pero más eficientes.
- **Soporte nativo para datos categóricos** (no necesitás hacer one-hot encoding).

```python
# Instalar: pip install lightgbm
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report

lgbm = LGBMClassifier(
    n_estimators=200,
    max_depth=-1,            # sin límite (controla con num_leaves)
    num_leaves=31,           # complejidad del árbol
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1               # silenciar output
)

lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)
print(classification_report(y_test, y_pred))
```

### CatBoost (mencion honorable)

Desarrollado por Yandex (el "Google ruso"). Su ventaja principal es el manejo nativo de variables categóricas sin necesidad de encoding. Es excelente "out of the box" con menos tuning necesario.

---

## Comparación: ¿Cuándo usar qué?

| Modelo | Velocidad | Precisión | Interpretabilidad | Cuándo usarlo |
|--------|-----------|-----------|-------------------|---------------|
| **Árbol de Decisión** | Muy rápido | Media | Alta | Cuando necesitás explicar cada decisión. Prototipos rápidos. |
| **Random Forest** | Rápido | Alta | Media | Buen punto de partida. Cuando querés algo robusto sin mucho tuning. |
| **XGBoost** | Medio | Muy alta | Baja | Competencias. Cuando necesitás maximizar rendimiento. |
| **LightGBM** | Rápido | Muy alta | Baja | Datasets grandes. Producción donde la velocidad importa. |
| **Regresión Logística** | Muy rápido | Media-Baja | Alta | Baseline. Cuando la relación es aproximadamente lineal. |

### Receta práctica para un proyecto nuevo:

1. **Empezá con un baseline simple:** regresión logística o un árbol de decisión. Esto te da un "piso" para comparar.
2. **Probá Random Forest:** buen rendimiento con poco esfuerzo.
3. **Si necesitás más:** pasá a XGBoost o LightGBM con tuning de hiperparámetros.
4. **Si el dataset es enorme (millones de filas):** usá LightGBM (es el más rápido).
5. **Si tenés muchas variables categóricas:** considerá CatBoost.

### Tuning de hiperparámetros

Los modelos de boosting tienen muchos parámetros que ajustar. Los más importantes:

- **n_estimators:** cantidad de árboles. Más = mejor hasta cierto punto, después overfittea.
- **learning_rate:** paso del boosting. Más bajo = necesitás más árboles pero generaliza mejor.
- **max_depth:** profundidad de cada árbol. Controla la complejidad.
- **subsample:** fracción de datos por árbol (introduce aleatoriedad).
- **colsample_bytree:** fracción de features por árbol.

**Regla de oro:** learning_rate bajo + muchos estimadores suele dar los mejores resultados.

```python
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.3],
}

grid = GridSearchCV(
    XGBClassifier(random_state=42, eval_metric="logloss"),
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)
print(f"Mejores parámetros: {grid.best_params_}")
print(f"Mejor F1: {grid.best_score_:.4f}")

# Evaluar con los mejores parámetros
mejor_modelo = grid.best_estimator_
y_pred = mejor_modelo.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## Ejemplo Integrador

Para cerrar, veamos cómo se ve un pipeline completo comparando modelos:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report

# Cargar datos
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Definir modelos
modelos = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss"),
    "LightGBM": LGBMClassifier(n_estimators=100, random_state=42, verbose=-1),
}

# Nota: para Logistic Regression necesitamos escalar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Evaluar cada modelo
print("=" * 60)
for nombre, modelo in modelos.items():
    # Logistic Regression usa datos escalados, el resto no necesita
    if "Logistic" in nombre:
        modelo.fit(X_train_scaled, y_train)
        scores = cross_val_score(modelo, X_train_scaled, y_train, cv=5, scoring="f1")
        y_pred = modelo.predict(X_test_scaled)
    else:
        modelo.fit(X_train, y_train)
        scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring="f1")
        y_pred = modelo.predict(X_test)
    
    print(f"\n{nombre}")
    print(f"  CV F1: {scores.mean():.4f} (+/- {scores.std():.4f})")
    print(f"  Test: {classification_report(y_test, y_pred, output_dict=True)['weighted avg']['f1-score']:.4f}")
print("=" * 60)
```

---

## 📝 Conceptos clave para recordar

1. **Árboles de decisión** dividen los datos con preguntas sí/no sucesivas. Son interpretables pero propensos al overfitting.

2. **Splits:** el árbol elige la pregunta (feature + valor de corte) que mejor separa las clases.

3. **Gini** mide la impureza de un nodo (0 = puro, 0.5 = máxima impureza binaria). **Information Gain** mide cuánta entropía se reduce con un split.

4. **Ensembles** combinan muchos modelos para obtener uno mejor. Principio: la sabiduría de las multitudes.

5. **Bagging:** modelos en paralelo, cada uno entrenado con una muestra aleatoria distinta. Reduce varianza.

6. **Boosting:** modelos secuenciales, cada uno corrige los errores del anterior. Reduce bias.

7. **Random Forest = Bagging + árboles + subconjunto aleatorio de features.** Robusto, buen default.

8. **Gradient Boosting** entrena árboles secuencialmente, cada uno prediciendo los residuos del anterior.

9. **XGBoost:** gradient boosting optimizado con regularización. El "clásico" de competencias ML.

10. **LightGBM:** más rápido que XGBoost gracias a histogramas y crecimiento leaf-wise. Ideal para datasets grandes.

11. **Hiperparámetros clave:** `n_estimators` (cantidad de árboles), `learning_rate` (paso), `max_depth` (complejidad), `subsample` y `colsample_bytree` (aleatoriedad).

12. **Learning rate bajo + muchos árboles** suele dar mejores resultados que learning rate alto + pocos árboles.

13. **Receta práctica:** empezá con un baseline simple → Random Forest → XGBoost/LightGBM con tuning.

14. **Los árboles y sus ensembles no necesitan escalado de features.** Ventaja sobre modelos lineales.

15. **GridSearchCV** te permite buscar los mejores hiperparámetros de forma sistemática usando cross-validation.
