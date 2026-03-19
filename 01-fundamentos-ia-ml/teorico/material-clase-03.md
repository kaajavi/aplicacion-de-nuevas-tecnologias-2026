# Clase 3: Regresión Lineal, Regresión Logística y Métricas de Clasificación

## Introducción

En esta clase vamos a ver los dos modelos más fundamentales de Machine Learning supervisado: **regresión lineal** (para predecir números) y **regresión logística** (para clasificar). Son modelos simples, pero entenderlos bien es la base para todo lo que viene después. Además, vamos a aprender a medir qué tan bien funciona un clasificador, porque "accuracy" no siempre cuenta toda la historia.

---

## Regresión Lineal

### La idea: encontrar la mejor línea

Supongamos que tenés datos de departamentos en Córdoba: m² y precio. Si los graficás, vas a ver una tendencia: a más metros cuadrados, más caro. La regresión lineal busca **la línea recta que mejor se ajusta a esos puntos**.

La ecuación de una recta es:

```
y = mx + b
```

Donde:
- **y** es lo que querés predecir (precio)
- **x** es el feature (m²)
- **m** es la pendiente (cuánto sube el precio por cada m² extra)
- **b** es la ordenada al origen (el "precio base")

En ML, usamos otra notación equivalente:

```
ŷ = w₁x + w₀
```

Donde **w₁** es el peso (weight) y **w₀** es el bias.

### ¿Qué significa "la mejor línea"?

Hay infinitas líneas que podés dibujar. ¿Cómo elegís la mejor? Necesitás una forma de medir qué tan mal está cada línea. Ahí entra el **error**.

Para cada punto, calculás la diferencia entre el valor real (y) y el valor predicho (ŷ). Esa diferencia es el **residuo**.

### MSE (Mean Squared Error - Error Cuadrático Medio)

La métrica más usada en regresión:

```
MSE = (1/n) × Σ(yᵢ - ŷᵢ)²
```

1. Para cada punto, calculás el error: (valor real - valor predicho)
2. Lo elevás al cuadrado (para que los errores negativos no cancelen los positivos, y para penalizar más los errores grandes)
3. Promediás todos los errores al cuadrado

**Cuanto menor el MSE, mejor el modelo.**

También vas a ver el **RMSE** (raíz del MSE), que tiene la ventaja de estar en las mismas unidades que la variable que predecís (si predecís precio en dólares, el RMSE está en dólares).

```python
from sklearn.metrics import mean_squared_error
import numpy as np

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
```

Otra métrica importante: **R² (R cuadrado)** — indica qué proporción de la variabilidad de los datos es explicada por el modelo. Va de 0 a 1 (1 = predicción perfecta, 0 = el modelo no explica nada).

```python
from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)
print(f"R²: {r2:.4f}")  # 0.85 significa que el modelo explica el 85% de la variación
```

### Gradient Descent (Descenso por Gradiente)

¿Cómo encuentra el modelo los mejores valores de w₁ y w₀? No prueba todas las combinaciones posibles (sería infinito). Usa un algoritmo llamado **gradient descent**.

**La analogía de la montaña:**

Imaginate que estás en una montaña con niebla. No podés ver el valle (el punto más bajo), pero podés sentir la pendiente del suelo bajo tus pies. ¿Qué hacés? Das un paso en la dirección donde baja más. Repetís. Eventualmente llegás al valle.

Eso es exactamente gradient descent:

1. Empezás con valores aleatorios para los pesos (w₁, w₀).
2. Calculás el MSE.
3. Calculás el gradiente (la "pendiente" del error respecto a cada peso).
4. Ajustás los pesos un poquito en la dirección que reduce el error.
5. Repetís hasta que el error deja de bajar significativamente.

El **learning rate** (tasa de aprendizaje) controla qué tan grande es cada paso:
- **Muy grande:** das pasos enormes y podés "pasarte" del mínimo, saltando de un lado al otro sin converger.
- **Muy chico:** convergés, pero tardás una eternidad.
- **Justo:** convergés eficientemente al mínimo.

> **Analogía extendida:** El learning rate es como el tamaño de tus pasos en la montaña. Pasos muy grandes y te pasás del valle. Pasos muy chiquitos y tardás horas en bajar. Necesitás un paso razonable.

Para la regresión lineal simple, hay una solución analítica exacta (ecuación normal), así que no siempre necesitás gradient descent. Pero para modelos más complejos (redes neuronales), gradient descent es el método estándar.

### Regresión Lineal Múltiple

En la vida real, rara vez tenés un solo feature. La regresión lineal múltiple usa **varias variables** para predecir:

```
ŷ = w₁x₁ + w₂x₂ + w₃x₃ + ... + w₀
```

Para el ejemplo de las casas:
```
precio = w₁(m²) + w₂(habitaciones) + w₃(antigüedad) + w₀
```

En código es exactamente igual:

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Datos
X = df[["m2", "habitaciones", "antiguedad", "distancia_centro"]]
y = df["precio"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar
modelo = LinearRegression()
modelo.fit(X_train, y_train)

# Predecir
y_pred = modelo.predict(X_test)

# Evaluar
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.4f}")

# Ver los pesos aprendidos
for feature, peso in zip(X.columns, modelo.coef_):
    print(f"  {feature}: {peso:.2f}")
print(f"  Intercepto (w₀): {modelo.intercept_:.2f}")
```

Los pesos te dicen la importancia relativa de cada feature y cómo influye en la predicción.

### Limitaciones de la regresión lineal

- **Asume relación lineal:** si la relación entre X e y no es una línea recta (es una curva), la regresión lineal va a funcionar mal.
- **Sensible a outliers:** un par de puntos extremos pueden torcer mucho la línea.
- **No sirve para clasificación:** si querés predecir "sí/no", necesitás otra cosa...

---

## Regresión Logística

### De números a categorías

A pesar del nombre, la regresión logística **NO es un modelo de regresión**. Es un modelo de **clasificación**. El nombre confunde, pero es por razones históricas.

**Problema:** querés predecir si un email es spam o no. La salida es binaria: 0 (no spam) o 1 (spam). Si usás regresión lineal, podría darte valores como -0.3 o 2.7, que no tienen sentido como probabilidades.

### La función sigmoide

La regresión logística toma la salida de una regresión lineal y la pasa por una **función sigmoide** que "aplasta" cualquier número al rango (0, 1):

```
σ(z) = 1 / (1 + e^(-z))
```

Donde z = w₁x₁ + w₂x₂ + ... + w₀ (la misma combinación lineal de antes).

La sigmoide tiene forma de S:
- Si z es muy negativo → σ(z) ≈ 0
- Si z = 0 → σ(z) = 0.5
- Si z es muy positivo → σ(z) ≈ 1

**El resultado es una probabilidad:** "este email tiene un 87% de probabilidad de ser spam".

### El umbral de decisión

La sigmoide te da una probabilidad, pero vos necesitás una decisión: ¿spam o no spam? Para eso usás un **umbral** (threshold), típicamente 0.5:

- Si P(spam) ≥ 0.5 → clasificar como spam
- Si P(spam) < 0.5 → clasificar como no spam

**Pero el umbral no tiene que ser 0.5.** Podés cambiarlo según el contexto:
- **Detección de cáncer:** mejor usar un umbral más bajo (0.3) para no perder ningún caso positivo, aunque tengas más falsos positivos.
- **Filtro de spam:** un umbral más alto (0.7) para no mandar emails legítimos a spam.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Entrenar
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# Predecir clases
y_pred = modelo.predict(X_test)

# Predecir probabilidades
y_proba = modelo.predict_proba(X_test)
# Devuelve algo como: [[0.13, 0.87], [0.92, 0.08], ...]
# Columna 0: P(clase 0), Columna 1: P(clase 1)

# Usar un umbral personalizado
umbral = 0.3
y_pred_custom = (y_proba[:, 1] >= umbral).astype(int)
```

> **Analogía:** La regresión logística es como un termómetro que en vez de medir temperatura mide "qué tan probable es que algo pase". La sigmoide convierte cualquier número a una probabilidad entre 0 y 1. El umbral es la línea que vos dibujás: "arriba de este número, digo que sí".

---

## Métricas de Clasificación

Acá se pone interesante. Cuando evaluás un clasificador, **accuracy sola no alcanza**. Veamos por qué.

### Accuracy (exactitud)

```
Accuracy = predicciones correctas / total de predicciones
```

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
```

**¿Cuál es el problema?** Imaginá un dataset de fraude bancario donde el 99% de las transacciones son legítimas y solo el 1% es fraude. Un modelo que **siempre diga "no fraude"** tiene 99% de accuracy. Pero no detecta ningún fraude — es completamente inútil.

Esto pasa con **datasets desbalanceados**, que son muy comunes en la vida real.

### La Matriz de Confusión

Para entender de verdad cómo se comporta un clasificador, necesitás la **matriz de confusión**:

```
                    Predicho
                  Positivo  Negativo
Real  Positivo  [   TP    |   FN    ]
      Negativo  [   FP    |   TN    ]
```

- **TP (True Positive):** predijo positivo y era positivo ✅
- **TN (True Negative):** predijo negativo y era negativo ✅
- **FP (False Positive):** predijo positivo pero era negativo ❌ (falsa alarma)
- **FN (False Negative):** predijo negativo pero era positivo ❌ (se le escapó)

```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualizar
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Fraude", "Fraude"])
disp.plot(cmap="Blues")
plt.title("Matriz de Confusión")
plt.show()
```

**Ejemplo concreto:** Detector de fraude con 1000 transacciones.

```
                    Predicho
                  Fraude    No Fraude
Real  Fraude    [   8    |    2     ]   → 10 fraudes reales
      No Fraude [  15    |   975    ]   → 990 legítimas
```

- **TP = 8:** detectó 8 fraudes correctamente
- **FN = 2:** se le escaparon 2 fraudes (¡peligroso!)
- **FP = 15:** marcó 15 transacciones legítimas como fraude (molesto, pero no grave)
- **TN = 975:** identificó correctamente 975 transacciones legítimas

Accuracy = (8 + 975) / 1000 = 98.3%. Suena bien, pero se le escaparon 2 fraudes de 10. ¿Eso es aceptable?

### Precision (Precisión)

**De todos los que marqué como positivo, ¿cuántos realmente lo eran?**

```
Precision = TP / (TP + FP)
```

En nuestro ejemplo: 8 / (8 + 15) = **34.8%**

Baja precision = muchas falsas alarmas.

### Recall (Sensibilidad / Exhaustividad)

**De todos los que realmente eran positivos, ¿cuántos detecté?**

```
Recall = TP / (TP + FN)
```

En nuestro ejemplo: 8 / (8 + 2) = **80%**

Bajo recall = se te escapan muchos positivos.

### F1-Score

Es la **media armónica** de precision y recall. Es un balance entre ambas:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

En nuestro ejemplo: 2 × (0.348 × 0.8) / (0.348 + 0.8) = **0.485**

El F1 penaliza cuando una de las dos métricas es muy baja. Solo es alto cuando **ambas** son razonablemente buenas.

```python
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_pred):.4f}")
print(f"F1: {f1_score(y_test, y_pred):.4f}")

# O todo junto, bien presentado:
print(classification_report(y_test, y_pred, target_names=["No Fraude", "Fraude"]))
```

`classification_report` te da un resumen hermoso con precision, recall, F1 y support (cantidad de muestras) para cada clase.

---

## ¿Cuándo importa más Precision vs. Recall?

Esta es una pregunta de parcial seguro, y más importante: es una decisión de diseño real en proyectos.

### Priorizar RECALL cuando...

**El costo de un falso negativo es muy alto** (no detectar algo que sí existía).

- **Detección de cáncer:** si el modelo dice "no tiene cáncer" y el paciente sí tiene, las consecuencias son gravísimas. Preferís que el modelo sea "paranoico" y marque más casos como positivos (aunque algunos sean falsas alarmas que después un médico descarta).
- **Detección de fraude:** dejar pasar un fraude real es peor que bloquear una transacción legítima.
- **Sistemas de seguridad:** un detector de intrusos que no detecta intrusos no sirve para nada.

> **Mantra:** "Mejor prevenir que curar" → priorizá recall.

### Priorizar PRECISION cuando...

**El costo de un falso positivo es muy alto** (marcar algo como positivo cuando no lo era).

- **Filtro de spam:** si un email legítimo importante termina en spam, podés perder un negocio o una oportunidad. Preferís dejar pasar algún spam antes que perder un email real.
- **Sistema judicial:** condenar a un inocente es peor que dejar libre a un culpable (en teoría).
- **Recomendaciones de contenido:** recomendar algo ofensivo o inapropiado es peor que no recomendar algo bueno.

> **Mantra:** "Si vas a acusar, asegurate" → priorizá precision.

### El trade-off

Precision y recall están **inversamente relacionados**. Si subís el umbral de decisión (ej: de 0.5 a 0.8), clasificás menos cosas como positivas → sube la precision (las que marcás son más seguras) pero baja el recall (se te escapan más positivos).

Si bajás el umbral (ej: de 0.5 a 0.2), marcás más cosas como positivas → sube el recall (atrapás más positivos) pero baja la precision (más falsas alarmas).

**No podés maximizar ambas a la vez.** Tenés que decidir según el contexto del problema.

```python
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt

# Obtener precision y recall para distintos umbrales
precision_vals, recall_vals, umbrales = precision_recall_curve(y_test, y_proba[:, 1])

plt.figure(figsize=(8, 5))
plt.plot(umbrales, precision_vals[:-1], label="Precision")
plt.plot(umbrales, recall_vals[:-1], label="Recall")
plt.xlabel("Umbral")
plt.ylabel("Score")
plt.title("Precision vs Recall según Umbral")
plt.legend()
plt.grid(True)
plt.show()
```

---

## Ejemplo Completo: Juntando Todo

Veamos un ejemplo end-to-end que combina lo que aprendimos:

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)

# 1. Cargar datos (ejemplo: dataset de diabetes de sklearn)
from sklearn.datasets import load_diabetes
# Para clasificación usamos otro dataset clásico:
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # 0 = maligno, 1 = benigno

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Escalar (fit solo en train!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Entrenar
modelo = LogisticRegression(max_iter=1000)
modelo.fit(X_train_scaled, y_train)

# 5. Predecir
y_pred = modelo.predict(X_test_scaled)

# 6. Evaluar
print("=== Resultados ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
print()
print("Matriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
print()
print("Reporte Completo:")
print(classification_report(y_test, y_pred, target_names=["Maligno", "Benigno"]))
```

Notá el parámetro `stratify=y` en el split: esto asegura que la proporción de clases se mantenga igual en train y test. Muy importante cuando las clases están desbalanceadas.

---

## 📝 Conceptos clave para recordar

1. **Regresión lineal** busca la mejor línea recta (ŷ = wx + b) para predecir un valor numérico continuo.

2. **MSE** mide el error promedio al cuadrado. **RMSE** es su raíz (en las mismas unidades que y). **R²** indica qué proporción de la variabilidad se explica (0 a 1).

3. **Gradient descent** encuentra los mejores pesos ajustándolos iterativamente en la dirección que reduce el error. El **learning rate** controla el tamaño del paso.

4. **Regresión múltiple** usa varios features: ŷ = w₁x₁ + w₂x₂ + ... + w₀

5. **Regresión logística** no es regresión: es clasificación. Usa la **función sigmoide** para convertir la salida lineal en una probabilidad (0 a 1).

6. **El umbral** (por defecto 0.5) decide la clase final. Ajustarlo cambia el balance entre precision y recall.

7. **Accuracy = correctos / total.** No alcanza cuando las clases están desbalanceadas.

8. **Matriz de confusión:** TP, TN, FP, FN. Es la base para entender todo lo demás.

9. **Precision = TP / (TP + FP)** → "De lo que dije positivo, ¿cuánto acerté?" Priorizala cuando las falsas alarmas son costosas.

10. **Recall = TP / (TP + FN)** → "De los positivos reales, ¿cuántos encontré?" Priorizalo cuando dejar pasar un positivo es peligroso.

11. **F1-Score** es el balance armónico entre precision y recall. Útil cuando necesitás una sola métrica de resumen.

12. **Precision y recall tienen un trade-off:** subir uno baja el otro. La decisión depende del contexto del problema.

13. **`classification_report`** de scikit-learn te da todo en una sola llamada. Usalo siempre.
