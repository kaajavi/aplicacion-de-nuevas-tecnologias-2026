# Clase 2: Datos — El Combustible del Machine Learning

## Introducción

Hay una frase en el mundo del ML que dice: **"Garbage in, garbage out"** (basura entra, basura sale). Podés tener el algoritmo más sofisticado del mundo, pero si los datos son malos, los resultados van a ser malos.

En esta clase vamos a entender cómo se estructuran los datos, cómo prepararlos, y cómo dividirlos correctamente para entrenar y evaluar un modelo. **Esta es probablemente la clase más importante de toda la unidad**, porque en la vida real un data scientist pasa el 80% del tiempo limpiando y preparando datos, y solo el 20% entrenando modelos.

---

## Anatomía de un Dataset

Un dataset (conjunto de datos) es simplemente una tabla. Pensalo como un Excel o un Google Sheets. Tiene:

### Filas (muestras/observaciones/registros)

Cada fila es **un ejemplo individual**. Si estás trabajando con datos de casas, cada fila es una casa. Si son pacientes, cada fila es un paciente.

### Columnas (variables/atributos)

Cada columna es **una característica** de ese ejemplo. Para casas: metros cuadrados, cantidad de habitaciones, barrio, precio.

### Features (características) vs. Label (etiqueta/target)

Esta distinción es **clave** en aprendizaje supervisado:

- **Features (X):** las columnas que usás como entrada para hacer la predicción. Son las "pistas" que le das al modelo.
- **Label / Target (y):** la columna que querés predecir. Es la "respuesta correcta".

**Ejemplo concreto:**

| m² | habitaciones | barrio | antigüedad | precio (USD) |
|----|-------------|--------|------------|-------------|
| 80 | 3 | Nueva Córdoba | 5 | 95.000 |
| 120 | 4 | Cerro | 20 | 78.000 |
| 60 | 2 | Centro | 2 | 110.000 |

- **Features:** m², habitaciones, barrio, antigüedad
- **Label:** precio (es lo que queremos predecir)

En pandas, esto se ve así:

```python
import pandas as pd

df = pd.read_csv("casas.csv")

# Separar features y label
X = df[["m2", "habitaciones", "barrio", "antiguedad"]]  # Features
y = df["precio"]  # Label / Target
```

---

## Datos Numéricos vs. Categóricos

No todas las columnas son iguales. Es fundamental entender los tipos de datos:

### Datos Numéricos

Son valores que representan cantidades y tienen un orden matemático:
- **Continuos:** pueden tomar cualquier valor (temperatura: 23.5°C, precio: $95.430)
- **Discretos:** solo valores enteros (cantidad de habitaciones: 3, cantidad de hijos: 2)

Los algoritmos de ML trabajan nativamente con números. Les encantan.

### Datos Categóricos

Representan categorías o grupos. No tienen un orden numérico natural:
- **Nominales:** no tienen orden (color: rojo/azul/verde, barrio: Centro/Cerro/Alberdi)
- **Ordinales:** tienen un orden pero no una distancia uniforme (educación: primario < secundario < universitario)

**El problema:** la mayoría de los algoritmos de ML **no entienden strings**. No podés pasarle "Nueva Córdoba" a un modelo y esperar que funcione. Hay que convertirlos a números (lo vemos más abajo en encoding).

> **Analogía:** Pensá en datos numéricos como cosas que podés medir con una regla. Datos categóricos son cosas que podés etiquetar con una etiqueta. No podés medir un color con una regla.

---

## Preparación de Datos

Acá es donde se gasta la mayor parte del tiempo. Los datos del mundo real son **sucios, incompletos y desordenados**.

### 1. Manejo de valores faltantes (NaN)

En la vida real, los datasets siempre tienen datos faltantes. Un paciente no reportó su peso, una casa no tiene registrada la antigüedad, etc.

**Detectar NaN:**
```python
import pandas as pd

df = pd.read_csv("datos.csv")

# Ver cuántos NaN hay por columna
print(df.isnull().sum())

# Ver porcentaje de NaN
print(df.isnull().mean() * 100)
```

**Estrategias para manejar NaN:**

**a) Eliminar filas con NaN:**
```python
df_limpio = df.dropna()
```
Simple pero peligroso: si tenés muchos NaN, perdés muchos datos.

**b) Rellenar con la media/mediana/moda:**
```python
# Numéricos: rellenar con la mediana (más robusta que la media ante outliers)
df["edad"].fillna(df["edad"].median(), inplace=True)

# Categóricos: rellenar con la moda (el valor más frecuente)
df["barrio"].fillna(df["barrio"].mode()[0], inplace=True)
```

**c) Rellenar con un valor constante:**
```python
df["comentario"].fillna("Sin comentario", inplace=True)
```

**d) Usar técnicas avanzadas:** imputación con KNN o modelos (scikit-learn tiene `SimpleImputer` y `KNNImputer`).

```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
df[["edad", "ingreso"]] = imputer.fit_transform(df[["edad", "ingreso"]])
```

**Regla de oro:** si una columna tiene más del 50-60% de NaN, probablemente conviene eliminarla directamente.

### 2. Normalización y Escalado

Los algoritmos de ML son sensibles a la **escala** de los datos. Si una columna va de 0 a 1 (ej: porcentaje) y otra va de 0 a 1.000.000 (ej: precio), el modelo va a darle más importancia a la segunda simplemente porque los números son más grandes.

**Min-Max Scaling (normalización al rango 0-1):**
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df[["edad", "ingreso"]] = scaler.fit_transform(df[["edad", "ingreso"]])
# Ahora ambas columnas van de 0 a 1
```

**StandardScaler (estandarización: media=0, desvío=1):**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[["edad", "ingreso"]] = scaler.fit_transform(df[["edad", "ingreso"]])
# Ahora tienen media 0 y desvío estándar 1
```

**¿Cuándo usar cada uno?**
- **MinMaxScaler:** cuando necesitás valores en un rango específico (redes neuronales, KNN)
- **StandardScaler:** cuando el algoritmo asume distribución normal (regresión lineal, SVM)
- **Árboles de decisión y Random Forest:** no necesitan escalado (son invariantes a la escala)

> **Analogía:** Si comparás el rendimiento de un futbolista mirando goles (0-30) y pases (0-2000), parecería que los pases importan más solo porque el número es más grande. Normalizar es como convertir todo a "puntos sobre 100" para que la comparación sea justa.

### 3. Encoding (codificación de variables categóricas)

Como dijimos, los modelos necesitan números. Hay varias formas de convertir categorías a números:

**a) Label Encoding:** asigna un número a cada categoría.
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["barrio_encoded"] = le.fit_transform(df["barrio"])
# Centro → 0, Cerro → 1, Nueva Córdoba → 2
```

**Problema:** el modelo puede interpretar que "Nueva Córdoba (2) > Cerro (1) > Centro (0)", y eso no tiene sentido para datos nominales.

**b) One-Hot Encoding:** crea una columna binaria por cada categoría.
```python
df_encoded = pd.get_dummies(df, columns=["barrio"])
```

El resultado:

| m² | barrio_Centro | barrio_Cerro | barrio_NuevaCba |
|----|--------------|-------------|-----------------|
| 80 | 0 | 0 | 1 |
| 120 | 0 | 1 | 0 |
| 60 | 1 | 0 | 0 |

**Ventaja:** no introduce orden falso.
**Desventaja:** si tenés una columna con 500 categorías distintas, creás 500 columnas nuevas (explosión dimensional).

**c) Ordinal Encoding:** para variables ordinales donde el orden sí importa.
```python
from sklearn.preprocessing import OrdinalEncoder

oe = OrdinalEncoder(categories=[["primario", "secundario", "universitario"]])
df["educacion_encoded"] = oe.fit_transform(df[["educacion"]])
# primario → 0, secundario → 1, universitario → 2 (el orden tiene sentido)
```

### 4. Feature Engineering (ingeniería de características)

Esto es **el arte de crear nuevas columnas** a partir de las existentes para darle mejor información al modelo.

**Ejemplos:**
```python
# A partir de fecha de nacimiento, calcular edad
df["edad"] = 2026 - df["anio_nacimiento"]

# A partir de m² y precio, calcular precio por m²
df["precio_por_m2"] = df["precio"] / df["m2"]

# A partir de fecha, extraer día de la semana
df["dia_semana"] = pd.to_datetime(df["fecha"]).dt.dayofweek

# Crear columna binaria
df["es_nuevo"] = (df["antiguedad"] < 5).astype(int)
```

Un buen feature engineering puede mejorar muchísimo el rendimiento de un modelo. Es donde el **conocimiento del dominio** (saber del tema) hace la diferencia.

> **Analogía:** El feature engineering es como cuando un chef no solo usa los ingredientes crudos, sino que prepara reducciones, mezcla especias, y crea salsas. Los ingredientes base son los mismos, pero la preparación hace toda la diferencia.

---

## Train/Test Split: Dividir para Evaluar

### El problema fundamental

Si entrenás un modelo con TODOS tus datos y después evaluás con los mismos datos, ¿qué pasa? El modelo va a tener un rendimiento excelente... **en esos datos**. Pero no sabés si funciona con datos nuevos que nunca vio.

Es como estudiar solo con el examen resuelto: vas a sacar 10 en ese examen, pero no sabés si aprendiste el tema.

### La solución: Train/Test Split

Dividís tus datos en dos partes:
- **Training set (70-80%):** para entrenar el modelo.
- **Test set (20-30%):** para evaluar el modelo. Estos datos **nunca** los ve durante el entrenamiento.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,    # 20% para test
    random_state=42    # para reproducibilidad
)

print(f"Entrenamiento: {X_train.shape[0]} muestras")
print(f"Test: {X_test.shape[0]} muestras")
```

**`random_state=42`**: es una semilla para que la división sea siempre la misma. Así tus resultados son reproducibles. (¿Por qué 42? Es una referencia a *The Hitchhiker's Guide to the Galaxy*. No tiene nada de especial técnicamente.)

### Validation Set (conjunto de validación)

A veces necesitás un tercer conjunto:

- **Training set (60-70%):** para entrenar
- **Validation set (10-15%):** para ajustar hiperparámetros y elegir el mejor modelo
- **Test set (20-25%):** evaluación final, **se usa UNA SOLA VEZ al final**

```python
# Primer split: separar test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Segundo split: separar validation del training
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.15, random_state=42)
```

**¿Para qué el validation?** Cuando estás probando distintos modelos o ajustando parámetros, usás el validation set para comparar. El test set lo reservás para la evaluación final — si lo usás para ajustar, deja de ser una evaluación honesta.

### Cross-Validation (Validación Cruzada)

¿Y si la división que hiciste fue "de suerte"? ¿Qué pasa si justo los datos fáciles quedaron en test?

**K-Fold Cross-Validation** resuelve esto:

1. Dividís los datos en **K partes** (típicamente K=5 o K=10).
2. Entrenás K veces, cada vez usando una parte distinta como validación y el resto como entrenamiento.
3. Promediás los resultados.

```
Fold 1: [VAL] [TRAIN] [TRAIN] [TRAIN] [TRAIN]
Fold 2: [TRAIN] [VAL] [TRAIN] [TRAIN] [TRAIN]
Fold 3: [TRAIN] [TRAIN] [VAL] [TRAIN] [TRAIN]
Fold 4: [TRAIN] [TRAIN] [TRAIN] [VAL] [TRAIN]
Fold 5: [TRAIN] [TRAIN] [TRAIN] [TRAIN] [VAL]
```

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

modelo = LinearRegression()
scores = cross_val_score(modelo, X, y, cv=5, scoring="r2")

print(f"Scores por fold: {scores}")
print(f"Score promedio: {scores.mean():.4f} (+/- {scores.std():.4f})")
```

**Ventaja:** estimación más robusta del rendimiento real del modelo.
**Desventaja:** tarda K veces más (entrenás K modelos en vez de uno).

---

## Data Leakage (fuga de datos)

Este es uno de los errores **más comunes y más peligrosos** en ML. Y lo peor: a veces es difícil de detectar.

**Data leakage ocurre cuando información del test set "se filtra" al training set**, haciendo que el modelo parezca mejor de lo que realmente es.

### Ejemplo 1: Normalizar antes de dividir

```python
# ❌ MAL - Data leakage!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Calcula media y desvío con TODOS los datos
X_train, X_test = train_test_split(X_scaled, ...)
# El scaler ya "vio" los datos de test al calcular la media y el desvío
```

```python
# ✅ BIEN - Sin leakage
X_train, X_test = train_test_split(X, ...)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Calcula solo con train
X_test_scaled = scaler.transform(X_test)         # Aplica sin recalcular
```

### Ejemplo 2: Feature que contiene la respuesta

Supongamos que querés predecir si un paciente tiene diabetes y en tu dataset hay una columna "medicación_diabetes". Esa columna ya te dice la respuesta — es leakage.

### Ejemplo 3: Datos del futuro

Querés predecir las ventas de mañana y usás un feature que incluye el precio de cierre de las acciones de mañana. En producción, no vas a tener ese dato.

> **Analogía:** Data leakage es como hacer un examen con las respuestas escritas en la mano. Sacás 10, pero no aprendiste nada. Cuando te toque un examen de verdad (datos nuevos en producción), te va a ir mal.

**Regla de oro:** todo procesamiento de datos (normalización, imputación, encoding) se **ajusta (fit)** solo con el training set y se **aplica (transform)** al test set.

---

## Overfitting vs. Underfitting

Estos son los dos grandes males del Machine Learning. Entenderlos es **fundamental**.

### Overfitting (sobreajuste)

El modelo **memorizó** los datos de entrenamiento en vez de aprender patrones generales. Funciona increíble en training pero pésimo en datos nuevos.

**Señales:**
- Accuracy en training: 99%
- Accuracy en test: 65%
- Gran diferencia entre rendimiento en train y test

**Causas:**
- Modelo demasiado complejo para la cantidad de datos
- Entrenar demasiadas épocas (en redes neuronales)
- Pocas muestras de entrenamiento
- Demasiadas features irrelevantes

**Soluciones:**
- Más datos de entrenamiento
- Modelo más simple
- Regularización (L1, L2)
- Eliminación de features irrelevantes
- Cross-validation
- Early stopping (en deep learning)

> **Analogía:** Es como el alumno que se memoriza las respuestas del parcial anterior sin entender los conceptos. Si le cambiás los números, no puede resolver nada.

### Underfitting (subajuste)

El modelo es **demasiado simple** y no captura los patrones de los datos. Funciona mal en todo: en training y en test.

**Señales:**
- Accuracy en training: 55%
- Accuracy en test: 52%
- Rendimiento bajo en ambos sets

**Causas:**
- Modelo demasiado simple para el problema
- Features insuficientes o irrelevantes
- No entrenar lo suficiente

**Soluciones:**
- Modelo más complejo
- Más features / mejor feature engineering
- Entrenar más tiempo
- Reducir la regularización

> **Analogía:** Es como un alumno que lee el resumen del resumen la noche anterior al examen. No entendió lo suficiente para resolver nada.

### El punto justo (sweet spot)

Lo que buscamos es un modelo que **generalice** bien: que funcione bien tanto en datos de entrenamiento como en datos nuevos.

```
Complejidad baja → Underfitting → "No aprendió suficiente"
Complejidad media → Buen ajuste → "Aprendió los patrones"
Complejidad alta → Overfitting → "Memorizó los datos"
```

El gráfico clásico tiene forma de U: el error en test baja cuando el modelo se complejiza, llega a un mínimo, y después vuelve a subir (porque empieza a sobreajustar). Ese mínimo es el sweet spot.

```python
# Ejemplo: comparar complejidad con árboles de decisión
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

for profundidad in [1, 3, 5, 10, 20, None]:
    modelo = DecisionTreeClassifier(max_depth=profundidad)
    modelo.fit(X_train, y_train)
    
    acc_train = accuracy_score(y_train, modelo.predict(X_train))
    acc_test = accuracy_score(y_test, modelo.predict(X_test))
    
    print(f"Profundidad {str(profundidad):>4s} | Train: {acc_train:.3f} | Test: {acc_test:.3f}")
```

Si ves que train sube y test baja, estás overfitteando. Si ambos son bajos, estás underfitteando.

---

## 📝 Conceptos clave para recordar

1. **Dataset = tabla.** Filas son muestras, columnas son variables. Features (X) son las entradas, Label (y) es lo que querés predecir.

2. **Datos numéricos** (continuos/discretos) vs **categóricos** (nominales/ordinales). Los modelos necesitan números.

3. **Manejo de NaN:** eliminar, rellenar con media/mediana/moda, o imputar. Nunca ignorar los NaN.

4. **Normalización:** escalar features para que ninguna domine solo por su magnitud. MinMaxScaler (0-1), StandardScaler (media=0, desvío=1).

5. **Encoding:** convertir categorías a números. LabelEncoder (ordinal), One-Hot (nominal), OrdinalEncoder (cuando hay orden real).

6. **Feature Engineering:** crear nuevas columnas a partir de las existentes. El conocimiento del dominio es tu mejor herramienta.

7. **Train/Test Split:** SIEMPRE dividir los datos. Train para aprender, test para evaluar. Nunca evaluar con datos de entrenamiento.

8. **Validation set:** para ajustar hiperparámetros. El test set se usa solo al final.

9. **Cross-Validation (K-Fold):** entrenar K veces con distintas divisiones para una estimación más robusta.

10. **Data Leakage:** información del test que se filtra al train. Error grave y sutil. Fit solo en train, transform en test.

11. **Overfitting:** el modelo memoriza → excelente en train, malo en test. Solución: simplificar, más datos, regularización.

12. **Underfitting:** el modelo no aprende → malo en todo. Solución: más complejidad, mejores features, entrenar más.

13. **El 80% del trabajo en ML es preparación de datos.** El modelo es solo la última parte.
