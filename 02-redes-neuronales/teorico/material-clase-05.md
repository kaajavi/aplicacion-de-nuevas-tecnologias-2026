# Clase 5 — El Perceptrón y las Redes Neuronales Multicapa

## Unidad 2: Redes Neuronales | Aplicación de Nuevas Tecnologías — ITS Villada

---

## 1. Inspiración biológica: de la neurona al perceptrón

Antes de meternos en código, entendamos de dónde viene la idea. Tu cerebro tiene aproximadamente **86 mil millones de neuronas**. Cada neurona:

1. Recibe señales eléctricas por sus **dendritas** (entradas).
2. Procesa esas señales en el **soma** (cuerpo celular).
3. Si la señal acumulada supera un **umbral**, dispara un impulso eléctrico por el **axón** (salida).
4. Ese impulso llega a otras neuronas a través de **sinapsis** (conexiones).

Las sinapsis no son todas iguales: algunas son más fuertes que otras. Tu cerebro **aprende** reforzando o debilitando esas conexiones. Cuando estudiás para un examen y repetís algo muchas veces, estás fortaleciendo ciertas sinapsis.

En 1943, **McCulloch y Pitts** propusieron un modelo matemático simplificado de la neurona. Años después, en 1958, **Frank Rosenblatt** lo convirtió en algo entrenable: el **perceptrón**.

---

## 2. El Perceptrón: la neurona artificial

El perceptrón es la unidad más simple de una red neuronal. Hace exactamente esto:

```
output = activación(x₁·w₁ + x₂·w₂ + ... + xₙ·wₙ + b)
```

Desglosemos cada parte:

### Inputs (entradas) — `x₁, x₂, ..., xₙ`

Son los datos que le das. Por ejemplo, si querés predecir si un alumno aprueba:
- `x₁` = horas de estudio
- `x₂` = asistencia a clase (%)
- `x₃` = nota del parcial anterior

### Pesos (weights) — `w₁, w₂, ..., wₙ`

Cada input tiene un peso asociado. El peso indica **qué tan importante** es esa entrada para la decisión final. Si `w₁ = 0.8` y `w₂ = 0.1`, significa que las horas de estudio importan mucho más que la asistencia.

**Analogía:** Pensá en un jurado de un concurso de cocina. Cada jurado (input) da un puntaje, pero no todos los jurados pesan igual. El chef estrella tiene más peso que el crítico novato.

### Bias (sesgo) — `b`

Es un valor extra que se suma antes de la activación. Permite que el modelo se ajuste mejor. Sin bias, la frontera de decisión siempre pasaría por el origen.

**Analogía:** Es como la nota mínima para aprobar. Aunque tengas buenos inputs, si el bias es muy negativo, necesitás más "evidencia" para que la neurona se active.

### Suma ponderada

```
z = x₁·w₁ + x₂·w₂ + ... + xₙ·wₙ + b
```

Esto en notación vectorial es simplemente:

```
z = w⃗ · x⃗ + b
```

Un producto punto más un escalar. Nada más.

### Función de activación

La suma ponderada `z` puede dar cualquier número real. La función de activación decide qué hacer con ese número. En el perceptrón original, era una función escalón:

```
output = 1  si z ≥ 0
output = 0  si z < 0
```

### Ejemplo concreto en Python

```python
import numpy as np

# Datos de un alumno: [horas_estudio, asistencia%, nota_anterior]
x = np.array([6, 0.85, 7])

# Pesos aprendidos
w = np.array([0.5, 0.3, 0.2])

# Bias
b = -4.0

# Suma ponderada
z = np.dot(x, w) + b  # 6*0.5 + 0.85*0.3 + 7*0.2 + (-4) = 3 + 0.255 + 1.4 - 4 = 0.655

# Activación (escalón)
output = 1 if z >= 0 else 0
print(f"z = {z:.3f}, output = {output}")  # z = 0.655, output = 1 → Aprueba
```

---

## 3. El perceptrón como clasificador lineal

¿Qué hace realmente un perceptrón? Traza una **línea recta** (o un hiperplano en más dimensiones) que separa los datos en dos grupos.

Imaginá un gráfico 2D donde:
- Eje X = horas de estudio
- Eje Y = nota anterior
- Puntos rojos = desaprueba
- Puntos azules = aprueba

El perceptrón encuentra la mejor línea recta que separe rojos de azules. Por eso se llama **clasificador lineal**.

### ¿Qué puede resolver?

El perceptrón puede resolver perfectamente las compuertas lógicas **AND** y **OR**:

```
AND:                    OR:
(0,0) → 0              (0,0) → 0
(0,1) → 0              (0,1) → 1
(1,0) → 0              (1,0) → 1
(1,1) → 1              (1,1) → 1
```

Si graficás estos puntos en 2D, podés trazar una línea que separe los 0s de los 1s en ambos casos.

---

## 4. La limitación XOR — Minsky & Papert (1969)

Ahora mirá la compuerta **XOR**:

```
XOR:
(0,0) → 0
(0,1) → 1
(1,0) → 1
(1,1) → 0
```

Graficá estos puntos. Los 1s están en diagonal opuesta a los 0s. **No hay ninguna línea recta que los separe.** Probá: no se puede.

En 1969, **Marvin Minsky** y **Seymour Papert** publicaron el libro *"Perceptrons"* donde demostraron matemáticamente esta limitación. Su conclusión: el perceptrón no puede resolver problemas que no sean linealmente separables.

Este libro tuvo un efecto devastador. Muchos investigadores abandonaron las redes neuronales. Se vino lo que se conoce como el **"invierno de la IA"** — casi una década donde nadie invertía en esta línea de investigación.

**La ironía:** Minsky y Papert sabían que redes multicapa podían resolver XOR, pero argumentaron que no había un método eficiente para entrenarlas. Tenían razón... hasta que apareció backpropagation.

---

## 5. Multi-Layer Perceptron (MLP): capas ocultas al rescate

La solución al problema XOR es agregar **capas ocultas** (hidden layers) entre la entrada y la salida. Esto es el **Multi-Layer Perceptron** o MLP.

### Arquitectura de un MLP

```
Capa de Entrada → Capa(s) Oculta(s) → Capa de Salida
   (inputs)         (procesamiento)      (predicción)
```

Cada capa tiene múltiples neuronas, y cada neurona de una capa se conecta con todas las neuronas de la capa siguiente (**fully connected** o **dense**).

### ¿Cómo resuelve XOR?

Con una capa oculta de 2 neuronas:
- La primera neurona oculta aprende algo parecido a OR.
- La segunda neurona oculta aprende algo parecido a NAND.
- La neurona de salida combina ambas (AND de los resultados).

OR(x₁, x₂) AND NAND(x₁, x₂) = XOR(x₁, x₂)

**Cada capa oculta permite al modelo aprender fronteras de decisión más complejas.** Una capa oculta puede aprender fronteras curvas. Dos capas pueden aprender regiones arbitrarias. Es como pasar de dibujar con una regla a dibujar a mano alzada.

### MLP para XOR en PyTorch

```python
import torch
import torch.nn as nn

# Datos XOR
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

# Modelo MLP: 2 entradas → 4 neuronas ocultas → 1 salida
model = nn.Sequential(
    nn.Linear(2, 4),    # capa oculta
    nn.ReLU(),           # activación
    nn.Linear(4, 1),    # capa de salida
    nn.Sigmoid()         # output entre 0 y 1
)

# Entrenamiento
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    output = model(X)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Resultado
with torch.no_grad():
    print(model(X).round())
# tensor([[0.], [1.], [1.], [0.]])  ← ¡XOR resuelto!
```

---

## 6. Funciones de activación

Las funciones de activación son **fundamentales**. Sin ellas, un MLP de 100 capas sería equivalente a un solo perceptrón (porque la composición de funciones lineales es una función lineal). La activación introduce **no linealidad**, que es lo que permite aprender patrones complejos.

### ReLU (Rectified Linear Unit)

```
ReLU(z) = max(0, z)
```

- Si `z > 0`, devuelve `z`.
- Si `z ≤ 0`, devuelve `0`.

**¿Por qué es tan popular?**
- Súper simple y rápida de calcular.
- No tiene el problema de "gradiente que se desvanece" (para valores positivos).
- Es la opción por defecto en la mayoría de redes modernas.

**Desventaja:** Las neuronas con entrada negativa quedan "muertas" (output siempre 0). Variantes como **Leaky ReLU** lo solucionan dejando pasar un poquito: `LeakyReLU(z) = max(0.01z, z)`.

### Sigmoid (Sigmoide)

```
σ(z) = 1 / (1 + e⁻ᶻ)
```

- Aplasta cualquier valor al rango **(0, 1)**.
- Útil para **probabilidades** en la capa de salida (clasificación binaria).

**Desventaja:** Para valores muy grandes o muy chicos, el gradiente es casi cero → el modelo deja de aprender (**vanishing gradient**). Por eso ya no se usa mucho en capas ocultas.

### Tanh (Tangente hiperbólica)

```
tanh(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)
```

- Rango **(-1, 1)** — centrado en cero, lo cual ayuda al entrenamiento.
- Similar a sigmoid pero mejor porque las salidas negativas ayudan en la optimización.

**Desventaja:** Mismo problema de vanishing gradient que sigmoid para valores extremos.

### Softmax

```
softmax(zᵢ) = eᶻⁱ / Σⱼ eᶻʲ
```

- Convierte un vector de valores en una **distribución de probabilidad** (todos suman 1).
- Se usa en la **capa de salida para clasificación multiclase**.

**Ejemplo:** Si tenés 3 clases (gato, perro, pájaro) y la red produce `[2.0, 1.0, 0.1]`, softmax lo convierte en `[0.66, 0.24, 0.10]` → 66% de probabilidad de que sea gato.

### Resumen visual

| Función | Rango | Uso principal | Cuándo usarla |
|---------|-------|---------------|---------------|
| ReLU | [0, ∞) | Capas ocultas | Default para casi todo |
| Sigmoid | (0, 1) | Salida binaria | Clasificación sí/no |
| Tanh | (-1, 1) | Capas ocultas (legacy) | RNNs, normalización |
| Softmax | (0, 1), suma=1 | Salida multiclase | Clasificar entre N clases |

```python
import torch.nn.functional as F

z = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

print("ReLU:   ", F.relu(z))        # [0, 0, 0, 1, 2]
print("Sigmoid:", torch.sigmoid(z))   # [0.12, 0.27, 0.50, 0.73, 0.88]
print("Tanh:   ", torch.tanh(z))      # [-0.96, -0.76, 0.00, 0.76, 0.96]

logits = torch.tensor([2.0, 1.0, 0.1])
print("Softmax:", F.softmax(logits, dim=0))  # [0.66, 0.24, 0.10]
```

---

## 7. Forward Pass: cómo fluyen los datos

El **forward pass** (pasada hacia adelante) es el proceso de pasar datos de entrada a través de toda la red hasta obtener una predicción. Es pura álgebra lineal + activaciones.

### Paso a paso

Para una red con una capa oculta:

**Paso 1:** Capa oculta
```
Z₁ = X · W₁ + b₁       ← multiplicación de matrices + bias
A₁ = activación(Z₁)      ← aplicar ReLU, sigmoid, etc.
```

**Paso 2:** Capa de salida
```
Z₂ = A₁ · W₂ + b₂
A₂ = activación(Z₂)      ← esta es la predicción final
```

### ¿Por qué multiplicación de matrices?

Porque es **eficiente**. En vez de calcular neurona por neurona, hacés una sola operación matricial que calcula todas las neuronas de una capa de golpe. Las GPUs son monstruosamente buenas para multiplicar matrices, y por eso las usamos para entrenar redes.

### Ejemplo numérico completo

Supongamos una red: 3 inputs → 2 neuronas ocultas (ReLU) → 1 salida (Sigmoid)

```python
import numpy as np

# Input: un alumno [horas_estudio=6, asistencia=0.85, nota=7]
X = np.array([[6, 0.85, 7]])  # shape (1, 3)

# Capa oculta: 3 inputs → 2 neuronas
W1 = np.array([[ 0.3, -0.1],
                [ 0.5,  0.8],
                [-0.2,  0.4]])  # shape (3, 2)
b1 = np.array([0.1, -0.3])     # shape (2,)

# Capa de salida: 2 neuronas → 1 salida
W2 = np.array([[0.6],
                [0.9]])          # shape (2, 1)
b2 = np.array([-0.5])           # shape (1,)

# Forward pass
Z1 = X @ W1 + b1                # (1,3) @ (3,2) = (1,2)
A1 = np.maximum(0, Z1)          # ReLU

Z2 = A1 @ W2 + b2               # (1,2) @ (2,1) = (1,1)
A2 = 1 / (1 + np.exp(-Z2))      # Sigmoid

print(f"Z1 = {Z1}")
print(f"A1 (ReLU) = {A1}")
print(f"Z2 = {Z2}")
print(f"A2 (Sigmoid) = {A2}")
print(f"Predicción: {'Aprueba' if A2 > 0.5 else 'Desaprueba'}")
```

### Dimensiones de las matrices

Entender las dimensiones es clave para no perderte:

```
Input:  (batch_size, n_features)
W1:     (n_features, n_hidden)
b1:     (n_hidden,)
Z1:     (batch_size, n_hidden)

W2:     (n_hidden, n_output)
b2:     (n_output,)
Z2:     (batch_size, n_output)
```

**Regla de oro:** la cantidad de columnas de una matriz tiene que coincidir con la cantidad de filas de la siguiente. Si no coincide, te va a explotar con un error de dimensiones (y vas a ver ese error *muchas* veces en tu carrera).

### Forward Pass en PyTorch

```python
import torch
import torch.nn as nn

class MiRed(nn.Module):
    def __init__(self):
        super().__init__()
        self.capa1 = nn.Linear(3, 2)   # 3 inputs → 2 ocultas
        self.capa2 = nn.Linear(2, 1)   # 2 ocultas → 1 salida

    def forward(self, x):
        x = torch.relu(self.capa1(x))  # capa oculta + ReLU
        x = torch.sigmoid(self.capa2(x))  # salida + Sigmoid
        return x

modelo = MiRed()
entrada = torch.tensor([[6.0, 0.85, 7.0]])
prediccion = modelo(entrada)  # esto ejecuta forward()
print(f"Predicción: {prediccion.item():.4f}")
```

Fijate que en PyTorch no necesitás manejar las matrices a mano. `nn.Linear` se encarga de los pesos y el bias. Pero es **fundamental** que entiendas qué pasa adentro.

---

## 8. Juntando todo: el panorama completo

Hasta acá vimos las piezas fundamentales:

1. **Perceptrón** = neurona artificial (inputs × pesos + bias → activación → output)
2. **Clasificador lineal** = lo que puede hacer un solo perceptrón
3. **Limitación XOR** = hay problemas que una sola neurona no puede resolver
4. **MLP** = apilar capas de neuronas para resolver problemas complejos
5. **Funciones de activación** = la no linealidad que hace posible aprender curvas
6. **Forward pass** = el recorrido de los datos a través de la red

Pero nos falta algo crucial: **¿cómo aprende la red?** ¿Cómo se ajustan los pesos para que las predicciones sean buenas? Eso lo vamos a ver en la próxima clase con **funciones de pérdida**, **gradient descent** y **backpropagation**.

---

## 📝 Conceptos clave para recordar

- **Perceptrón:** Neurona artificial. Calcula `z = Σ(xᵢ · wᵢ) + b` y aplica una función de activación.
- **Pesos (weights):** Determinan la importancia de cada input. Se ajustan durante el entrenamiento.
- **Bias:** Permite desplazar la frontera de decisión. Sin él, todo pasa por el origen.
- **Clasificador lineal:** El perceptrón solo puede separar datos con una línea/hiperplano.
- **Limitación XOR (Minsky & Papert, 1969):** Un perceptrón no puede resolver problemas no linealmente separables. Esto frenó la investigación en redes neuronales por años.
- **MLP (Multi-Layer Perceptron):** Agregar capas ocultas permite resolver problemas no lineales como XOR.
- **Funciones de activación:** Introducen no linealidad. Sin ellas, mil capas equivalen a una.
  - **ReLU:** Default para capas ocultas. Simple y efectiva.
  - **Sigmoid:** Salida entre 0 y 1. Para clasificación binaria.
  - **Softmax:** Distribución de probabilidad. Para clasificación multiclase.
  - **Tanh:** Salida entre -1 y 1. Centrada en cero.
- **Forward pass:** El recorrido de datos de entrada a salida. Multiplicación de matrices + activaciones en cada capa.
- **Dimensiones:** Si `W` tiene shape `(a, b)`, necesitás que la entrada tenga `a` columnas. La salida tendrá `b` columnas.
