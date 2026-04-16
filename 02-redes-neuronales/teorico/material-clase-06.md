# Clase 6 — Entrenamiento: Pérdida, Gradient Descent y Backpropagation

## Unidad 2: Redes Neuronales | Aplicación de Nuevas Tecnologías — ITS Villada

---

## 1. ¿Cómo aprende una red neuronal?

En la clase anterior vimos cómo una red hace predicciones (forward pass). Pero los pesos iniciales son **aleatorios**, así que las primeras predicciones son basura. El proceso de entrenamiento consiste en:

1. **Hacer una predicción** (forward pass).
2. **Medir qué tan mal le fue** (función de pérdida).
3. **Calcular cómo ajustar cada peso** (backpropagation).
4. **Ajustar los pesos un poquito** (gradient descent).
5. **Repetir** miles o millones de veces.

Es como aprender a embocar una pelota en un aro. Tirás, ves si erraste (y por cuánto), ajustás el ángulo y la fuerza, y tirás de nuevo. Con suficientes tiros, embocás casi siempre.

---

## 2. Función de pérdida (Loss Function)

La función de pérdida (o **loss function** / **cost function**) es el número que le dice a la red **qué tan equivocada está**. Cuanto menor el loss, mejor el modelo.

### MSE — Mean Squared Error (para regresión)

Cuando tu modelo predice un **valor numérico continuo** (precio de una casa, temperatura, nota de un examen), usás MSE:

```
MSE = (1/n) Σ (yᵢ - ŷᵢ)²
```

Donde:
- `yᵢ` = valor real (lo que debería dar)
- `ŷᵢ` = predicción del modelo (lo que dio)
- `n` = cantidad de ejemplos

**¿Por qué al cuadrado?**
- Elimina los negativos (un error de -3 pesa igual que uno de +3).
- Penaliza más los errores grandes. Un error de 10 contribuye 100 al MSE, mientras que un error de 2 contribuye solo 4.

```python
import numpy as np

y_real = np.array([85, 72, 91, 60])      # notas reales
y_pred = np.array([82, 75, 88, 65])      # predicciones del modelo

mse = np.mean((y_real - y_pred) ** 2)
print(f"MSE = {mse}")  # MSE = 12.5
```

### Cross-Entropy — (para clasificación)

Cuando tu modelo clasifica (gato/perro, spam/no-spam, dígito 0-9), usás **Cross-Entropy**. La idea es distinta: medimos qué tan lejos está la **distribución de probabilidad** predicha de la real.

#### Binary Cross-Entropy (2 clases)

```
BCE = -(1/n) Σ [yᵢ · log(ŷᵢ) + (1 - yᵢ) · log(1 - ŷᵢ)]
```

Donde `y` es 0 o 1, y `ŷ` es la probabilidad predicha (entre 0 y 1).

**¿Por qué logaritmo?** Porque penaliza **mucho** cuando el modelo está confiado pero equivocado. Si la respuesta correcta es 1 y el modelo dice 0.99, el loss es bajísimo (~0.01). Pero si dice 0.01 (súper confiado de que no), el loss es enorme (~4.6).

```python
import torch
import torch.nn as nn

# El modelo predijo estas probabilidades
y_pred = torch.tensor([0.9, 0.1, 0.8, 0.3])
# Las etiquetas reales
y_real = torch.tensor([1.0, 0.0, 1.0, 0.0])

bce = nn.BCELoss()
loss = bce(y_pred, y_real)
print(f"BCE Loss = {loss.item():.4f}")  # bajo, porque acertó bastante
```

#### Categorical Cross-Entropy (N clases)

Para clasificación multiclase (dígitos 0-9, tipos de animal, etc.):

```
CCE = -Σ yᵢ · log(ŷᵢ)
```

En la práctica, `y` es one-hot encoded (`[0, 0, 1, 0]` para la clase 2), y `ŷ` son las probabilidades que salen del softmax.

```python
# En PyTorch se usa CrossEntropyLoss que combina softmax + cross-entropy
criterion = nn.CrossEntropyLoss()

# Logits crudos (antes de softmax) para 3 clases
logits = torch.tensor([[2.0, 1.0, 0.1]])   # el modelo "piensa" que es clase 0
target = torch.tensor([0])                    # efectivamente es clase 0

loss = criterion(logits, target)
print(f"Loss = {loss.item():.4f}")  # bajo, porque acertó
```

### ¿Cuál usar?

| Problema | Loss Function | Activación de salida |
|----------|--------------|---------------------|
| Regresión (predecir número) | MSE | Ninguna (lineal) |
| Clasificación binaria (sí/no) | Binary Cross-Entropy | Sigmoid |
| Clasificación multiclase | Categorical Cross-Entropy | Softmax |

---

## 3. Gradient Descent: bajando la montaña

Ahora que sabemos medir el error, necesitamos **minimizarlo**. Acá es donde entra **Gradient Descent** (descenso por gradiente).

### La analogía del paisaje montañoso

Imaginá que estás en una montaña con niebla espesa. No ves nada. Querés llegar al punto más bajo del valle. ¿Qué hacés?

1. **Sentís la pendiente del suelo bajo tus pies** (calculás el gradiente).
2. **Caminás en la dirección donde baja más** (dirección opuesta al gradiente).
3. **Dás un paso** de cierto tamaño (learning rate).
4. **Repetís** hasta que no podés bajar más.

Eso es exactamente gradient descent. El "paisaje" es la **superficie de la función de pérdida**, donde cada punto corresponde a una combinación de pesos, y la altura es el valor del loss.

### El gradiente

El **gradiente** es un vector de derivadas parciales que te dice:
- **Dirección:** hacia dónde crece más rápido la función.
- **Magnitud:** qué tan empinada es la pendiente.

Como queremos **minimizar** el loss, nos movemos en la **dirección contraria al gradiente**:

```
w_nuevo = w_viejo - learning_rate × gradiente
```

### Ejemplo visual en 1D

Si el loss en función de un peso `w` forma una parábola:

```
Loss
 |    *
 |   * *
 |  *   *
 | *     *
 |*       *
 +----------→ w
      ↑
   mínimo
```

- Si estás a la izquierda del mínimo, la derivada es negativa → restás un negativo → te movés a la derecha ✓
- Si estás a la derecha, la derivada es positiva → restás un positivo → te movés a la izquierda ✓

Siempre te acercás al mínimo. Elegante, ¿no?

### Implementación simple

```python
# Gradient descent para encontrar el mínimo de f(w) = (w - 3)²
# El mínimo está en w = 3

w = 10.0          # empezamos lejos
lr = 0.1          # learning rate

for i in range(50):
    grad = 2 * (w - 3)            # derivada de (w-3)²
    w = w - lr * grad              # actualización
    loss = (w - 3) ** 2
    if i % 10 == 0:
        print(f"Paso {i}: w = {w:.4f}, loss = {loss:.6f}")

# Paso 0:  w = 8.6000, loss = 31.360000
# Paso 10: w = 3.0872, loss = 0.007600
# Paso 20: w = 3.0010, loss = 0.000001
# Paso 30: w = 3.0000, loss = 0.000000
```

---

## 4. Learning Rate: el tamaño del paso

El **learning rate** (tasa de aprendizaje, `lr` o `η`) es quizás el **hiperparámetro más importante** del entrenamiento. Controla qué tan grande es cada paso de actualización.

### Learning rate muy alto

Si dás pasos enormes, podés **saltarte el mínimo** y rebotar de un lado al otro, sin converger nunca. Incluso podés **divergir** (el loss se va a infinito).

```
Loss
 |  ←*→  rebotando
 |  * * *
 | *     *
 |*       *
 +----------→ w
```

### Learning rate muy bajo

Si dás pasos microscópicos, eventualmente vas a llegar... pero te va a tomar **una eternidad**. Y podrías quedarte atrapado en un **mínimo local** (un vallecito que no es el punto más bajo de todo el paisaje).

```
Loss
 |*
 | *............→ avanzando a paso de tortuga
 |  *
 |   *
 +----------→ w
```

### El punto justo

Generalmente se empieza con valores como `0.001` o `0.01` y se ajusta experimentalmente. Valores comunes:

- **SGD:** `lr = 0.01` a `0.1`
- **Adam:** `lr = 0.001` (el default funciona bien la mayoría de las veces)

### Learning Rate Schedulers

En la práctica, es útil **cambiar el learning rate durante el entrenamiento**:

- **Step Decay:** Reducir el lr cada N epochs (ej: multiplicar por 0.1 cada 30 epochs).
- **Cosine Annealing:** El lr sigue una curva coseno, bajando suavemente.
- **ReduceLROnPlateau:** Si el loss deja de mejorar por X epochs, reducir el lr.
- **Warmup:** Empezar con un lr muy bajo e ir subiéndolo gradualmente. Útil con modelos grandes.

```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001)

# Reducir lr cuando el loss se estanca
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

for epoch in range(100):
    loss = train_one_epoch()
    scheduler.step(loss)  # el scheduler decide si bajar el lr
```

**Analogía:** Es como cuando estás estacionando. Arrancás moviéndote rápido para acercarte, pero cuando estás cerca del lugar, vas despacito para no chocar.

---

## 5. Backpropagation: la cadena de la culpa

**Backpropagation** (propagación hacia atrás) es el algoritmo que calcula el gradiente de cada peso en la red. Es la pieza clave que permite entrenar redes profundas. Fue popularizado por **Rumelhart, Hinton y Williams en 1986**, y fue lo que revivió las redes neuronales después del invierno de la IA.

### La intuición

Cuando el modelo se equivoca, necesitamos saber: **¿cuánta culpa tiene cada peso?**

Pensá en una fábrica con varias etapas de producción. Sale un producto defectuoso al final. ¿Quién tuvo la culpa? ¿La última etapa? ¿La primera? ¿Todas un poquito? Backpropagation responde exactamente eso: propaga el error desde la salida hacia atrás, capa por capa, asignando "culpa" a cada peso.

### La Chain Rule (regla de la cadena)

Backpropagation se basa en la **regla de la cadena** del cálculo. Si tenés funciones compuestas:

```
L = f(g(h(w)))
```

La derivada de L respecto de w es:

```
dL/dw = dL/df × df/dg × dg/dh × dh/dw
```

Cada capa de la red es una función. El loss es la función final. Backpropagation calcula estas derivadas en cadena, desde la última capa hasta la primera.

### Ejemplo paso a paso

Red simple: 1 input → 1 neurona oculta → 1 salida, sin bias, activación sigmoid.

```
Forward:
z₁ = x · w₁
a₁ = sigmoid(z₁)
z₂ = a₁ · w₂
ŷ = sigmoid(z₂)
L = (y - ŷ)²
```

Backpropagation (de atrás hacia adelante):

```
∂L/∂ŷ  = -2(y - ŷ)                           ← derivada del loss
∂L/∂z₂ = ∂L/∂ŷ × sigmoid'(z₂)               ← a través de la activación
∂L/∂w₂ = ∂L/∂z₂ × a₁                         ← gradiente del peso w₂
∂L/∂a₁ = ∂L/∂z₂ × w₂                         ← propagamos hacia atrás
∂L/∂z₁ = ∂L/∂a₁ × sigmoid'(z₁)               ← a través de la activación
∂L/∂w₁ = ∂L/∂z₁ × x                          ← gradiente del peso w₁
```

Ahora actualizamos:
```
w₁ = w₁ - lr × ∂L/∂w₁
w₂ = w₂ - lr × ∂L/∂w₂
```

### En código

```python
import torch

# Datos
x = torch.tensor(2.0, requires_grad=False)
y = torch.tensor(1.0)

# Pesos (con tracking de gradientes)
w1 = torch.tensor(0.5, requires_grad=True)
w2 = torch.tensor(-0.3, requires_grad=True)

# Forward pass
z1 = x * w1
a1 = torch.sigmoid(z1)
z2 = a1 * w2
y_hat = torch.sigmoid(z2)
loss = (y - y_hat) ** 2

print(f"Predicción: {y_hat.item():.4f}, Loss: {loss.item():.4f}")

# Backward pass — PyTorch calcula TODOS los gradientes automáticamente
loss.backward()

print(f"∂L/∂w1 = {w1.grad.item():.4f}")
print(f"∂L/∂w2 = {w2.grad.item():.4f}")

# Actualizar pesos
lr = 0.1
with torch.no_grad():
    w1 -= lr * w1.grad
    w2 -= lr * w2.grad
```

**La magia de PyTorch/TensorFlow:** Vos no tenés que calcular las derivadas a mano. El framework construye un **grafo computacional** durante el forward pass, y después recorre ese grafo de atrás para adelante calculando gradientes automáticamente. Esto se llama **autograd** (diferenciación automática).

### ¿Por qué "hacia atrás"?

Porque es **eficiente**. Si calculases los gradientes "hacia adelante" (desde los inputs), tendrías que recorrer toda la red una vez por cada peso. Con backpropagation, recorrés la red **una sola vez** de atrás hacia adelante y obtenés todos los gradientes. Para redes con millones de pesos, la diferencia es abismal.

---

## 6. Epochs, Batches e Iteraciones

Cuando tenés un dataset grande (miles o millones de ejemplos), no podés pasar todos los datos de una sola vez por la red. Acá entran estos tres conceptos:

### Epoch (época)

Una epoch es **una pasada completa por todo el dataset**. Si tenés 10,000 imágenes y entrenás durante 50 epochs, tu modelo ve cada imagen 50 veces.

### Batch (lote)

Un batch es un **subconjunto del dataset** que se procesa junto. En vez de pasar las 10,000 imágenes de golpe, las dividís en batches de (por ejemplo) 32.

### Iteración

Una iteración es **un paso de actualización de pesos** (un forward + backward + update). Si tenés 10,000 ejemplos y batches de 32:

```
Iteraciones por epoch = 10,000 / 32 ≈ 313
```

### Tipos de Gradient Descent

#### Batch Gradient Descent (BGD)
- Usa **todo el dataset** para cada actualización.
- Gradiente muy preciso, pero lento y necesita mucha memoria.
- `batch_size = todo el dataset`

#### Stochastic Gradient Descent (SGD)
- Usa **un solo ejemplo** por actualización.
- Muy ruidoso (el gradiente varía mucho), pero rápido.
- `batch_size = 1`

#### Mini-Batch Gradient Descent ⭐
- Usa un **subconjunto** (típicamente 16, 32, 64, 128, 256).
- **El estándar en la práctica.** Balance entre precisión y velocidad.
- El ruido del mini-batch actúa como **regularización** (ayuda a no sobreajustar).

```python
from torch.utils.data import DataLoader, TensorDataset

# Dataset de ejemplo
X_train = torch.randn(10000, 784)  # 10k imágenes de 28x28
y_train = torch.randint(0, 10, (10000,))  # 10 clases

dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Entrenamiento
model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10

for epoch in range(num_epochs):
    total_loss = 0
    for batch_X, batch_y in dataloader:  # cada iteración procesa 64 ejemplos
        # Forward
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward
        optimizer.zero_grad()  # limpiar gradientes del paso anterior
        loss.backward()        # calcular gradientes
        optimizer.step()       # actualizar pesos

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss promedio: {avg_loss:.4f}")
```

### ¿Qué batch size elegir?

| Batch size | Ventaja | Desventaja |
|-----------|---------|------------|
| Pequeño (8-32) | Más ruido → mejor generalización, menos memoria | Más lento por epoch, gradiente menos estable |
| Mediano (64-256) | Buen balance | — |
| Grande (512+) | Gradiente estable, aprovecha GPU al máximo | Puede generalizar peor, necesita más memoria |

**Regla práctica:** Empezá con 32 o 64. Si tu GPU tiene memoria de sobra, probá 128 o 256. Si el modelo no generaliza bien, bajá el batch size.

### El loop de entrenamiento completo

Para que quede claro el flujo:

```
Para cada epoch (1 a N):
    Para cada mini-batch del dataset:
        1. Forward pass → obtener predicción
        2. Calcular loss → qué tan mal le fue
        3. Backward pass → calcular gradientes (backpropagation)
        4. Actualizar pesos → gradient descent
    Fin mini-batches
    (Opcionalmente: evaluar en datos de validación)
Fin epochs
```

---

## 7. Optimizadores modernos

**SGD puro** funciona, pero es lento. Los optimizadores modernos agregan trucos para converger más rápido:

### SGD con Momentum

Agrega "inercia" al movimiento. Si venís bajando en una dirección, seguís con más fuerza. Evita quedar atrapado en mínimos locales.

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

### Adam (Adaptive Moment Estimation)

Combina momentum con learning rates adaptativos por parámetro. **Es el default que vas a usar el 90% de las veces.**

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

### AdamW

Adam con weight decay correcto. Preferido para modelos grandes y fine-tuning.

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```

---

## 📝 Conceptos clave para recordar

- **Función de pérdida:** Mide qué tan equivocado está el modelo. Menor = mejor.
  - **MSE:** Para regresión. Penaliza más los errores grandes.
  - **Cross-Entropy:** Para clasificación. Penaliza predicciones confiadas pero incorrectas.
- **Gradient Descent:** Algoritmo para minimizar el loss. Movete en dirección opuesta al gradiente.
  - `w_nuevo = w_viejo - lr × gradiente`
- **Learning Rate:** Tamaño del paso. Muy alto → diverge. Muy bajo → no aprende. Empezá con 0.001 (Adam) o 0.01 (SGD).
- **Schedulers:** Cambian el lr durante el entrenamiento. Empezá rápido, terminá despacio.
- **Backpropagation:** Calcula gradientes de atrás hacia adelante usando la regla de la cadena. Es lo que permite entrenar redes profundas.
- **Autograd:** PyTorch/TensorFlow calculan los gradientes automáticamente. Vos solo definís el forward pass.
- **Epoch:** Una pasada completa por el dataset.
- **Batch:** Subconjunto del dataset procesado junto. Tamaño típico: 32-256.
- **Iteración:** Un paso de actualización (forward + backward + update).
- **Mini-Batch GD:** El estándar. Balance entre eficiencia y calidad del gradiente.
- **Adam:** El optimizador por defecto. Rápido, estable, y funciona bien casi siempre.
- **Loop de entrenamiento:** `for epoch → for batch → forward → loss → backward → step`.
