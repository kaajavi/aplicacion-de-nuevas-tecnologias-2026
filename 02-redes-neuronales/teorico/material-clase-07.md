# Clase 7 — Redes Neuronales Convolucionales (CNN)

## Unidad 2: Redes Neuronales | Aplicación de Nuevas Tecnologías — ITS Villada

---

## 1. ¿Por qué no usar un MLP para imágenes?

En la clase anterior aprendimos a entrenar un MLP. Podrías pensar: "si un MLP puede aprender cualquier función, ¿por qué no lo usamos para clasificar imágenes?". Técnicamente podés, pero es una **pésima idea**. Veamos por qué.

### El problema de los parámetros

Una imagen de 224×224 píxeles en color (RGB) tiene:

```
224 × 224 × 3 = 150,528 valores de entrada
```

Si tu primera capa oculta tiene 1,000 neuronas:

```
150,528 × 1,000 = 150,528,000 pesos (¡150 millones!)
```

Y eso es solo la primera capa. Una red con varias capas tendría **miles de millones de parámetros** para una tarea simple. Esto causa:

- **Overfitting masivo:** Demasiados parámetros para la cantidad de datos. El modelo memoriza en vez de aprender.
- **Lentitud:** Entrenar millones de parámetros requiere enormes cantidades de GPU y tiempo.
- **Memoria:** No te alcanza la RAM/VRAM para procesar un solo batch.

### El problema de la estructura espacial

Peor aún: un MLP **aplana** la imagen en un vector 1D. Pierde toda la información de que los píxeles están organizados en una grilla 2D.

Para un MLP, un píxel en la esquina superior izquierda no tiene ninguna relación especial con su vecino. Pero para nosotros, los píxeles cercanos sí están relacionados: forman bordes, texturas, formas.

**Analogía:** Es como si alguien cortara una foto en 150,000 cuadraditos, los mezclara en una bolsa, y te pidiera que clasifiques qué hay en la foto. Imposible. Necesitás ver los píxeles **en su posición original**.

### El problema de la invarianza de posición

Si entrenás un MLP para reconocer gatos y todos los gatos del dataset están centrados, el modelo no va a reconocer un gato que esté en la esquina. El MLP aprende **posiciones fijas**, no patrones que se repiten.

Las CNN resuelven los tres problemas.

---

## 2. La operación de convolución

La convolución es la operación fundamental de las CNN. Es lo que les da el nombre.

### ¿Qué es un filtro/kernel?

Un **filtro** (o **kernel**) es una matriz pequeña (típicamente 3×3 o 5×5) que se desliza sobre la imagen. En cada posición, calcula un **producto punto** entre el filtro y el pedazo de imagen debajo.

```
Imagen (5×5):              Filtro (3×3):
┌─────────────────┐        ┌─────────┐
│ 1  0  1  0  1   │        │ 1  0  1 │
│ 0  1  0  1  0   │        │ 0  1  0 │
│ 1  0  1  0  1   │   *    │ 1  0  1 │
│ 0  1  0  1  0   │        └─────────┘
│ 1  0  1  0  1   │
└─────────────────┘
```

El filtro se posiciona en la esquina superior izquierda, multiplica elemento a elemento, y suma todo. Después se mueve un paso a la derecha y repite. Cuando llega al borde, baja una fila y empieza de nuevo. El resultado es un **mapa de características** (feature map).

### Ejemplo numérico

```python
import numpy as np

# Imagen 5x5 (simplificada, un canal)
imagen = np.array([
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1]
])

# Filtro 3x3 para detectar un patrón específico
filtro = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 0, 1]
])

# Convolución manual (sin padding)
output = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        region = imagen[i:i+3, j:j+3]
        output[i, j] = np.sum(region * filtro)

print(output)
# [[5. 0. 5.]
#  [0. 5. 0.]
#  [5. 0. 5.]]
```

Donde el filtro coincide exactamente con el patrón de la imagen, el valor es alto (5). Donde no coincide, es bajo (0). **El filtro detecta su propio patrón.**

### Detección de patrones: de lo simple a lo complejo

Acá viene la magia de las CNN. Cada capa de convolución detecta patrones de diferente complejidad:

- **Capa 1 (primeras capas):** Detecta **bordes** — líneas horizontales, verticales, diagonales.
- **Capa 2:** Combina bordes en **texturas y formas simples** — esquinas, curvas, gradientes.
- **Capa 3-4:** Detecta **partes de objetos** — ojos, ruedas, patas.
- **Capas profundas:** Reconoce **objetos completos** — caras, autos, gatos.

**Analogía:** Es como leer. Primero aprendés las letras (bordes), después las sílabas (texturas), después las palabras (partes), y finalmente entendés oraciones (objetos). Cada nivel se construye sobre el anterior.

### Parámetros de la convolución

#### Stride (paso)

Cuántos píxeles se mueve el filtro en cada paso. Con stride=1, se mueve de a 1. Con stride=2, salta de a 2, produciendo un output más chico.

#### Padding (relleno)

Se agregan ceros alrededor de la imagen para controlar el tamaño del output. Con `padding='same'`, el output tiene el mismo tamaño que el input.

#### Cantidad de filtros

Cada filtro detecta un patrón diferente. Si usás 32 filtros en una capa, obtenés 32 feature maps. La red **aprende automáticamente** qué filtros son útiles durante el entrenamiento.

### En PyTorch

```python
import torch
import torch.nn as nn

# Capa convolucional: 3 canales de entrada (RGB), 16 filtros de 3x3
conv = nn.Conv2d(
    in_channels=3,     # RGB
    out_channels=16,   # 16 filtros
    kernel_size=3,     # filtro 3x3
    stride=1,          # paso de 1
    padding=1           # same padding
)

# Input: batch de 4 imágenes RGB de 32x32
x = torch.randn(4, 3, 32, 32)  # (batch, canales, alto, ancho)
output = conv(x)
print(output.shape)  # torch.Size([4, 16, 32, 32])
# 4 imágenes, 16 feature maps, 32x32 cada uno
```

### ¿Por qué son eficientes?

Acá está el truco genial de las CNN:

1. **Compartición de pesos:** El mismo filtro 3×3 se aplica en **toda** la imagen. En vez de 150 millones de pesos, un filtro 3×3 tiene solo 27 pesos (3×3×3 para RGB). 32 filtros = ~900 pesos. Comparalo con los 150 millones del MLP.

2. **Conexiones locales:** Cada neurona solo mira una región pequeña de la imagen (el tamaño del filtro), no todos los píxeles.

3. **Invarianza de traslación:** Como el mismo filtro recorre toda la imagen, puede detectar un gato esté donde esté. No importa si está centrado o en una esquina.

---

## 3. Pooling: reduciendo la dimensión

Después de la convolución, los feature maps todavía son grandes. **Pooling** reduce su tamaño manteniendo la información más importante.

### Max Pooling

La operación más común. Divide el feature map en regiones (típicamente 2×2) y se queda con el **valor máximo** de cada región.

```
Input (4×4):                  Max Pooling 2×2:
┌─────────────┐               ┌───────┐
│ 1  3 │ 2  1 │               │ 3   2 │
│ 0  2 │ 1  0 │      →        │ 6   8 │
├───────┼──────┤               └───────┘
│ 6  4 │ 3  8 │
│ 1  2 │ 5  2 │
└─────────────┘
```

De cada cuadrado 2×2, nos quedamos con el más grande: `max(1,3,0,2)=3`, `max(2,1,1,0)=2`, `max(6,4,1,2)=6`, `max(3,8,5,2)=8`.

**Resultado:** El feature map se reduce a la mitad en cada dimensión (4×4 → 2×2).

### ¿Por qué funciona?

- **Reduce parámetros:** Menos datos = menos cómputo en las siguientes capas.
- **Invarianza local:** Si un borde se mueve 1 píxel, el máximo de la región probablemente sigue siendo el mismo. Esto hace al modelo más robusto a pequeñas variaciones.
- **Evita overfitting:** Menos parámetros = menos riesgo de memorizar.

### Average Pooling

En vez del máximo, toma el **promedio**. Se usa menos en capas intermedias, pero es común como última operación antes de la clasificación (**Global Average Pooling**).

### En PyTorch

```python
pool = nn.MaxPool2d(kernel_size=2, stride=2)

x = torch.randn(4, 16, 32, 32)   # 4 imágenes, 16 canales, 32x32
output = pool(x)
print(output.shape)  # torch.Size([4, 16, 16, 16])
# Se redujo a la mitad: 32x32 → 16x16
```

---

## 4. Arquitectura típica de una CNN

Una CNN típica sigue este patrón:

```
[Input] → [Conv + ReLU + Pool] × N → [Flatten] → [Dense + ReLU] → [Dense + Softmax] → [Output]
```

O más detallado:

```
Imagen (224×224×3)
    ↓
Conv2D (32 filtros, 3×3) + ReLU → 224×224×32
MaxPool (2×2)                   → 112×112×32
    ↓
Conv2D (64 filtros, 3×3) + ReLU → 112×112×64
MaxPool (2×2)                   → 56×56×64
    ↓
Conv2D (128 filtros, 3×3) + ReLU → 56×56×128
MaxPool (2×2)                    → 28×28×128
    ↓
Flatten                          → 100,352
Dense (256) + ReLU               → 256
Dense (10) + Softmax             → 10 clases
```

Observá el patrón: a medida que avanzás en la red, la resolución espacial **baja** (224 → 112 → 56 → 28) pero la cantidad de canales **sube** (3 → 32 → 64 → 128). La red comprime la información espacial y expande la información semántica.

### Implementación completa en PyTorch

```python
import torch
import torch.nn as nn

class CNNClasificador(nn.Module):
    def __init__(self, num_clases=10):
        super().__init__()

        # Bloque convolucional
        self.features = nn.Sequential(
            # Bloque 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),         # 224→112

            # Bloque 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),         # 112→56

            # Bloque 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),         # 56→28
        )

        # Clasificador (capas dense)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),            # regularización
            nn.Linear(256, num_clases)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CNNClasificador(num_clases=10)

# Probamos con una imagen dummy
img = torch.randn(1, 3, 224, 224)
output = model(img)
print(output.shape)  # torch.Size([1, 10])
```

### Batch Normalization

Un componente que se agrega frecuentemente:

```python
nn.Conv2d(32, 64, kernel_size=3, padding=1),
nn.BatchNorm2d(64),   # normaliza las activaciones
nn.ReLU(),
```

Normaliza las activaciones en cada capa para que tengan media ~0 y varianza ~1. Esto **estabiliza y acelera** el entrenamiento significativamente.

---

## 5. Arquitecturas famosas

A lo largo de los años, investigadores propusieron arquitecturas cada vez más sofisticadas. Conocer las principales te da perspectiva histórica y práctica.

### LeNet-5 (LeCun, 1998) — La pionera

La primera CNN exitosa, diseñada para reconocer dígitos escritos a mano (dataset MNIST).

```
Input (32×32×1) → Conv(6) → Pool → Conv(16) → Pool → FC(120) → FC(84) → Output(10)
```

- Solo 60,000 parámetros.
- Demostró que las CNN funcionan para visión.
- Yann LeCun la usó en producción para leer cheques bancarios.

### AlexNet (Krizhevsky, 2012) — La revolución

Ganó el desafío ImageNet por un margen enorme y **marcó el inicio de la era del deep learning**.

- 8 capas (5 conv + 3 dense), 60 millones de parámetros.
- Primera red grande entrenada en GPUs (dos NVIDIA GTX 580).
- Introdujo ReLU, Dropout, y data augmentation como prácticas estándar.
- Redujo el error top-5 en ImageNet de 26% a 16%.

### VGGNet (Simonyan & Zisserman, 2014) — La simple y profunda

Filosofía: **solo filtros 3×3, apilar muchas capas**.

- VGG-16: 16 capas, 138 millones de parámetros.
- Demostró que la profundidad importa mucho.
- Muy usada como feature extractor en transfer learning.
- Problema: muchos parámetros, lenta de entrenar.

### ResNet (He et al., 2015) — Las conexiones residuales

El gran salto. Hasta ResNet, redes con más de ~20 capas dejaban de mejorar (o empeoraban). ResNet introdujo las **skip connections** (conexiones residuales):

```
x → [Conv → BN → ReLU → Conv → BN] → (+x) → ReLU
     └──────── residual block ────────┘
```

La idea: en vez de aprender la transformación directa `F(x)`, la red aprende la **diferencia** `F(x) = H(x) - x`. El shortcut `+x` permite que el gradiente fluya directamente hacia atrás sin degradarse.

- ResNet-50: 50 capas, 25 millones de parámetros.
- ResNet-152: 152 capas.
- Ganó ImageNet 2015 con 3.57% de error (¡mejor que humanos!).

**Analogía:** Es como tomar apuntes anotando solo lo que cambió respecto de la clase anterior, en vez de reescribir todo. Más eficiente y no perdés información.

### EfficientNet (Tan & Le, 2019) — La eficiente

Hasta EfficientNet, la gente escalaba redes de tres formas: más profundas, más anchas, o mayor resolución de entrada. EfficientNet demostró que hay que escalar las tres dimensiones de forma **balanceada** usando un coeficiente compuesto.

- EfficientNet-B0: 5.3 millones de parámetros, mejor que ResNet-50.
- EfficientNet-B7: Logra la mejor accuracy con menos parámetros que las alternativas.
- Usa **depthwise separable convolutions** para ser más eficiente.

### Línea temporal resumida

```
1998: LeNet-5        →  60K params  →  Dígitos escritos a mano
2012: AlexNet        →  60M params  →  Inicio era deep learning
2014: VGGNet         → 138M params  →  Profundidad importa
2015: ResNet         →  25M params  →  Skip connections, +152 capas
2019: EfficientNet   →   5M params  →  Escalar inteligentemente
```

Fijate la tendencia: los modelos más nuevos logran **mejor performance con menos parámetros**. No se trata de hacer redes más grandes, sino más inteligentes.

---

## 6. CNN en acción: clasificar CIFAR-10

Para cerrar, un ejemplo completo de entrenamiento:

```python
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

# Datos
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True
)

# Modelo simple
model = nn.Sequential(
    nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(256, 10)
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Entrenamiento (simplificado)
for epoch in range(10):
    running_loss = 0.0
    for images, labels in trainloader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(trainloader):.4f}")
```

---

## 📝 Conceptos clave para recordar

- **MLP para imágenes = mala idea.** Demasiados parámetros, pierde estructura espacial, no tiene invarianza de posición.
- **Convolución:** Un filtro pequeño (3×3) se desliza sobre la imagen, detectando patrones locales. Producto punto en cada posición.
- **Filtros/Kernels:** Matrices pequeñas que aprenden a detectar bordes, texturas, partes de objetos. La red los aprende sola durante el entrenamiento.
- **Compartición de pesos:** El mismo filtro se aplica en toda la imagen → pocos parámetros, invarianza de traslación.
- **Jerarquía de features:** Capas tempranas → bordes. Capas medias → texturas y formas. Capas profundas → objetos completos.
- **Pooling (Max Pooling):** Reduce la resolución espacial quedándose con el máximo de cada región. Baja parámetros y agrega robustez.
- **Arquitectura típica:** `[Conv + ReLU + Pool] × N → Flatten → Dense → Softmax`
- **El patrón:** Resolución espacial baja, cantidad de canales sube.
- **LeNet (1998):** La primera CNN exitosa.
- **AlexNet (2012):** Inicio del deep learning moderno.
- **ResNet (2015):** Skip connections para entrenar redes de +100 capas.
- **EfficientNet (2019):** Escalar profundidad, ancho y resolución de forma balanceada.
- **Batch Normalization:** Normaliza activaciones para estabilizar el entrenamiento.
