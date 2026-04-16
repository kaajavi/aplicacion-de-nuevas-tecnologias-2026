# Clase 8 — Transfer Learning y Fine-Tuning

## Unidad 2: Redes Neuronales | Aplicación de Nuevas Tecnologías — ITS Villada

---

## 1. El problema: entrenar desde cero es carísimo

En la clase anterior vimos arquitecturas como ResNet y EfficientNet. Estos modelos se entrenaron en **ImageNet**, un dataset con:

- **14 millones de imágenes**
- **1,000 categorías** (desde "golden retriever" hasta "espagueti")
- Semanas de entrenamiento en **múltiples GPUs de alta gama**

Ahora supongamos que vos querés hacer algo más modesto: clasificar fotos de perros y gatos. Tenés 2,000 imágenes. ¿Qué hacés?

### Opción A: Entrenar desde cero

- Diseñás tu propia arquitectura CNN.
- Inicializás todos los pesos aleatoriamente.
- Entrenás con tus 2,000 imágenes.

**Resultado probable:** El modelo no aprende bien. ¿Por qué?

1. **Datos insuficientes:** 2,000 imágenes son muy pocas para que una CNN aprenda desde cero a detectar bordes, texturas, formas y objetos. Necesitarías al menos cientos de miles.
2. **Overfitting garantizado:** Con pocos datos y muchos parámetros, el modelo memoriza las imágenes de entrenamiento en vez de aprender patrones generalizables.
3. **Recursos:** Incluso una red chica tarda horas o días en entrenar. Una red grande como ResNet necesitaría GPUs que probablemente no tenés.

### Opción B: Transfer Learning 🏆

¿Y si en vez de empezar de cero, arrancás con un modelo que **ya sabe ver**?

Un ResNet entrenado en ImageNet ya aprendió a detectar bordes, texturas, formas, partes de objetos... todo eso a partir de 14 millones de imágenes. Esos conocimientos son **transferibles** a tu tarea, incluso si tu dataset es completamente distinto.

**Analogía:** Imaginá que querés aprender a tocar la guitarra. Empezar "desde cero" sería como nunca haber escuchado música ni haber usado las manos. Pero vos ya sabés coordinación motriz, entendés ritmo, y probablemente tocaste algún otro instrumento. Todo eso se **transfiere**. No empezás de cero, empezás con ventaja.

---

## 2. Transfer Learning: reusar lo aprendido

Transfer Learning es la técnica de tomar un modelo **pre-entrenado** en una tarea grande y adaptarlo a tu tarea específica. Es la forma más común de trabajar con deep learning en la práctica.

### ¿Cómo funciona?

Recordá la estructura de una CNN:

```
[Capas convolucionales]  →  [Capas dense]  →  [Capa de salida]
   "Feature extractor"       "Clasificador"     "Predicción"
   (bordes, texturas,        (combina features   (1000 clases de
    formas, partes)           para decidir)       ImageNet)
```

La clave es que las **capas convolucionales** (el feature extractor) aprenden representaciones **genéricas** que sirven para casi cualquier tarea de visión:

- Capas tempranas → bordes y texturas (útiles para TODO).
- Capas medias → formas y patrones (útiles para casi todo).
- Capas finales → partes específicas de objetos (más específicas al dataset original).

### El procedimiento estándar

1. **Tomá un modelo pre-entrenado** (ej: ResNet-50 entrenado en ImageNet).
2. **Eliminá la última capa** (la que clasifica en 1,000 categorías de ImageNet).
3. **Agregá tu propia capa de salida** (ej: 2 neuronas para perro/gato).
4. **Congelá las capas convolucionales** (que no se modifiquen durante el entrenamiento).
5. **Entrenás solo la capa nueva** con tus datos.

### Implementación en PyTorch

```python
import torch
import torch.nn as nn
import torchvision.models as models

# 1. Cargar ResNet-50 pre-entrenada en ImageNet
model = models.resnet50(weights='IMAGENET1K_V2')

# 2. Congelar TODAS las capas (no se van a actualizar)
for param in model.parameters():
    param.requires_grad = False

# 3. Reemplazar la última capa (originalmente 1000 clases)
# ResNet-50 tiene model.fc como última capa
num_features = model.fc.in_features  # 2048
model.fc = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 2)  # 2 clases: perro y gato
)

# Solo los parámetros de model.fc tienen requires_grad=True
# Verificamos:
params_entrenables = sum(p.numel() for p in model.parameters() if p.requires_grad)
params_totales = sum(p.numel() for p in model.parameters())
print(f"Parámetros entrenables: {params_entrenables:,}")  # ~525,000
print(f"Parámetros totales: {params_totales:,}")           # ~23,500,000
# ¡Solo entrenamos el 2% de los parámetros!
```

### Entrenamiento con Transfer Learning

```python
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# Preprocesamiento (mismo que usó ResNet en ImageNet)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Datos: estructura de carpetas
# data/train/perro/  (imágenes de perros)
# data/train/gato/   (imágenes de gatos)
# data/val/perro/
# data/val/gato/
train_dataset = ImageFolder('data/train', transform=transform)
val_dataset = ImageFolder('data/val', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# Entrenamiento
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)  # ¡solo model.fc!

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(10):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    print(f"Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, Acc={acc:.1f}%")
```

Con solo 2,000 imágenes y 10 epochs, podés llegar a **95%+ de accuracy**. Entrenar desde cero con la misma cantidad de datos probablemente te daría 60-70%.

---

## 3. Fine-Tuning: ajuste fino

Transfer Learning básico congela todo el feature extractor y solo entrena la capa final. **Fine-tuning** va un paso más allá: **descongelás algunas capas** del modelo pre-entrenado y las re-entrenás con tu dataset, usando un **learning rate muy bajo**.

### ¿Por qué fine-tuning?

- Las capas congeladas aprendieron features genéricas (bordes, texturas) → perfectas para cualquier tarea.
- Pero las capas más profundas aprendieron features **específicas de ImageNet** → podrían mejorar si se adaptan a tu dataset.
- Fine-tuning permite que esas capas profundas se **especialicen** en tu tarea.

### El procedimiento

1. **Primero:** Entrenás solo la capa nueva (Transfer Learning estándar) durante algunos epochs. Esto permite que la capa nueva se estabilice.
2. **Después:** Descongelás las últimas capas del feature extractor.
3. **Entrenás todo junto** con un learning rate **mucho más bajo** (10x-100x menor).

### ¿Por qué learning rate bajo?

Porque los pesos pre-entrenados ya están cerca de una buena solución. Si usás un learning rate alto, vas a **destruir** lo que el modelo aprendió en ImageNet. Un lr bajo permite ajustes finos sin romper nada.

**Analogía:** Es como calibrar un instrumento de precisión. Ya está casi perfecto, solo necesitás girar las perillas un poquito. Si las movés bruscamente, lo descalibrás.

### Implementación en PyTorch

```python
# Paso 1: Ya entrenaste la capa final (código anterior)
# Ahora vamos a descongelar las últimas capas de ResNet

# Descongelar layer4 (la última capa convolucional de ResNet)
for param in model.layer4.parameters():
    param.requires_grad = True

# También podés descongelar layer3 si querés
# for param in model.layer3.parameters():
#     param.requires_grad = True

# Optimizer con learning rates diferentes
optimizer = torch.optim.Adam([
    # Capas descongeladas: learning rate bajo
    {'params': model.layer4.parameters(), 'lr': 1e-5},
    # Capa nueva: learning rate normal
    {'params': model.fc.parameters(), 'lr': 1e-3},
])

# Entrenar más epochs con fine-tuning
for epoch in range(10):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    acc = 100. * correct / total
    print(f"Fine-tune Epoch {epoch+1}: Loss={running_loss/len(train_loader):.4f}, Acc={acc:.1f}%")
```

### Learning rates diferenciados

Fijate que en el optimizer usamos **learning rates diferentes** para diferentes partes del modelo:

- **Capas pre-entrenadas descongeladas:** `lr = 1e-5` (muy bajo, para no destruir lo aprendido).
- **Capa nueva:** `lr = 1e-3` (normal, porque estos pesos son nuevos y necesitan moverse más).

Esto se llama **discriminative learning rates** o **differential learning rates**, y es una práctica muy común.

---

## 4. ¿Cuándo usar cada estrategia?

No siempre conviene lo mismo. Depende de dos factores principales: **cuántos datos tenés** y **qué tan similar es tu tarea a la tarea original** (ImageNet).

### Escenario 1: Pocos datos + tarea similar

**Ejemplo:** Clasificar razas de perros (ImageNet ya tiene muchas razas).

**Estrategia:** Transfer Learning (congelar todo, entrenar solo la capa final).

¿Por qué? Tus datos son pocos y la tarea es parecida. Las features de ImageNet ya son perfectas. Solo necesitás una nueva cabeza clasificadora.

### Escenario 2: Muchos datos + tarea similar

**Ejemplo:** Clasificar 500 tipos de productos en un e-commerce (muchos datos de imágenes de productos).

**Estrategia:** Fine-tuning (descongelar las últimas capas).

¿Por qué? Tenés suficientes datos para ajustar las capas profundas sin riesgo de overfitting, y las features genéricas de ImageNet pueden mejorar si se adaptan a tu dominio.

### Escenario 3: Pocos datos + tarea diferente

**Ejemplo:** Clasificar imágenes médicas de rayos X (muy distinto a fotos cotidianas).

**Estrategia:** Transfer Learning con features de capas intermedias.

¿Por qué? Las primeras capas (bordes, texturas) siguen siendo útiles, pero las capas profundas aprendieron cosas muy específicas de ImageNet. Podrías usar features de una capa intermedia como input para un clasificador nuevo.

### Escenario 4: Muchos datos + tarea diferente

**Ejemplo:** Clasificar imágenes satelitales con millones de ejemplos.

**Estrategia:** Fine-tuning agresivo (descongelar muchas o todas las capas) o incluso entrenar desde cero.

¿Por qué? Tenés suficientes datos para entrenar, y la tarea es tan distinta que las features de ImageNet podrían no ser ideales.

### Tabla resumen

```
                    │  Tarea similar      │  Tarea diferente
────────────────────┼────────────────────┼─────────────────────
 Pocos datos        │  TL (congelar)     │  TL (features medias)
                    │  Solo capa final    │  Cuidado con overfit
────────────────────┼────────────────────┼─────────────────────
 Muchos datos       │  Fine-tuning       │  Fine-tuning agresivo
                    │  Últimas capas      │  o desde cero
```

---

## 5. Más allá de la visión: Transfer Learning en NLP y más

Transfer Learning no es solo para imágenes. Es la base de la IA moderna:

### NLP (Procesamiento de Lenguaje Natural)

- **BERT** (Google, 2018): Pre-entrenado en texto masivo. Se hace fine-tuning para clasificación de texto, preguntas y respuestas, análisis de sentimiento, etc.
- **GPT** (OpenAI): Pre-entrenado para predecir la siguiente palabra. Fine-tuned para seguir instrucciones (ChatGPT).

El concepto es el mismo: un modelo grande aprende representaciones generales del lenguaje, y después lo adaptás a tu tarea.

### Audio

- **Whisper** (OpenAI): Pre-entrenado en 680,000 horas de audio. Se puede hacer fine-tuning para tu idioma o dominio específico.

### El patrón universal

```
1. Alguien con muchos recursos (Google, OpenAI, Meta) entrena un modelo enorme
   en datos masivos → "foundation model"
2. Vos descargás ese modelo
3. Lo adaptás a tu tarea con tus datos (pocos)
4. Profit
```

Este paradigma se llama **"foundation models"** y es la forma dominante de hacer IA hoy en día.

---

## 6. Hugging Face Model Hub: el supermercado de modelos

**Hugging Face** es la plataforma más importante para compartir y usar modelos pre-entrenados. Es como GitHub pero para modelos de IA.

### ¿Qué encontrás?

- **+500,000 modelos** pre-entrenados.
- Para todo: visión, texto, audio, multimodal.
- Modelos de Google, Meta, OpenAI, Microsoft, y la comunidad.
- Documentación, métricas, y ejemplos de uso.

### ¿Cómo se usa?

```bash
pip install transformers timm
```

#### Para visión (con `timm`)

```python
import timm
import torch

# Cargar un EfficientNet-B0 pre-entrenado
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=2)

# ¡Listo! Ya podés entrenar con tus datos
img = torch.randn(1, 3, 224, 224)
output = model(img)
print(output.shape)  # torch.Size([1, 2])
```

#### Para texto (con `transformers`)

```python
from transformers import pipeline

# Análisis de sentimiento con un modelo pre-entrenado
clasificador = pipeline("sentiment-analysis")
resultado = clasificador("Me encanta la programación")
print(resultado)
# [{'label': 'POSITIVE', 'score': 0.9998}]
```

#### Fine-tuning con Hugging Face

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import Trainer, TrainingArguments

# Cargar modelo pre-entrenado
model_name = "bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Configurar entrenamiento
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=2e-5,          # lr bajo para fine-tuning
    weight_decay=0.01,
    evaluation_strategy="epoch",
)

# Trainer se encarga del loop de entrenamiento
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

trainer.train()
```

### Explorando el Hub

Entrá a [huggingface.co/models](https://huggingface.co/models) y explorá. Podés filtrar por:

- **Tarea:** Image Classification, Text Generation, Object Detection, etc.
- **Librería:** PyTorch, TensorFlow, JAX
- **Dataset:** ImageNet, COCO, etc.
- **Idioma:** Para modelos de NLP

Cada modelo tiene una **Model Card** con:
- Descripción del modelo
- Cómo usarlo (código listo para copiar y pegar)
- Métricas de performance
- Limitaciones y sesgos
- Licencia

### Modelos populares para Transfer Learning en visión

| Modelo | Params | Top-1 Acc (ImageNet) | Velocidad | Uso recomendado |
|--------|--------|---------------------|-----------|-----------------|
| ResNet-50 | 25M | 80.4% | Rápido | Buen balance general |
| EfficientNet-B0 | 5.3M | 77.1% | Muy rápido | Recursos limitados |
| EfficientNet-B4 | 19M | 82.9% | Medio | Alta accuracy |
| ViT-Base | 86M | 81.8% | Medio | Datasets grandes |
| ConvNeXt-Base | 89M | 83.8% | Medio | Estado del arte CNN |

---

## 7. Workflow práctico completo

Para cerrar, así es como se ve un proyecto real de clasificación de imágenes hoy:

```python
import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms

# 1. PREPROCESAMIENTO
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_data = ImageFolder('data/train', transform=train_transform)
val_data = ImageFolder('data/val', transform=val_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

# 2. MODELO (Transfer Learning)
model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=len(train_data.classes))

# 3. Congelar backbone
for name, param in model.named_parameters():
    if 'classifier' not in name:
        param.requires_grad = False

# 4. ENTRENAMIENTO — Fase 1: Solo clasificador
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=1e-3
)

print("=== Fase 1: Transfer Learning ===")
for epoch in range(5):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        loss = criterion(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 5. ENTRENAMIENTO — Fase 2: Fine-tuning
print("=== Fase 2: Fine-tuning ===")
for param in model.parameters():
    param.requires_grad = True

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

for epoch in range(10):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        loss = criterion(model(images), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

# 6. GUARDAR
torch.save(model.state_dict(), 'mi_modelo.pth')
print("Modelo guardado!")
```

### Pasos clave del workflow

1. **Elegir modelo pre-entrenado** (EfficientNet, ResNet, etc. desde timm o Hugging Face).
2. **Preparar datos** con las mismas transformaciones que usó el modelo original.
3. **Fase 1 — Transfer Learning:** Congelar backbone, entrenar solo clasificador (lr alto, pocos epochs).
4. **Fase 2 — Fine-tuning:** Descongelar todo, lr muy bajo, más epochs.
5. **Evaluar** en datos de validación.
6. **Guardar** el modelo para producción.

---

## 📝 Conceptos clave para recordar

- **Entrenar desde cero** requiere millones de datos y muchas GPUs. Para la mayoría de los proyectos, es innecesario y contraproducente.
- **Transfer Learning:** Tomás un modelo pre-entrenado (ej: ResNet en ImageNet), congelás sus pesos, reemplazás la última capa, y entrenás solo esa capa nueva con tus datos.
- **Fine-tuning:** Después de Transfer Learning, descongelás algunas capas del modelo pre-entrenado y las re-entrenás con un learning rate **muy bajo** (10x-100x menor).
- **Learning rate bajo en fine-tuning** es crucial. Un lr alto destruye lo que el modelo aprendió.
- **Discriminative learning rates:** Diferentes partes del modelo usan diferentes learning rates. Capas pre-entrenadas → lr bajo. Capas nuevas → lr normal.
- **Cuándo cada estrategia:**
  - Pocos datos + tarea similar → Transfer Learning (congelar).
  - Muchos datos + tarea similar → Fine-tuning (descongelar últimas capas).
  - Pocos datos + tarea diferente → Transfer Learning con features intermedias.
  - Muchos datos + tarea diferente → Fine-tuning agresivo o desde cero.
- **Hugging Face Model Hub:** Plataforma con +500K modelos pre-entrenados. Usá `timm` para visión y `transformers` para NLP.
- **Foundation models:** El paradigma actual. Alguien entrena un modelo enorme, vos lo adaptás a tu tarea.
- **Workflow práctico:** Elegir modelo → preparar datos → Transfer Learning → Fine-tuning → evaluar → guardar.
