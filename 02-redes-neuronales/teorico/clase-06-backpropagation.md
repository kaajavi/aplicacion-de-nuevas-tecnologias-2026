# Clase 6 — Backpropagation y Gradient Descent

## Objetivos
- Entender cómo una red neuronal "aprende"
- Conocer backpropagation y gradient descent
- Entender el rol del learning rate

## Contenido

### 1. La Función de Pérdida (Loss)
- Mide qué tan lejos está la predicción del valor real
- **MSE** para regresión, **Cross-Entropy** para clasificación
- El objetivo del entrenamiento: minimizar esta función

### 2. Gradient Descent
- Imaginar la loss como un paisaje montañoso — queremos llegar al valle
- El gradiente indica la dirección de mayor subida → vamos en dirección contraria
- **Paso a paso:** calcular loss → calcular gradientes → actualizar pesos → repetir

### 3. Learning Rate
- Qué tan grande es cada paso
- **Muy alto:** salta por todos lados, no converge
- **Muy bajo:** tarda una eternidad, se queda en mínimos locales
- En la práctica: empezar con 0.001 y ajustar (learning rate schedulers)

### 4. Backpropagation
- Algoritmo para calcular los gradientes eficientemente
- Usa la regla de la cadena del cálculo (chain rule)
- Propaga el error desde la salida hacia atrás, capa por capa
- Sin backprop, entrenar redes profundas sería computacionalmente imposible

### 5. Epochs, Batches, Iteraciones
- **Epoch:** una pasada completa por todo el dataset
- **Batch:** subconjunto de datos procesados juntos
- **Mini-batch gradient descent:** el estándar moderno
- Batch size típico: 32, 64, 128

## Recursos
- [3Blue1Brown: Gradient Descent (YouTube)](https://www.youtube.com/watch?v=IHZwWFHWa-w)
- [3Blue1Brown: Backpropagation (YouTube)](https://www.youtube.com/watch?v=Ilg3gGewQ5U)
