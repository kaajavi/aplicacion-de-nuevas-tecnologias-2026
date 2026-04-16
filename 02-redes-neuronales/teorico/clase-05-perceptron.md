# Clase 5 — El Perceptrón y Redes Neuronales Básicas

## Objetivos
- Entender qué es un perceptrón y cómo funciona
- Conocer las funciones de activación
- Entender el concepto de capas y forward pass

## Contenido

### 1. El Perceptrón
- Inspirado en la neurona biológica (simplificado)
- Inputs × pesos + bias → función de activación → output
- Un perceptrón = un clasificador lineal
- Limitación histórica: no puede resolver XOR (Minsky & Papert, 1969)

### 2. Multi-Layer Perceptron (MLP)
- Agregar capas ocultas resuelve el problema de XOR
- Capa de entrada → capas ocultas → capa de salida
- Cada neurona está conectada con todas las de la capa siguiente (fully connected)

### 3. Funciones de Activación
- **ReLU:** max(0, x) — la más usada en capas ocultas, simple y eficiente
- **Sigmoid:** 1/(1+e^(-x)) — mapea a [0,1], usada en clasificación binaria
- **Softmax:** convierte un vector de números en probabilidades que suman 1
- **Tanh:** mapea a [-1,1] — menos usada hoy

### 4. Forward Pass
- Los datos entran por la izquierda, se transforman capa por capa, sale una predicción
- Es solo multiplicación de matrices + funciones de activación
- En este punto el modelo no "aprende" nada, solo calcula

## Recursos
- [3Blue1Brown: Neural Networks (YouTube)](https://www.youtube.com/watch?v=aircAruvnKk)
- [Playground de TensorFlow](https://playground.tensorflow.org/)
