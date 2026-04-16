# Clase 7 — Redes Neuronales Convolucionales (CNNs)

## Objetivos
- Entender por qué las CNNs son buenas para imágenes
- Conocer convolución, pooling y arquitecturas famosas

## Contenido

### 1. ¿Por qué no usar MLPs para imágenes?
- Una imagen 224×224 RGB = 150,528 inputs → demasiados parámetros
- Un MLP no entiende que los píxeles vecinos están relacionados
- Las CNNs explotan la estructura espacial de las imágenes

### 2. La Operación de Convolución
- Un filtro/kernel (ej: 3×3) se desliza por la imagen
- Detecta patrones locales: bordes, texturas, formas
- Primeras capas: detectan bordes simples
- Capas profundas: detectan patrones complejos (ojos, ruedas, letras)

### 3. Pooling
- Reduce el tamaño de la representación (downsampling)
- **Max Pooling:** toma el valor máximo en una ventana
- Hace al modelo más robusto a pequeños desplazamientos

### 4. Arquitectura típica de una CNN
- Conv → ReLU → Conv → ReLU → Pool → ... → Flatten → Dense → Output
- Cada bloque Conv+Pool extrae features más abstractas

### 5. Arquitecturas famosas
- **LeNet (1998):** dígitos escritos a mano
- **AlexNet (2012):** el boom del deep learning (ganó ImageNet)
- **ResNet (2015):** conexiones residuales, hasta 152 capas
- **EfficientNet (2019):** optimización de escala

## Recursos
- [CNN Explainer (visual interactivo)](https://poloclub.github.io/cnn-explainer/)
- [CS231n Stanford: CNNs](https://cs231n.github.io/convolutional-networks/)
