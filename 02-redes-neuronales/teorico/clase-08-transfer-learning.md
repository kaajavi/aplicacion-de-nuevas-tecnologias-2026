# Clase 8 — Transfer Learning y Fine-Tuning

## Objetivos
- Entender por qué no siempre conviene entrenar desde cero
- Conocer transfer learning y fine-tuning
- Saber cuándo aplicar cada estrategia

## Contenido

### 1. El Problema
- Entrenar una CNN desde cero necesita millones de imágenes y días de GPU
- La mayoría de nosotros no tiene ni los datos ni el cómputo
- Solución: usar modelos que alguien ya entrenó

### 2. Transfer Learning
- Tomar un modelo pre-entrenado (ej: ResNet entrenado en ImageNet, 1M imágenes)
- Las primeras capas ya saben detectar bordes, texturas, formas genéricas
- Reemplazar solo la última capa (clasificador) para nuestra tarea
- Congelar las capas pre-entrenadas, entrenar solo la nueva

### 3. Fine-Tuning
- Similar a transfer learning, pero descongelamos algunas capas
- Permite que el modelo se adapte mejor a nuestros datos específicos
- Se usa un learning rate más bajo para no destruir lo aprendido
- Estrategia común: congelar todo → entrenar última capa → descongelar últimas capas → re-entrenar

### 4. ¿Cuándo usar cada uno?
- **Datos similares al pre-entrenamiento + pocos datos:** Transfer learning (solo cambiar clasificador)
- **Datos diferentes + suficientes datos:** Fine-tuning (descongelar capas)
- **Muchos datos + dominio muy diferente:** Quizás entrenar desde cero
- **En la práctica:** casi siempre conviene empezar con transfer learning

### 5. Hugging Face Model Hub
- Repositorio de modelos pre-entrenados para todo: imágenes, texto, audio
- Cómo buscar y descargar modelos
- La democratización del ML

## Recursos
- [Hugging Face Model Hub](https://huggingface.co/models)
- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
