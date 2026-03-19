# Clase 1: Introducción a la Inteligencia Artificial y Machine Learning

## ¿Qué es la Inteligencia Artificial?

Empecemos por lo básico: ¿qué es realmente la IA? Si le preguntás a diez personas, probablemente te den diez respuestas distintas. Y eso ya te dice algo sobre el estado del tema.

**Definición clásica (académica):** La Inteligencia Artificial es el campo de la informática que estudia cómo crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana: reconocer imágenes, entender lenguaje natural, tomar decisiones, aprender de la experiencia.

**Definición más práctica:** IA es un conjunto de técnicas y algoritmos que permiten a las computadoras resolver problemas complejos sin que un programador escriba explícitamente cada regla.

### La IA como campo científico vs. la IA como marketing

Acá hay una distinción **fundamental** que tenés que entender para no caer en la trampa del hype.

**IA como campo científico** existe desde los años 50. Es una disciplina seria, con décadas de investigación en universidades y laboratorios. Incluye subcampos como visión por computadora, procesamiento de lenguaje natural (NLP), robótica, planificación automática, y por supuesto, Machine Learning.

**IA como término de marketing** es lo que ves en las publicidades: "¡Nuestra heladera tiene IA!", "¡Nuestro cepillo de dientes usa IA!". En muchos casos, lo que hay detrás es un `if/else` glorificado o un algoritmo estadístico simple que existe hace 30 años.

**¿Cómo distinguirlos?** Preguntate: ¿el sistema aprende de datos y mejora con el tiempo, o simplemente sigue reglas predefinidas? Si es lo segundo, probablemente no sea IA en el sentido técnico, aunque el departamento de marketing diga que sí.

> **Analogía:** Es como cuando los supermercados ponen "artesanal" en cualquier producto. El pan puede ser industrial y tener la etiqueta "artesanal". Con la IA pasa lo mismo: el término se usa muy libremente.

---

## El Test de Turing

En 1950, Alan Turing publicó un paper legendario: *"Computing Machinery and Intelligence"*. En vez de preguntarse "¿pueden pensar las máquinas?" (una pregunta filosófica sin fin), propuso algo más práctico:

**El Test de Turing funciona así:**
1. Un humano (el juez) conversa por texto con dos entidades: una es humana, la otra es una máquina.
2. El juez no sabe cuál es cuál.
3. Si después de un rato el juez no puede distinguir de forma consistente quién es quién, se dice que la máquina "pasó" el test.

### ¿Alguien pasó el Test de Turing?

Es debatible. En 2014, un chatbot llamado Eugene Goostman supuestamente lo pasó, pero el chatbot simulaba ser un nene de 13 años que no hablaba inglés nativo — o sea, las expectativas del juez eran bajas. La comunidad científica no lo tomó muy en serio.

Los LLMs modernos (GPT-4, Claude, Gemini) probablemente pasarían versiones del test en muchas conversaciones, pero eso no significa que "piensen". Pueden generar texto coherente sin entender nada de lo que dicen. Esto nos lleva al siguiente tema...

### Limitaciones del Test

- **No mide inteligencia real**, solo la capacidad de imitar comportamiento humano en texto.
- **Un sistema puede ser muy inteligente en un dominio específico** (ganar al ajedrez, diagnosticar enfermedades) sin poder mantener una conversación casual.
- **Es antropocéntrico**: asume que la inteligencia tiene que parecerse a la humana.

---

## IA Débil vs. IA Fuerte (AGI)

### IA Débil (Narrow AI)

Es toda la IA que existe hoy. Sistemas que son **muy buenos en UNA tarea específica** pero no pueden hacer nada fuera de eso.

Ejemplos:
- **AlphaGo** le ganó al campeón mundial de Go, pero no puede jugar al truco.
- **GPT-4** genera texto increíble, pero no puede hacer una cuenta matemática compleja sin equivocarse (a veces).
- **El algoritmo de recomendación de Spotify** te sugiere música genial, pero no sabe hacer otra cosa.

Cada uno de estos sistemas fue diseñado y entrenado para una tarea. No tienen "comprensión general" del mundo.

### IA Fuerte (AGI - Artificial General Intelligence)

La AGI sería una IA con capacidad de entender, aprender y aplicar conocimiento **en cualquier dominio**, como un humano. Podría aprender a programar, luego aprender carpintería, luego filosofía — todo con la misma "mente".

**¿Existe la AGI?** No. No todavía. Hay debate sobre si estamos cerca o lejos. Algunos investigadores dicen que faltan 5-10 años, otros dicen que faltan 50, otros creen que quizás nunca se logre.

> **Analogía:** La IA débil es como un empleado ultra-especialista: el mejor radiólogo del mundo, pero no sabe freír un huevo. La AGI sería como una persona que puede aprender cualquier oficio.

### ¿Y la superinteligencia?

Es el concepto teórico de una IA que supera la inteligencia humana en todos los aspectos. Es tema de ciencia ficción y de debates filosóficos serios (Nick Bostrom, Eliezer Yudkowsky), pero no es algo con lo que tengamos que preocuparnos hoy en lo técnico.

---

## Los Inviernos de la IA

La historia de la IA no fue un progreso lineal. Tuvo períodos de euforia y de decepción. A los períodos de decepción se los llama **"inviernos de la IA"**.

### Primer invierno (~1974-1980)

En los 60s hubo un optimismo desmedido. Los investigadores prometían que en 20 años tendríamos máquinas que traducirían idiomas perfectamente y resolverían cualquier problema. Spoiler: no pasó.

Cuando los resultados no llegaron, los gobiernos cortaron el financiamiento. El campo se estancó.

### Segundo invierno (~1987-1993)

En los 80s surgieron los "sistemas expertos" (programas basados en reglas que imitaban el conocimiento de expertos humanos). Hubo mucha inversión empresarial. Pero los sistemas eran frágiles, caros de mantener, y no escalaban bien. El entusiasmo se desplomó de nuevo.

### ¿Por qué importa esto?

Porque te enseña que **el hype no siempre refleja la realidad técnica**. Cuando alguien te dice "la IA va a reemplazar a todos los programadores en 2 años", recordá los inviernos. El progreso es real, pero las predicciones extremas casi siempre fallan.

### ¿Y ahora? ¿Estamos en un verano?

Sí, y un verano muy intenso. Desde ~2012 (cuando las redes neuronales profundas empezaron a ganar competencias de visión por computadora) y especialmente desde 2022 (ChatGPT), estamos en un boom enorme. ¿Va a haber otro invierno? Nadie sabe. Pero lo que sí es seguro es que muchas de las tecnologías actuales (LLMs, computer vision, etc.) llegaron para quedarse, aunque el hype se enfríe.

---

## Machine Learning: El Corazón de la IA Moderna

### Programación Tradicional vs. Machine Learning

Esta es la distinción más importante de la clase. Prestá atención.

**Programación tradicional:**
```
DATOS + REGLAS → RESULTADO
```
Vos como programador escribís las reglas. Le decís al programa exactamente qué hacer en cada caso.

```python
# Programación tradicional: detectar spam
def es_spam(email):
    palabras_spam = ["gratis", "ganaste", "premio", "click aquí", "oferta"]
    for palabra in palabras_spam:
        if palabra in email.lower():
            return True
    return False
```

El problema: ¿y si los spammers cambian las palabras? ¿Y si escriben "gr4tis"? Tenés que actualizar las reglas manualmente, todo el tiempo.

**Machine Learning:**
```
DATOS + RESULTADOS → REGLAS (el modelo las aprende solo)
```
Le das al sistema miles de emails etiquetados como "spam" o "no spam", y el algoritmo **descubre las reglas solo**.

```python
# Machine Learning: el modelo aprende de los datos
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(emails)  # los emails
y_train = etiquetas  # "spam" o "no_spam"

modelo = MultinomialNB()
modelo.fit(X_train, y_train)  # acá aprende las reglas

# Ahora puede predecir emails nuevos
prediccion = modelo.predict(vectorizer.transform(["Ganaste un premio gratis!"]))
```

> **Analogía clave:** La programación tradicional es como darle a alguien una receta de cocina paso a paso. Machine Learning es como darle 1000 fotos de platos terminados y que la persona descubra sola cómo cocinarlos.

### ¿Cuándo usar ML en vez de programación tradicional?

Usá ML cuando:
- Las reglas son demasiado complejas para escribirlas a mano (reconocimiento facial)
- Las reglas cambian constantemente (detección de fraude)
- Tenés muchos datos y querés encontrar patrones que un humano no ve
- La tarea es inherentemente "difusa" (entender lenguaje natural)

Usá programación tradicional cuando:
- Las reglas son claras y no cambian ("si el usuario tiene más de 18 años, dejalo pasar")
- No tenés datos suficientes para entrenar un modelo
- Necesitás que el comportamiento sea 100% predecible y explicable

---

## Tipos de Machine Learning

Hay tres grandes familias. Pensá en cómo aprende un pibe:

### 1. Aprendizaje Supervisado (Supervised Learning)

**Es como aprender con un profesor que te corrige.**

Le das al modelo datos **etiquetados**: cada ejemplo viene con la respuesta correcta.

- **Input:** foto de un gato, etiqueta: "gato"
- **Input:** foto de un perro, etiqueta: "perro"
- El modelo ve miles de estas y aprende a distinguir gatos de perros.

**Dos subtipos:**
- **Clasificación:** la salida es una categoría (spam/no spam, gato/perro, benigno/maligno)
- **Regresión:** la salida es un número continuo (precio de una casa, temperatura mañana)

**Ejemplos reales:**
- Filtro de spam en Gmail (clasificación)
- Predicción del precio de propiedades en Zonaprop (regresión)
- Diagnóstico médico a partir de imágenes (clasificación)

### 2. Aprendizaje No Supervisado (Unsupervised Learning)

**Es como aprender solo, sin profesor, descubriendo patrones.**

Le das datos **sin etiquetas** y el modelo busca estructura oculta.

**Técnicas principales:**
- **Clustering:** agrupar datos similares. Le das información de 10.000 clientes y el modelo descubre que hay 4 tipos de clientes distintos.
- **Reducción de dimensionalidad:** simplificar datos complejos manteniendo la información importante (PCA, t-SNE).
- **Detección de anomalías:** encontrar datos que no encajan (transacciones fraudulentas).

**Ejemplos reales:**
- Segmentación de clientes en marketing
- Agrupar noticias por tema (Google News)
- Detectar transacciones bancarias sospechosas

> **Analogía:** El aprendizaje supervisado es como clasificar ropa con etiquetas ("remera", "pantalón"). El no supervisado es como ordenar un placard sin etiquetas: agrupás las cosas que se parecen, aunque no tengas nombre para cada grupo.

### 3. Aprendizaje por Refuerzo (Reinforcement Learning)

**Es como aprender por prueba y error, con premios y castigos.**

Un **agente** interactúa con un **entorno**, toma **acciones**, y recibe **recompensas** (positivas o negativas). Con el tiempo, aprende qué acciones maximizan la recompensa.

**Ejemplos:**
- **AlphaGo:** aprendió a jugar Go jugando millones de partidas contra sí mismo.
- **Robots:** un robot aprende a caminar cayéndose miles de veces hasta que descubre cómo mantenerse de pie.
- **Autos autónomos:** aprenden a navegar recibiendo recompensa por llegar a destino y penalización por chocar.
- **ChatGPT y RLHF:** GPT se mejora con feedback humano — le dicen qué respuestas son buenas y cuáles no (Reinforcement Learning from Human Feedback).

> **Analogía:** Es como entrenar un perro. No le explicás las reglas — le das una galletita cuando hace algo bien y lo retás cuando hace algo mal. Con el tiempo, aprende.

---

## ML en la Vida Real: Ejemplos que Usás Todos los Días

Para que veas que esto no es teoría abstracta:

| Aplicación | Tipo de ML | Qué hace |
|---|---|---|
| Netflix/Spotify recomendaciones | Supervisado + No supervisado | Predice qué te va a gustar |
| Google Translate | Supervisado (deep learning) | Traduce entre idiomas |
| Face ID del iPhone | Supervisado | Reconoce tu cara |
| Autocomplete del teclado | Supervisado | Predice la siguiente palabra |
| Detección de fraude bancario | Supervisado + Anomalías | Detecta transacciones sospechosas |
| Google Maps (tiempo estimado) | Supervisado (regresión) | Predice cuánto vas a tardar |
| Piloto automático de Tesla | Refuerzo + Supervisado | Maneja el auto |
| ChatGPT / Claude | Supervisado + RLHF | Genera texto coherente |

### El pipeline típico de un proyecto ML

Para que tengas una visión general de cómo se trabaja en ML (lo vamos a profundizar en las próximas clases):

1. **Definir el problema:** ¿Qué queremos predecir/clasificar/agrupar?
2. **Conseguir datos:** Datasets públicos, APIs, scraping, datos internos.
3. **Preparar los datos:** Limpiar, transformar, normalizar (Clase 2).
4. **Elegir un modelo:** Regresión, árbol de decisión, red neuronal, etc. (Clases 3 y 4).
5. **Entrenar el modelo:** Darle los datos para que aprenda.
6. **Evaluar:** ¿Qué tan bien funciona? Métricas (Clase 3).
7. **Ajustar:** Mejorar hiperparámetros, probar otros modelos.
8. **Desplegar:** Poner el modelo en producción para que lo use gente real.

---

## Python y el Ecosistema de ML

Python es **el lenguaje dominante** en ML y Data Science. ¿Por qué?

- **Librerías increíbles:** scikit-learn, TensorFlow, PyTorch, pandas, numpy
- **Comunidad enorme:** si tenés un problema, alguien ya lo resolvió
- **Sintaxis simple:** podés enfocarte en el problema, no en el lenguaje
- **Jupyter Notebooks:** ideales para experimentar y visualizar

**Las librerías que más vamos a usar:**

```python
import numpy as np          # operaciones numéricas
import pandas as pd         # manejo de datos tabulares
import matplotlib.pyplot as plt  # gráficos
import seaborn as sns       # gráficos lindos
from sklearn import ...     # TODA la maquinaria de ML
```

En las próximas clases vamos a meter mano con código de verdad. Por ahora, lo importante es que entiendas los conceptos.

---

## 📝 Conceptos clave para recordar

1. **IA es un campo amplio** que incluye muchas técnicas. Machine Learning es una de ellas (la más importante hoy).

2. **Distinguí IA real de marketing.** No todo lo que dice "IA" en la caja realmente lo es.

3. **Test de Turing:** un test de imitación. Útil históricamente, pero limitado. Pasar el test no implica "pensar".

4. **IA débil (Narrow AI):** sistemas buenos en una tarea. Es TODO lo que existe hoy.

5. **IA fuerte (AGI):** inteligencia general como la humana. No existe todavía.

6. **Inviernos de la IA:** períodos donde el hype superó a la realidad y la inversión se frenó. Lección: no creerse todo el hype.

7. **Programación tradicional:** vos escribís las reglas. **ML:** el sistema aprende las reglas de los datos.

8. **Tres tipos de ML:**
   - **Supervisado:** aprende con datos etiquetados (clasificación y regresión)
   - **No supervisado:** encuentra patrones sin etiquetas (clustering, anomalías)
   - **Refuerzo:** aprende por prueba y error con recompensas

9. **ML está en todos lados:** recomendaciones, traducción, detección de fraude, autos autónomos, asistentes virtuales.

10. **Python es el lenguaje de ML** gracias a scikit-learn, pandas, numpy, TensorFlow y PyTorch.
