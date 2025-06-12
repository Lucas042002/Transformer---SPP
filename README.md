# Transformer + SPP

## Contexto

Este proyecto aborda el **problema de empaquetado de rectángulos en 2D** (2D Strip Packing Problem, SPP), un clásico problema de optimización combinatoria. El objetivo es acomodar un conjunto de rectángulos de diferentes dimensiones dentro de un contenedor de ancho fijo y altura mínima, permitiendo rotaciones de 90° y sin permitir cortes guillotinables.

Los archivos de datos (`c1p1.txt`, `c1p2.txt`, `c1p3.txt`, etc.) contienen instancias de prueba extraídas de benchmarks clásicos, como los presentados en el paper de Hopper y Turton (European Journal of Operations Research, 1999). Cada archivo representa una instancia con una lista de rectángulos (alto, ancho) a empaquetar.

El archivo principal `hr_algorithm.py` implementa un **algoritmo heurístico recursivo (HR)** para resolver el SPP, incluyendo funciones para:

- Ordenar los rectángulos por área.
- Empaquetar los rectángulos usando heurísticas de división de espacios.
- Visualizar la solución obtenida.

## Estructura de archivos

- **hr_algorithm.py**: Implementación del algoritmo heurístico recursivo y funciones de visualización.
- **transformer.py**: Implementación de un modelo Transformer personalizado para el SPP.
- **main.py**: Script principal para ejecutar experimentos, entrenamiento y evaluación de modelos.
- **cXpY.txt**: Instancias de prueba (C1, C2, ..., C7; P1, P2, P3) con dimensiones de los rectángulos.
- **strip1.txt**: Descripción detallada de los benchmarks y contexto académico.

## Integración de Transformers

El proyecto incluye la integración de un **modelo Transformer** (ver `transformer.py`) para abordar el SPP desde un enfoque de aprendizaje profundo. El modelo recibe secuencias de estados generadas por el algoritmo heurístico y aprende a predecir la acción óptima (rectángulo a colocar) en cada paso. El flujo de trabajo incluye:

- Generación de datos de entrenamiento a partir de la heurística.
- Preparación de secuencias de estados y etiquetas (acciones) como tensores.
- Entrenamiento supervisado del Transformer usando `nn.CrossEntropyLoss`.
- Evaluación y visualización de métricas como la pérdida y accuracy por época.

Esto permite comparar el desempeño entre heurísticas clásicas y modelos basados en Transformers, y explorar la capacidad del modelo para aprender estrategias de empaquetado eficientes.

---

**Referencias:**

- Hopper, E., & Turton, B. C. H. (1999). An Empirical Investigation of Meta-heuristic and Heuristic Algorithms for a 2D Packing Problem. *European Journal of Operations Research*.

---