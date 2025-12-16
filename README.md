# Transformer + SPP 

**Solución híbrida del Strip Packing Problem usando Transformers y Aprendizaje por Imitación**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

##  Tabla de Contenidos

- [Contexto](#contexto)
- [Instalación](#instalación)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Modelos Pre-entrenados](#modelos-pre-entrenados)
- [Guía de Uso Rápido](#guía-de-uso-rápido)
- [Manual Completo](#manual-completo)
- [Arquitectura del Modelo](#arquitectura-del-modelo)
- [Resultados](#resultados)
- [Referencias](#referencias)

---

##  Contexto

Este proyecto aborda el **Strip Packing Problem (SPP)**, un problema NP-hard de optimización combinatoria que consiste en empaquetar rectángulos de diferentes dimensiones en un contenedor de ancho fijo minimizando la altura total.

**Características del problema:**
- Rotaciones de 90° permitidas
- División de espacios tipo guillotina
- Sin solapamiento entre rectángulos
- Benchmarks clásicos de Hopper & Turton (1999)

**Solución propuesta:**
- **Modelo Pointer Network** basado en Transformer que aprende del algoritmo heurístico HR
- **8.5× más rápido** que HR manteniendo calidad competitiva (Gap 24.59%)
- **Imitation Learning** supervisado con data augmentation
- **Arquitectura híbrida** que combina aprendizaje profundo con heurísticas clásicas

---

## Instalación

### **Requisitos del Sistema**
- Python 3.10 o superior
- GPU NVIDIA (opcional, recomendado para entrenamiento)
- 8GB RAM mínimo (16GB recomendado)

### **1. Clonar el repositorio**
```bash
git clone https://github.com/Lucas042002/Transformer---SPP.git
cd Transformer---SPP
```

### **2. Crear entorno virtual (recomendado)**
```bash
# Con venv
python -m venv venv

# Activar entorno
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

### **3. Instalar dependencias**
```bash
pip install -r requirements.txt
```

### **Librerías principales requeridas:**

#### **Core (obligatorias)**
```bash
torch>=2.0.0              # Framework de deep learning
numpy>=1.24.0             # Computación numérica
matplotlib>=3.7.0         # Visualización
```

#### **Adicionales (opcionales pero recomendadas)**
```bash
pandas>=2.0.0             # Análisis de datos (para analisis_complejidad.py)
seaborn>=0.12.0           # Visualización estadística
scipy>=1.10.0             # Funciones científicas
tqdm>=4.65.0              # Barras de progreso
```

#### **Instalación rápida completa:**
```bash
pip install torch numpy matplotlib pandas seaborn scipy tqdm
```

#### **Para GPU (CUDA 11.8):**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### **4. Verificar instalación**
```bash
python -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'CUDA disponible: {torch.cuda.is_available()}')"
```

---

## Estructura del Proyecto

```
Transformer---SPP/
│
├── Core - Algoritmos y Modelos
│   ├── hr_algorithm.py           # Algoritmo Heurístico Recursivo (HR) + generación de datos
│   ├── hr_pointer.py             # Versión híbrida HR + Pointer Model
│   ├── pointer_model.py          # Arquitectura Transformer (Encoder-Decoder)
│   ├── pointer_training.py       # Pipeline de entrenamiento (Imitation Learning)
│   └── main.py                   # Script principal (entrenar/testear/visualizar)
│
├── Testing y Análisis
│   ├── test_pointer.py           # Suite de testing con comparaciones
│   ├── analisis_complejidad.py   # Análisis de complejidad temporal/espacial
│   ├── greedy_algorithms.py      # Algoritmos baseline (FFDH, BFDH, BL, NF)
│   └── run_pointer.py            # Script rápido de prueba
│
├── Visualización y Documentación
│   ├── GUIA_TESTING.md           # Guía completa de testing
│   ├── VISUALIZACION_TESTING.md  # Documentación de visualización
│   └── README.md                 # Este archivo
│
├── Datos y Configuración
│   ├── generator.py              # Generador de problemas sintéticos
│   ├── categories.py             # Definición de categorías C1-C7
│   ├── states.py                 # Codificación de features (10+19 dims)
│   ├── tests/                    # Instancias benchmark (c1p1.txt, c2p1.txt, ...)
│   └── strip1.txt                # Descripción de benchmarks originales
│
├── Modelos Pre-entrenados (4 modelos)
│   └── models/
│       ├── pointer_c1_L4_H8_acc8212.pth    # Modelo C1 (82.12% accuracy)
│       ├── pointer_c2_L4_H8_acc8155.pth    # Modelo C2 (81.55% accuracy)
│       ├── pointer_c3_L4_H8_acc7616.pth    # Modelo C3 (76.16% accuracy)
│       └── pointer_c4_L4_H8_acc7497.pth    # Modelo C4 (74.97% accuracy)
│
└── Resultados (generados automáticamente)
    └── img/
        ├── complexity_analysis/       # Análisis de complejidad
        │   ├── tabla_C1.csv
        │   ├── tabla_C2.csv
        │   ├── mejores_C1/            # Mejores casos por categoría
        │   └── ...
        └── *.png                      # Gráficos de entrenamiento
```

---

## Modelos Pre-entrenados

El repositorio incluye **4 modelos entrenados** listos para usar:

| Categoría | Archivo | Accuracy | N Rectángulos | Tamaño |
|-----------|---------|----------|---------------|--------|
| **C1** | `pointer_c1_L4_H8_acc8212.pth` | 82.12% | 17 | ~20MB |
| **C2** | `pointer_c2_L4_H8_acc8155.pth` | 81.55% | 25 | ~20MB |
| **C3** | `pointer_c3_L4_H8_acc7616.pth` | 76.16% | 29 | ~20MB |
| **C4** | `pointer_c4_L4_H8_acc7497.pth` | 74.97% | 49 | ~20MB |

**Hiperparámetros comunes:**
- `d_model=256` (dimensión del modelo)
- `num_enc_layers=4` (capas del encoder)
- `num_heads=8` (attention heads)
- `d_ff=512` (feed-forward hidden dim)
- Entrenados con 500 problemas base × 3 (augmentation) = 1500 trayectorias

**Uso rápido:**
```bash
# Testear modelo C1 pre-entrenado con 50 problemas
python main.py C1 test models/pointer_c1_L4_H8_acc8212.pth 50

# Visualizar casos extremos (mejores/peores)
python main.py C2 best-worst models/pointer_c2_L4_H8_acc8155.pth 30
```

---

## Guía de Uso Rápido

### ** Modo 1: Usar modelos pre-entrenados (Recomendado)**

```bash
# Testear modelo C1 con 50 problemas
python main.py C1 test models/pointer_c1_L4_H8_acc8212.pth 50

# Comparar con todos los algoritmos (HR, FFDH, BFDH, etc.)
python main.py C2 compare models/pointer_c2_L4_H8_acc8155.pth 30

# Visualizar problema individual (Pointer vs HR)
python main.py C3 visual models/pointer_c3_L4_H8_acc7616.pth

# Analizar mejores y peores casos
python main.py C4 best-worst models/pointer_c4_L4_H8_acc7497.pth 20
```

### ** Modo 2: Entrenar nuevo modelo**

```bash
# Entrenar modelo para categoría C1 (usa configuración por defecto)
python main.py C1

# Entrenar para otra categoría
python main.py C3
```

**Configuración de entrenamiento (en `main.py`):**
-  500 problemas base × 3 (augmentation) = 1500 trayectorias
-  200 épocas con early stopping (paciencia 40)
-  Batch size: 16
-  Learning rate: 1e-4 (AdamW)
-  Tiempo aproximado: 2-4 horas (GPU) / 8-12 horas (CPU)

### ** Modo 3: Análisis de complejidad completo**

```bash
# Analizar todas las categorías con 500 problemas cada una
python main.py complexity C1 C2 C3 C4 500
```

Genera:
- Tablas CSV con métricas (tiempo, memoria, altura, gaps)
- Boxplots comparativos
- Identificación de mejores/peores casos
- Análisis de superioridad Pointer vs HR

---

## Manual Completo

### **Comandos disponibles**

#### **1. Entrenamiento**
```bash
python main.py <CATEGORIA>
```
- `<CATEGORIA>`: C1, C2, C3, C4, C5, C6, C7
- Entrena un nuevo modelo desde cero
- Guarda automáticamente en `models/pointer_<cat>_L<layers>_H<heads>_acc<accuracy>.pth`
- Genera gráficas de entrenamiento en `img/`

**Ejemplo:**
```bash
python main.py C1
# Output: models/pointer_c1_L4_H8_acc8212.pth
#         img/pointer_c1_L4_H8_acc8212.png
```

---

#### **2. Testing**
```bash
python main.py <CATEGORIA> test [<MODELO_PATH>] [<N_PROBLEMAS>]
```
- Sin argumentos: usa último modelo entrenado con 10 problemas
- `<MODELO_PATH>`: ruta al archivo .pth específico
- `<N_PROBLEMAS>`: cantidad de problemas a evaluar

**Ejemplos:**
```bash
# Testear último modelo C1 con 10 problemas
python main.py C1 test

# Testear modelo específico con 50 problemas
python main.py C2 test models/pointer_c2_L4_H8_acc8155.pth 50

# Evaluación exhaustiva (100 problemas)
python main.py C3 test models/pointer_c3_L4_H8_acc7616.pth 100
```

**Output:**
```
Estadísticas del Pointer Model:
  Problemas evaluados: 50
  Altura promedio: 45.32
  Altura mínima: 32
  Altura máxima: 67
  Desv. estándar: 8.45
  
Comparación vs HR:
  Casos superiores: 3 (6.0%)
  Casos iguales: 5 (10.0%)
  Casos inferiores: 42 (84.0%)
  Gap promedio vs HR: 23.45%
```

---

#### **3. Comparación completa**
```bash
python main.py <CATEGORIA> compare [<MODELO_PATH>] [<N_PROBLEMAS>]
```
Compara el Pointer Model con **todos los algoritmos baseline**:
- HR (Heuristic Recursive)
- FFDH (First Fit Decreasing Height)
- BFDH (Best Fit Decreasing Height)
- Next Fit
- Bottom-Left

**Ejemplo:**
```bash
python main.py C1 compare models/pointer_c1_L4_H8_acc8212.pth 30
```

**Output:**
```
Resultados Comparativos (30 problemas):
┌─────────────┬──────────┬──────────┬──────────┬────────────┐
│ Algoritmo   │ Altura   │ Tiempo   │ Gap HR   │ Gap Href   │
├─────────────┼──────────┼──────────┼──────────┼────────────┤
│ HR          │ 35.20    │ 245.3ms  │ 0.00%    │ 12.45%     │
│ Pointer     │ 43.87    │ 28.5ms   │ 24.63%   │ 39.89%     │
│ FFDH        │ 45.12    │ 3.2ms    │ 28.18%   │ 44.00%     │
│ BFDH        │ 45.01    │ 3.5ms    │ 27.86%   │ 43.65%     │
│ Bottom-Left │ 39.85    │ 15.7ms   │ 13.21%   │ 27.20%     │
│ Next Fit    │ 58.34    │ 1.8ms    │ 65.68%   │ 86.30%     │
└─────────────┴──────────┴──────────┴──────────┴────────────┘

Speedup Pointer vs HR: 8.6×
```

---

#### **4. Visualización interactiva**
```bash
python main.py <CATEGORIA> visual [<MODELO_PATH>]
```
Visualiza un problema aleatorio con comparación lado a lado Pointer vs HR.

**Ejemplo:**
```bash
python main.py C2 visual models/pointer_c2_L4_H8_acc8155.pth
```

**Muestra:**
- Gráfico comparativo (Pointer a la izquierda, HR a la derecha)
- Altura de cada solución con línea roja en Href
- Colores únicos por rectángulo
- Diferencia absoluta y porcentual

---

#### **5. Análisis de casos extremos**
```bash
python main.py <CATEGORIA> best-worst [<MODELO_PATH>] [<N_PROBLEMAS>]
```
Identifica y visualiza los **3 mejores** y **3 peores** casos del modelo.

**Ejemplo:**
```bash
python main.py C3 best-worst models/pointer_c3_L4_H8_acc7616.pth 50
```

**Output:**
```
Analizando 50 problemas...

 TOP 3 MEJORES CASOS (Pointer supera a HR):
  #1: Problema 12 → Pointer: 45, HR: 48 (diferencia: -3, -6.25%)
  #2: Problema 37 → Pointer: 52, HR: 54 (diferencia: -2, -3.70%)
  #3: Problema 8  → Pointer: 38, HR: 39 (diferencia: -1, -2.56%)

 TOP 3 PEORES CASOS (Pointer muy inferior a HR):
  #1: Problema 23 → Pointer: 87, HR: 45 (diferencia: +42, +93.33%)
  #2: Problema 41 → Pointer: 72, HR: 48 (diferencia: +24, +50.00%)
  #3: Problema 15 → Pointer: 65, HR: 47 (diferencia: +18, +38.30%)

[Se generan 6 visualizaciones comparativas]
```

---

#### **6. Análisis de complejidad completo**
```bash
python main.py complexity <CAT1> <CAT2> ... <N_PROBLEMAS>
```
Ejecuta análisis exhaustivo en múltiples categorías simultáneamente.

**Ejemplo:**
```bash
python main.py complexity C1 C2 C3 C4 500
```

**Genera:**
1. **Tablas CSV** (`img/complexity_analysis/tabla_C1.csv`, etc.)
   - Tiempo (ms)
   - Memoria (MB)
   - Altura final
   - Gap vs Href
   - Gap vs HR

2. **Tabla unificada** (`tabla_unificada.csv`) con métricas globales

3. **Archivos de mejores casos** (`img/complexity_analysis/mejores_C1/`)

4. **Boxplots comparativos** por algoritmo y categoría

**Tiempo estimado:** 2-4 horas para 500 problemas × 4 categorías

---

### **Personalización avanzada**

#### **Modificar hiperparámetros de entrenamiento**
Editar `main.py` líneas 280-295:

```python
NUM_PROBLEMAS = 500              # Problemas base
BATCH_SIZE = 16                  # Tamaño de batch
EPOCHS = 200                     # Épocas máximas
NUM_ENC_LAYERS = 4               # Capas del encoder
NUM_HEADS = 8                    # Attention heads
LEARNING_RATE = 1e-4             # Learning rate
EARLY_STOPPING_PATIENCE = 40     # Paciencia early stopping
augment_permutations=2           # Data augmentation (×3 datos)
```

#### **Modificar dimensiones del modelo**
Editar `main.py` línea 335:

```python
model = SPPPointerModel(
    d_model=256,           # Dimensión embeddings
    rect_feat_dim=10,      # Features rectángulos
    space_feat_dim=19,     # Features espacios
    num_enc_layers=4,      # Capas encoder
    num_heads=8,           # Attention heads
    d_ff=512,              # FFN hidden dim
    dropout=0.1,           # Dropout rate
    max_steps=N            # Máximo pasos
)
```

---

### **Solución de problemas comunes**

#### **Error: "CUDA out of memory"**
```bash
# Reducir batch size en main.py
BATCH_SIZE = 8  # o menor
```

#### **Error: "libiomp5md.dll duplicado" (Windows)**
Ya está solucionado en el código:
```python
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
```

#### **Entrenamiento muy lento (CPU)**
```bash
# Reducir problemas y épocas para prueba rápida
NUM_PROBLEMAS = 100
EPOCHS = 50
```

#### **Modelos no encontrados**
```bash
# Verificar que exista la carpeta models/
ls models/

# Si no existe, crearla:
mkdir models
```

---

## Arquitectura del Modelo

### **Pointer Network basado en Transformer**

El modelo aprende a resolver el SPP mediante **Imitation Learning**, imitando las decisiones del algoritmo HR experto.

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT STAGE                          │
├─────────────────────────────────────────────────────────┤
│  Rectángulos (N×10)         Espacio Activo (19)        │
│  • Altura, ancho            • Posición (x, y)           │
│  • Área                     • Dimensiones (w, h)        │
│  • Aspect ratio             • Utilización potencial     │
│  • Compactness              • Fragmentación             │
│  • Categoría tamaño         • Compatibilidad (8 dims)   │
│  • Ranking área             • Fit quality               │
│  • Densidad                 • Altura actual             │
│  • ... (10 features)        • ... (19 features)         │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│              ENCODER (Transformer, L=4)                 │
├─────────────────────────────────────────────────────────┤
│  RectEncoder:                                           │
│  • Proyección lineal: 10 → d_model (256)                │
│  • 4 capas TransformerEncoder                           │
│  • Multi-head attention (8 heads)                       │
│  • Feed-forward (d_ff=512)                              │
│  • Dropout (0.1)                                        │
│                                                         │
│  Output: Rect Embeddings (N × 256) [CACHEADO]          │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│           DECODER (por cada paso t=1...N)               │
├─────────────────────────────────────────────────────────┤
│  1. SpaceEncoder:                                       │
│     space_feat (19) → space_emb (256)                   │
│                                                         │
│  2. Global Context (dinámico):                          │
│     global_ctx = MeanPool(RectEnc[mask])                │
│                                                         │
│  3. StepEmbedding:                                      │
│     step_emb = Embedding(t)                             │
│                                                         │
│  4. QueryBuilder:                                       │
│     query = FFN([space_emb || global_ctx || step_emb])  │
│                                                         │
│  5. PointerAttention:                                   │
│     scores = (query · RectEnc^T) / √d                   │
│     scores[~mask] = -∞                                  │
│     probs = softmax(scores)                             │
│                                                         │
│  Output: Probabilidades sobre rectángulos (N)           │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│                  TRAINING (Supervised)                  │
├─────────────────────────────────────────────────────────┤
│  Loss: Cross-Entropy (con weighted loss opcional)       │
│  L = (1/T) Σ w_t · (-log(p_t[y_t]))                     │
│  donde w_t ∈ [1.0, 2.0] (pasos finales pesan más)       │
│                                                         │
│  Optimizer: AdamW (lr=1e-4, weight_decay=1e-4)          │
│  Data Augmentation: 2 permutaciones × problema (×3)     │
│  Early Stopping: Paciencia 40 épocas                    │
└─────────────────────────────────────────────────────────┘
```

### **Complejidad Computacional**

| Componente | Complejidad | Explicación |
|------------|-------------|-------------|
| **Encoder** (1 vez) | O(L·N²·d) | L=4 capas, self-attention cuadrática |
| **Decoder** (N pasos) | O(N²·d + N·d²) | Atención + FFN |
| **Total** | **O(L·N²·d + N·d²)** | Dominado por self-attention |

**Comparación:**
- **HR:** O(N²·M) donde M = espacios activos (búsqueda exhaustiva)
- **Pointer:** O(N²·d) con d=256 (paralelizable en GPU)
- **Speedup observado:** **8.5× en promedio**, hasta 16.4× en C4

---

## Resultados Experimentales

### **Métricas Generales (2000 problemas de test)**

| Métrica | Valor | Observaciones |
|---------|-------|---------------|
| **Accuracy promedio** | 78.85% | Decisiones correctas vs HR |
| **Gap vs HR** | 24.59% | Altura promedio 24.59% mayor que HR |
| **Gap vs Href** | 57.63% | Respecto a altura de referencia óptima |
| **Tiempo promedio** | 28.43 ms | vs 241.09 ms de HR (8.5× más rápido) |
| **Casos superiores** | 52 (2.6%) | Pointer encuentra mejor solución que HR |
| **Casos iguales** | 166 (8.3%) | Pointer iguala a HR |
| **Casos inferiores** | 1782 (89.1%) | Pointer inferior a HR |

### **Desempeño por Categoría**

| Categoría | N Rects | Accuracy | Gap HR | Tiempo Pointer | Speedup |
|-----------|---------|----------|--------|----------------|---------|
| **C1** | 17 | 82.12% | 18.45% | 15.2 ms | 7.8× |
| **C2** | 25 | 81.55% | 22.67% | 22.5 ms | 8.2× |
| **C3** | 29 | 76.16% | 27.83% | 28.9 ms | 9.1× |
| **C4** | 49 | 74.97% | 29.41% | 47.3 ms | 16.4× |

### **Comparación con Algoritmos Baseline**

```
Ranking por Altura Final (menor = mejor):
1. HR (Heuristic Recursion)     → Gap: 0.00%    Referencia
2. Bottom-Left                  → Gap: 13.43%   Segundo mejor
3. Pointer Model                → Gap: 24.59%   Tercero (pero 8.5× más rápido)
4. BFDH                         → Gap: 28.40%
5. FFDH                         → Gap: 28.50%
6. Next Fit                     → Gap: 76.89%

Ranking por Tiempo (menor = mejor):
1. Next Fit        → 2.1 ms    Más rápido (pero peor calidad)
2. FFDH            → 3.8 ms
3. BFDH            → 4.2 ms
4. Pointer Model   → 28.4 ms   Balance calidad/velocidad
5. Bottom-Left     → 156.7 ms
6. HR              → 241.1 ms
```

### **Análisis Crítico**

**Fortalezas identificadas:**
- Speedup consistente (8.5×) sin hardware especializado
- Ocasionalmente descubre soluciones superiores (2.6% casos)
- Iguala a HR en 8.3% de instancias (estrategias equivalentes)
- Mejor que algoritmos greedy clásicos (FFDH, BFDH)

**Limitaciones observadas:**
- Fragmentación vertical excesiva en casos complejos
- Dificultad con rectángulos heterogéneos (C3, C4)
- Decisiones tempranas subóptimas propagan errores
- Representación vectorial fija (d=256) limita captura espacial

**Mejoras futuras propuestas:**
- Graph Neural Networks para relaciones espaciales explícitas
- Reinforcement Learning con reward shaping directo
- Beam search con k candidatos (exploración limitada)
- Refinamiento post-procesamiento con movimientos locales

Ver documentación completa en la tesis para análisis detallado.

---

## Documentación Adicional

- **GUIA_TESTING.md** - Guía exhaustiva de testing y evaluación
- **VISUALIZACION_TESTING.md** - Sistema de visualización completa
- **Tesis completa** - Fundamentos teóricos y análisis experimental

---

##  Contribuciones

Este proyecto es parte de una tesis de pregrado. Para consultas o colaboraciones:

-  Email: lucas.sepulveda@example.com
-  GitHub: [@Lucas042002](https://github.com/Lucas042002)

---

##  Referencias

1. **Hopper, E., & Turton, B. C. H. (1999).** "An Empirical Investigation of Meta-heuristic and Heuristic Algorithms for a 2D Packing Problem." *European Journal of Operations Research*, 113(3), 503-521.

2. **Vaswani, A., et al. (2017).** "Attention is All You Need." *Advances in Neural Information Processing Systems (NeurIPS)*, 30.

3. **Vinyals, O., Fortunato, M., & Jaitly, N. (2015).** "Pointer Networks." *Advances in Neural Information Processing Systems (NeurIPS)*, 28.

4. **Bello, I., et al. (2016).** "Neural Combinatorial Optimization with Reinforcement Learning." *arXiv preprint arXiv:1611.09940*.

---

##  Agradecimientos

- Benchmarks basados en el trabajo de Hopper & Turton (1999)
- Arquitectura inspirada en Pointer Networks (Vinyals et al., 2015) y Transformer (Vaswani et al., 2017)
- Framework PyTorch para implementación eficiente del modelo
- Comunidad de investigación en optimización combinatoria y deep learning

---

**Desarrollado como parte de tesis de pregrado en Ingeniería Civil Informatica**

*Última actualización: Noviembre 2025*