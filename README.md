# Predicción de resultados de batallas en Clash Royale con Machine Learning

Primer proyecto de la asignatura **Técnicas de Aprendizaje de Máquina** de la Pontificia Universidad Javeriana (Bogotá, 2024-3). El proyecto compara tres enfoques de clasificación supervisada para predecir el resultado (victoria/derrota) de partidas de alto nivel de Clash Royale.

## Tabla de contenidos

- [Descripción](#descripción)
- [Datos](#datos)
- [Modelos implementados](#modelos-implementados)
- [Resultados](#resultados)
- [Tecnologías](#tecnologías)
- [Instalación y ejecución](#instalación-y-ejecución)
- [Estructura del repositorio](#estructura-del-repositorio)
- [Conclusiones](#conclusiones)
- [Autores](#autores)

## Descripción

Clash Royale es un juego de estrategia en tiempo real con un ecosistema complejo de cartas, tropas y mecánicas que influyen en el desenlace de cada batalla. El objetivo del proyecto fue responder a la pregunta:

> ¿Cómo podemos utilizar diferentes modelos de machine learning para predecir con precisión los resultados de las batallas en Clash Royale?

Para responderla se implementaron, entrenaron y evaluaron tres modelos de clasificación binaria, comparando su desempeño con métricas estándar (precisión, recall, F1-score, matriz de confusión y curva ROC/AUC).

## Datos

Se trabajó con dos conjuntos de datos:

| Dataset | Dimensiones | Descripción |
|---------|-------------|-------------|
| `cardlist.csv` | 106 x 3 | Lista de cartas con sus IDs |
| `data_ord.csv` | 718.886 x 20 | Registros detallados de partidas |

La variable objetivo es `outcome` (resultado binario: victoria/derrota del Jugador 1). Las variables predictoras corresponden a las 8 cartas del mazo de cada jugador (`p1card1`..`p1card8`, `p2card1`..`p2card8`) y otras variables de partida.

### Preprocesamiento común

- Codificación one-hot (dummies) de las columnas de cartas mediante `pd.get_dummies`.
- División entrenamiento/prueba 70/30 con `train_test_split` (random_state=42).
- Normalización con `StandardScaler` (media 0, desviación 1).
- Reducción de dimensionalidad con **PCA a 34 componentes** para los modelos 1 y 2.

## Modelos implementados

### 1. Regresión logística

- `LogisticRegression` con `class_weight='balanced'` para compensar el desbalance de clases.
- Umbral de decisión ajustado a **0.4** para mejorar la sensibilidad en la clase minoritaria.

### 2. Red neuronal feed-forward

Arquitectura secuencial implementada en Keras/TensorFlow:

| Capa | Neuronas | Activación |
|------|----------|------------|
| Entrada (Dense) | 28 | ReLU |
| Oculta (Dense) | 14 | ReLU |
| Salida (Dense) | 1 | Sigmoid |

- Total de parámetros entrenables: **1.401**
- Optimizador: Adam (learning_rate = 0.001)
- Función de pérdida: `binary_crossentropy`
- 25 épocas, batch size = 64
- Umbral de decisión: 0.56

### 3. Modelo híbrido con LightGBM

Pipeline que combina preprocesamiento avanzado y un clasificador eficiente:

1. Muestreo aleatorio del 10% del dataset para manejar el volumen de datos.
2. Imputación de valores faltantes con `SimpleImputer`.
3. Escalado con `StandardScaler`.
4. Selección de características con `SelectFromModel` usando LightGBM.
5. Clasificación final con `LGBMClassifier` (n_estimators=100, learning_rate=0.1).

Durante el desarrollo también se probaron Random Forest, SVM con distintos kernels, XGBoost y técnicas de stacking, pero LightGBM ofreció el mejor balance entre tiempo de cómputo y desempeño.

## Resultados

### Regresión logística

```
              precision    recall  f1-score   support
           0       0.52      0.00      0.01     95434
           1       0.56      1.00      0.72    120232
    accuracy                           0.56    215666
```

- **Matriz de confusión:** [[262, 95172], [243, 119989]]
- **AUC ROC:** 0.50 (sin capacidad discriminativa)
- Fuerte sesgo hacia la clase mayoritaria.

### Red neuronal feed-forward

```
              precision    recall  f1-score   support
           0       0.60      0.48      0.54    119471
           1       0.49      0.61      0.54     96195
    accuracy                           0.54    215666
```

- **Matriz de confusión:** [[25045, 70389], [22672, 97560]]
- **AUC ROC:** 0.57
- Mejor balance entre clases que la regresión logística, aunque con signos de sobreajuste en la curva de error de validación.

### Modelo híbrido (LightGBM)

```
              precision    recall  f1-score   support
           0       0.51      0.18      0.26      9513
           1       0.57      0.87      0.69     12054
    accuracy                           0.56     21567
```

- **Precisión global:** 0.5634
- **AUC ROC:** 0.56
- Rendimiento moderado superando ligeramente al azar, con sesgo persistente hacia la clase mayoritaria.

### Comparación

| Modelo | Accuracy | AUC | Comentario |
|--------|----------|-----|------------|
| Regresión logística | 0.56 | 0.50 | Predice casi siempre la clase mayoritaria |
| Red neuronal | 0.54 | 0.57 | Mejor capacidad discriminativa, posible sobreajuste |
| Híbrido LightGBM | 0.56 | 0.56 | Capacidad predictiva modesta pero estable |

## Tecnologías

- Python 3
- pandas, NumPy
- scikit-learn (PCA, StandardScaler, LogisticRegression, train_test_split, métricas)
- TensorFlow / Keras (red neuronal feed-forward)
- LightGBM (modelo híbrido)
- Matplotlib, Seaborn (visualizaciones)
- Jupyter Notebook

## Instalación y ejecución

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/clash-royale-ml.git
cd clash-royale-ml
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Datos

El notebook espera dos archivos en la raíz del proyecto:

- `data_ord.csv`
- `cardlist.csv`

Estos datasets no se incluyen en el repositorio por su tamaño. Deben obtenerse de la fuente original del proyecto académico.

### 4. Ejecutar el notebook

```bash
jupyter notebook TecnicasProyecto_VFinal.ipynb
```

## Estructura del repositorio

```
clash-royale-ml/
├── TecnicasProyecto_VFinal.ipynb            # Notebook con los tres modelos
├── Informe_Tecnicas_Aprendizaje_Maquina.pdf # Documento académico del proyecto
├── README.md                                # Este archivo
├── requirements.txt                         # Dependencias
└── .gitignore                               # Archivos ignorados por git
```

## Conclusiones

La predicción de resultados de batallas en Clash Royale demostró ser una tarea compleja. Los tres modelos obtuvieron desempeños moderados, ligeramente superiores al azar, lo que sugiere que las variables disponibles (composición de mazos) no capturan completamente los factores que determinan el resultado de una partida: habilidad del jugador, momento de juego, sinergias y patrones tácticos no presentes en los datos.

Hallazgos principales:

- La regresión logística no logró discriminar entre clases (AUC = 0.50) y mostró un fuerte sesgo hacia la clase mayoritaria.
- La red neuronal mejoró el balance entre clases, pero mostró indicios de sobreajuste.
- El modelo híbrido con LightGBM ofreció el mejor balance entre rendimiento y costo computacional, aunque también obtuvo una capacidad predictiva modesta.
- Posibles líneas futuras: ingeniería de características basada en sinergias entre cartas, embeddings, modelos secuenciales que consideren el orden de jugadas, y técnicas explícitas de balanceo (SMOTE, undersampling).

El proyecto ilustra desafíos típicos de aplicar machine learning en escenarios reales: selección de características, manejo del desbalance de clases y la importancia de iterar sobre los modelos para mejorar su capacidad predictiva.

## Autores

- **Diego Caballero Sarmiento**
- **Liseth Lozano**
- **Aura Atuesta**
