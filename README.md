# Tarea de Ejecución 03 — Modelos CART y Ensamblados

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-red.svg)](https://xgboost.readthedocs.io/)

Este repositorio contiene la solución completa, rigurosa y reproducible de la **Tarea de Ejecución 03: Aplicar modelos CART y ensamblados**, desarrollada para el curso de **Machine Learning**.

---

## 🚀 Cómo abrir y ejecutar en Google Colab

Puedes ejecutar el notebook interactivo directamente en Google Colab siguiendo cualquiera de estos pasos:

### Opción A (Recomendada - Carga Directa):
1. Ingresa a [Google Colab](https://colab.research.google.com/).
2. Haz clic en la pestaña **Subir** (*Upload*).
3. Selecciona o arrastra el archivo local [`CART_Ensambles.ipynb`](CART_Ensambles.ipynb).
4. En el menú superior de Colab, selecciona **Entorno de ejecución** > **Ejecutar todas**.

### Opción B (Desde Google Drive):
1. Sube [`CART_Ensambles.ipynb`](CART_Ensambles.ipynb) a tu unidad de Google Drive.
2. Haz clic derecho sobre el archivo > **Abrir con** > **Google Colaboratory**.

### Opción C (Desde GitHub si publicas el repositorio):
1. Sube este proyecto a tu repositorio de GitHub (ej. `https://github.com/tu-usuario/t3-am`).
2. Abre la siguiente URL en tu navegador:
   ```text
   https://colab.research.google.com/github/tu-usuario/t3-am/blob/main/CART_Ensambles.ipynb
   ```

> **Nota sobre datos y dependencias en Colab:**
> - El notebook instala automáticamente `xgboost` si no se encuentra disponible.
> - El dataset `auto_mpg` cuenta con carga por contingencia en 4 niveles (archivo local de Kaggle, repositorio público `seaborn-data`, UCI ML Repository oficial, o subida manual). No requiere llaves de API ni configuraciones adicionales.

---

## 📋 Resumen de Actividades y Respuestas a las Preguntas

Cada ítem equivale a **4 puntos** (Total: 20/20 pts). Todos los modelos fueron entrenados **desde cero** (sin modelos preentrenados), evaluados en partición de prueba (Hold-Out 80/20) y validados mediante **Validación Cruzada Repetida (5 pliegues × 3 repeticiones = 15 evaluaciones)** con pruebas estadísticas pareadas de **Wilcoxon** para determinar significancia real ($p < 0.05$).

| Actividad | Dataset | Modelos comparados | Métrica clave | Ganador / Respuesta a "¿Cuál es mejor?" |
|---|---|---|---|---|
| **1. CART Gini vs Entropía** | `breast_cancer` | Árboles (Gini y Entropía con/sin poda) vs Regresión Logística | Accuracy CV / AUC | **Regresión Logística** (97.6% CV vs ≤93.7% CV, $p = 0.0008$). Gini y Entropía rinden igual. |
| **2. CART Hiperparámetros** | `auto_mpg` | Árboles (6 variantes: profundidad, hojas, splits, GridSearch) vs Regresión Lineal | RMSE CV / R² | **Árbol regularizado** (depth=6, leaf=5, split=10; RMSE CV 3.286 vs 3.336), pero la diferencia con la Regresión Lineal **no es estadísticamente significativa** ($p = 0.42$). |
| **3. Ensamble Clasificador** | `breast_cancer` | Voting (Hard/Soft) y Stacking (meta-modelo LR) vs Modelos individuales (LR, SVM, KNN, Árbol, NB) | Accuracy CV / F1 | **El Ensamble supera ampliamente a los modelos débiles** y aporta mayor estabilidad ($\sigma = 0.010$), pero **empata estadísticamente** con el mejor individual conocido (LogReg/SVM 97.6% vs Ensamble 97.4%, $p = 0.72$). |
| **4. Bagging (Clasif. y Regresor)** | `breast_cancer` & `auto_mpg` | BaggingClassifier & BaggingRegressor (10 a 400 estimadores) vs Árbol individual sin poda | Acc CV, AUC, RMSE CV, OOB | **Bagging en ambos problemas de forma contundente**. Reduce la varianza drásticamente (+3 pp en accuracy, +0.07 AUC en clasificación; −27% de reducción en RMSE de regresión). |
| **5. Random Forest vs XGBoost** | `breast_cancer` & `auto_mpg` | RandomForest vs XGBoost (con ajuste equivalente vía `RandomizedSearchCV`) | Accuracy, Recall, AUC, RMSE | **Empate técnico/estadístico** ($p > 0.05$). XGBoost ofrece ligera ventaja en Recall (+1.4 pp) y AUC en clasificación; Random Forest ofrece mayor simplicidad y menor costo en datos limpios. |

---

## 🔬 Detalle de los Hallazgos por Actividad

### Actividad 1: Árboles de Decisión (Gini vs Entropía) vs Regresión Logística
- **Datasets:** `breast_cancer` de scikit-learn (569 muestras, 30 variables numéricas).
- **Resultados:**
  - Árbol Gini sin poda: Accuracy CV = 92.9%, AUC test = 0.916
  - Árbol Entropía sin poda: Accuracy CV = 93.7%, AUC test = 0.921
  - Árbol Gini (depth=4): Accuracy CV = 93.0%, AUC test = 0.934
  - Árbol Entropía (depth=4): Accuracy CV = 92.8%, AUC test = 0.963
  - **Regresión Logística (estandarizada):** **Accuracy CV = 97.6%, AUC test = 0.995**
- **¿Cuál es mejor?** La **Regresión Logística**. La prueba pareada de Wilcoxon sobre los 15 pliegues ($p \approx 0.0008$) confirma que su ventaja es estadísticamente significativa. Al ser un dataset con clases casi linealmente separables en $\mathbb{R}^{30}$, los hiperplanos continuos superan a las particiones ortogonales de un solo árbol. Entre Gini y Entropía no existen diferencias prácticas significativas.

---

### Actividad 2: Árboles de Decisión con Hiperparámetros vs Regresión Lineal
- **Dataset:** `auto_mpg` (Kaggle/UCI, 398 muestras, predicción de millas por galón `mpg`).
- **Resultados (RMSE en test / RMSE en CV 5×3):**
  - Regresión Lineal (baseline): RMSE test = 2.888, RMSE CV = 3.336 (R² CV = 0.815)
  - Árbol sin límites (`max_depth=None`): RMSE test = 3.368, RMSE CV = 3.771 (sobreajusta con R² train = 1.00)
  - Árbol subajustado (`max_depth=3`): RMSE test = 3.719, RMSE CV = 3.667
  - Árbol podado con hojas (`min_samples_leaf=10`): RMSE test = 2.746, RMSE CV = 3.286
  - **Árbol afinado (`depth=6, leaf=5, split=10`):** **RMSE test = 2.573, RMSE CV = 3.286 (R² CV = 0.819)**
  - Árbol GridSearchCV (`depth=6, leaf=10`): RMSE test = 2.712, RMSE CV = 3.290
- **¿Cuál es el mejor?** El **árbol bien regularizado** logra el menor error puntual, pero la prueba de Wilcoxon frente a la Regresión Lineal arroja $p \approx 0.42$, demostrando que **la diferencia no es estadísticamente significativa**. Un árbol sin podar es claramente peor.

---

### Actividad 3: Ensamble Clasificador (Voting y Stacking) vs Modelos Individuales
- **Dataset:** `breast_cancer`.
- **Modelos base heterogéneos:** Regresión Logística, SVM RBF, KNN, Árbol CART y Gaussian Naive Bayes.
- **Resultados (Accuracy CV 5×3):**
  - Individuales: LogReg (0.976), SVM (0.976), KNN (0.967), Naive Bayes (0.939), Árbol (0.930).
  - Ensambles: **Voting Hard (0.974)**, **Stacking (0.974)**, Voting Soft (0.963).
- **¿Cuál es mejor?** Depende de la referencia:
  - Frente a modelos individuales promedio o débiles (Árbol, Naive Bayes), el ensamble es **muy superior**.
  - Frente al mejor modelo individual conocido (Regresión Logística / SVM), el ensamble **empata estadísticamente** ($p \approx 0.72$). La ventaja del Stacking radica en su **menor variabilidad** ($\sigma = 0.010$ vs $0.013$), ofreciendo mayor robustez ante distribuciones imprevistas.

---

### Actividad 4: Bagging (Clasificador y Regresor) vs Modelo Individual
- **Clasificación (`breast_cancer`):**
  - Árbol individual sin poda: Accuracy CV = 92.9%, AUC CV = 0.922, Accuracy test = 91.2%
  - Bagging (10 árboles): Accuracy CV = 95.6%, AUC CV = 0.985
  - Bagging (100 árboles): Accuracy CV = 95.8%, AUC CV = 0.988, OOB Score = 0.956
  - **Bagging (300 árboles):** **Accuracy CV = 95.9%, AUC CV = 0.989, Accuracy test = 94.7%, OOB Score = 0.965**
  - Significancia: Wilcoxon $p \approx 0.0006$ (diferencia altamente significativa).
- **Regresión (`auto_mpg`):**
  - Árbol individual sin poda: RMSE CV = 3.771, RMSE test = 3.368
  - **Bagging Regressor (100 árboles):** **RMSE CV = 2.751, RMSE test = 2.144 (Reducción de ~27% en RMSE)**
- **¿Cuál es el mejor?** El **Bagging es contundentemente mejor** en ambos dominios. Al promediar árboles no correlacionados sobre muestras *bootstrap*, cancela la varianza sin aumentar el sesgo. La curva de convergencia demuestra estabilidad a partir de 50–100 árboles.

---

### Actividad 5: Random Forest vs XGBoost
- Ambos modelos optimizados con presupuesto de búsqueda idéntico (`RandomizedSearchCV`, 25 iteraciones, 5 folds sobre el conjunto de entrenamiento).
- **Clasificación (`breast_cancer`):**
  - Random Forest: Accuracy CV = 95.7%, AUC CV = 0.991, Accuracy test = 95.6%, Recall test = 97.2%
  - **XGBoost:** **Accuracy CV = 96.1%, AUC CV = 0.994, Accuracy test = 95.6%, Recall test = 98.6%**
- **Regresión (`auto_mpg`):**
  - **Random Forest:** **RMSE CV = 2.743 (R² CV = 0.875), RMSE test = 2.164**
  - XGBoost: RMSE CV = 2.753 (R² CV = 0.874), RMSE test = 2.341
- **¿Cuál es mejor?** **Empate estadístico** ($p = 0.33$ en clasificación, $p = 0.93$ en regresión).
  - Para clasificación médica (`breast_cancer`), **XGBoost** es preferible por su mayor Recall (minimiza falsos negativos en detección de tumores) y mayor AUC.
  - Para regresión tabular (`auto_mpg`), ambos ofrecen prácticamente el mismo desempeño, con ventaja de Random Forest en facilidad de ajuste y velocidad de entrenamiento multihilo.

---

## 🛠️ Estructura del Proyecto

```text
t3-am/
├── CART_Ensambles.ipynb           # Notebook interactivo con modelos, validación cruzada, métricas y gráficos
└── README.md                      # Documentación completa con respuestas, análisis y enlaces
```

---

## 👥 Integrantes
- *Estudiante / Pareja:* Ediluz
- *Curso:* Machine Learning
