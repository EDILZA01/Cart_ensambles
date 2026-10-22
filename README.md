# Modelos CART y Ensamblados

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EDILZA01/Cart_ensambles/blob/main/CART_Ensambles.ipynb)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-red.svg)](https://xgboost.readthedocs.io/)

Comparacion empirica y validacion estadistica entre arboles de decision CART y metodos de ensamble (Voting, Stacking, Bagging, Random Forest y XGBoost) frente a modelos lineales en problemas de clasificacion y regresion.

## Integrantes
- Estudiante: EdiLza Luz Carrasco Yupanqui

---

## Ejecucion en Google Colab

El notebook se puede abrir y ejecutar directamente en Google Colab:

- [Abrir CART_Ensambles.ipynb en Google Colab](https://colab.research.google.com/github/EDILZA01/Cart_ensambles/blob/main/CART_Ensambles.ipynb)
- Menu superior de Colab: **Entorno de ejecucion** > **Ejecutar todas**.

---

## Resumen de Resultados

Todos los modelos fueron entrenados desde cero (sin modelos preentrenados), evaluados en particion de prueba (Hold-Out 80/20) y validados mediante **Validacion Cruzada Repetida (5 pliegues x 3 repeticiones = 15 evaluaciones)** con pruebas estadisticas pareadas de **Wilcoxon** para determinar significancia real (p < 0.05).

| Seccion | Dataset | Modelos comparados | Metrica clave | Respuesta a "Cual es mejor" |
|---|---|---|---|---|
| **1. CART Gini vs Entropia** | `breast_cancer` | Arboles (Gini y Entropia con/sin poda) vs Regresion Logistica | Accuracy CV / AUC | **Regresion Logistica** (97.6% CV vs <=93.7% CV, p = 0.0008). Gini y Entropia presentan desempeno equivalente. |
| **2. CART Hiperparametros** | `auto_mpg` | Arboles (6 variantes: profundidad, hojas, divisiones, GridSearch) vs Regresion Lineal | RMSE CV / R2 | **Arbol regularizado** (depth=6, leaf=5, split=10; RMSE CV 3.286 vs 3.336), aunque la diferencia frente a la Regresion Lineal **no es estadisticamente significativa** (p = 0.42). |
| **3. Ensamble Clasificador** | `breast_cancer` | Voting (Hard/Soft) y Stacking (meta-modelo LR) vs Modelos individuales (LR, SVM, KNN, Arbol, NB) | Accuracy CV / F1 | **El Ensamble supera ampliamente a los modelos debiles** y reduce variabilidad (sigma = 0.010), pero **empata estadisticamente** con el mejor modelo individual (LogReg/SVM 97.6% vs Ensamble 97.4%, p = 0.72). |
| **4. Bagging (Clasificador y Regresor)** | `breast_cancer` & `auto_mpg` | BaggingClassifier & BaggingRegressor (10 a 400 estimadores) vs Arbol individual sin poda | Acc CV, AUC, RMSE CV, OOB | **Bagging es superior en ambos problemas de forma contundente**. Reduce la varianza drasticamente (+3 pp en accuracy, +0.07 AUC en clasificacion; -27% de reduccion en RMSE de regresion). |
| **5. Random Forest vs XGBoost** | `breast_cancer` & `auto_mpg` | RandomForest vs XGBoost (con ajuste equivalente via `RandomizedSearchCV`) | Accuracy, Recall, AUC, RMSE | **Empate estadistico** (p > 0.05). XGBoost presenta ventaja en Recall (+1.4 pp) y AUC en clasificacion; Random Forest ofrece menor costo computacional y rapidez de ajuste. |

---

## Detalle de Hallazgos

### 1. Arboles de Decision (Gini vs Entropia) vs Regresion Logistica
- **Dataset:** `breast_cancer` (569 muestras, 30 variables continuas).
- **Resultados:**
  - Arbol Gini sin poda: Accuracy CV = 92.9%, AUC test = 0.916
  - Arbol Entropia sin poda: Accuracy CV = 93.7%, AUC test = 0.921
  - Arbol Gini (depth=4): Accuracy CV = 93.0%, AUC test = 0.934
  - Arbol Entropia (depth=4): Accuracy CV = 92.8%, AUC test = 0.963
  - **Regresion Logistica (estandarizada):** Accuracy CV = 97.6%, AUC test = 0.995
- **Conclusion:** La **Regresion Logistica** es significativamente superior (prueba de Wilcoxon pareada sobre 15 pliegues, p = 0.0008). Al ser un espacio con fronteras continuas casi separables, los hiperplanos superan a las divisiones ortogonales de un arbol individual. Los criterios Gini y Entropia arrojan resultados equivalentes.

---

### 2. Arboles de Decision con Hiperparametros vs Regresion Lineal
- **Dataset:** `auto_mpg` (398 muestras, prediccion continua de consumo `mpg`).
- **Resultados (RMSE en test / RMSE en CV 5x3):**
  - Regresion Lineal (baseline): RMSE test = 2.888, RMSE CV = 3.336 (R2 CV = 0.815)
  - Arbol sin limites (`max_depth=None`): RMSE test = 3.368, RMSE CV = 3.771 (sobreajuste severo, R2 train = 1.00)
  - Arbol subajustado (`max_depth=3`): RMSE test = 3.719, RMSE CV = 3.667
  - Arbol podado con hojas (`min_samples_leaf=10`): RMSE test = 2.746, RMSE CV = 3.286
  - **Arbol optimizado (`depth=6, leaf=5, split=10`):** RMSE test = 2.573, RMSE CV = 3.286 (R2 CV = 0.819)
  - Arbol via GridSearchCV (`depth=6, leaf=10`): RMSE test = 2.712, RMSE CV = 3.290
- **Conclusion:** El **arbol regularizado** obtiene el menor error absoluto, pero la comparacion frente a la Regresion Lineal tiene p = 0.42, lo cual indica que la diferencia **no es estadisticamente representativa**. El arbol sin podar deteriora sensiblemente la capacidad de generalizacion.

---

### 3. Ensamble Clasificador (Voting y Stacking) vs Modelos Individuales
- **Dataset:** `breast_cancer`.
- **Modelos base:** Regresion Logistica, SVM RBF, KNN, Arbol CART y Gaussian Naive Bayes.
- **Resultados (Accuracy CV 5x3):**
  - Modelos individuales: LogReg (0.976), SVM (0.976), KNN (0.967), Naive Bayes (0.939), Arbol (0.930).
  - Ensambles: Voting Hard (0.974), Stacking (0.974), Voting Soft (0.963).
- **Conclusion:** 
  - Respecto a modelos individuales de menor precision (Arbol, Naive Bayes), los ensambles son claramente superiores.
  - Frente al mejor clasificador individual (Regresion Logistica / SVM), el ensamble **empata en desempeno** (p = 0.72). La ventaja principal de Stacking radica en una **menor dispersion de error** (sigma = 0.010 vs 0.013), incrementando la robustez.

---

### 4. Bagging (Clasificador y Regresor) vs Arbol Individual
- **Clasificacion (`breast_cancer`):**
  - Arbol individual sin poda: Accuracy CV = 92.9%, AUC CV = 0.922, Accuracy test = 91.2%
  - Bagging (10 estimadores): Accuracy CV = 95.6%, AUC CV = 0.985
  - Bagging (100 estimadores): Accuracy CV = 95.8%, AUC CV = 0.988, OOB Score = 0.956
  - **Bagging (300 estimadores):** Accuracy CV = 95.9%, AUC CV = 0.989, Accuracy test = 94.7%, OOB Score = 0.965
  - Prueba de significancia: Wilcoxon p = 0.0006.
- **Regresion (`auto_mpg`):**
  - Arbol individual sin poda: RMSE CV = 3.771, RMSE test = 3.368
  - **Bagging Regressor (100 estimadores):** RMSE CV = 2.751, RMSE test = 2.144 (reduccion de ~27% en error cuadratico medio)
- **Conclusion:** **Bagging es superior de forma contundente**. El remuestreo bootstrap con agregacion reduce la varianza sin sesgar el estimador, estabilizando la prediccion a partir de 50 a 100 estimadores.

---

### 5. Random Forest vs XGBoost
- Ambos modelos ajustados con espacio de busqueda balanceado (`RandomizedSearchCV`, 25 iteraciones, 5 pliegues).
- **Clasificacion (`breast_cancer`):**
  - Random Forest: Accuracy CV = 95.7%, AUC CV = 0.991, Accuracy test = 95.6%, Recall test = 97.2%
  - **XGBoost:** Accuracy CV = 96.1%, AUC CV = 0.994, Accuracy test = 95.6%, Recall test = 98.6%
- **Regresion (`auto_mpg`):**
  - Random Forest: RMSE CV = 2.743 (R2 CV = 0.875), RMSE test = 2.164
  - XGBoost: RMSE CV = 2.753 (R2 CV = 0.874), RMSE test = 2.341
- **Conclusion:** **Empate estadistico** (p = 0.33 en clasificacion, p = 0.93 en regresion).
  - En clasificacion clinica, **XGBoost** ofrece mejor sensibilidad (Recall), disminuyendo falsos negativos criticos.
  - En regresion sobre datos tabulares continuos, **Random Forest** ofrece desempeno equivalente con menor sensibilidad a la calibracion fina de hiperparametros.

---

## Estructura del Repositorio

```text
Cart_ensambles/
├── CART_Ensambles.ipynb     # Notebook con implementacion, validacion cruzada y graficos
└── README.md                # Documentacion tecnica de resultados y analisis comparativo
```