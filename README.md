# Distancia Tecnológica y Valor de la Innovación en Alianzas Universidad-Empresa

Repositorio correspondiente a la tesis:

> **“Experticia de los colaboradores de la alianza universidad–empresa y su efecto moderador en la relación entre la distancia tecnológica y el valor de la innovación conjunta.”**

Este proyecto analiza cómo la distancia tecnológica entre universidades y empresas afecta el valor de la innovación generada conjuntamente, incorporando además el efecto moderador de la experticia de los colaboradores y del stock acumulado de conocimiento tecnológico.

---

# Objetivo de la investigación

Evaluar la relación entre la distancia tecnológica y el valor de la innovación conjunta universidad–empresa, considerando:

- El efecto no lineal de la distancia tecnológica.
- El impacto del stock acumulado de patentes.
- El efecto moderador de la experticia y experiencia tecnológica.
- Diferencias entre universidades prestigiosas y no prestigiosas.

---

# Contenido del repositorio

El repositorio está organizado en diferentes ramas y módulos que contienen:

## Hipótesis 1 — La distancia tecnológica tiene un efecto en forma de U invertida en el valor de la innovación conjunta Universidad–Empresa

Análisis de la relación entre la distancia tecnológica (DeTech) y el valor de la innovación conjunta mediante modelos Binomial Negativa (NB2), incluyendo:

- Modelos polinomiales.
- Análisis de no linealidad.
- Segmentación por prestigio universitario.
- Pruebas de robustez.
- Diagnósticos estadísticos.

---

## Hipótesis 2 — La experticia de los colaboradores de la alianza universidad–empresa tiene un efecto en el valor de la innovación conjunta. 

Evaluación del efecto del stock acumulado de patentes (TPI) sobre el valor de la innovación utilizando:

- Modelos OLS.
- Transformaciones Log-Log.
- Modelos polinomiales.
- Diagnósticos estadísticos.
- Visualizaciones comparativas.

---

## Hipótesis 3 — La experticia de los colaboradores de la alianza universidad–empresa modera la relación entre la distancia tecnológica y el valor de la innovación conjunta.

Modelos de moderación para evaluar cómo la experticia y el stock tecnológico modifican la relación entre distancia tecnológica e innovación:

- Modelos NB2 moderados.
- Interacciones cuadráticas.
- Segmentación Rank=1 vs Rank=0.
- Comparación AIC/BIC.
- Efectos marginales.

---

## Procesamiento y construcción de datos

Incluye scripts para:

- Limpieza y filtrado de bases de datos.
- Procesamiento de archivos Excel.
- Eliminación de duplicados.
- Identificación de universidades prestigiosas.
- Conteo automático de inventores.
- Comparación de perfiles tecnológicos.
- Cálculo de distancia tecnológica (DeTech).

---

# Metodología utilizada

## Variables principales

| Variable | Descripción |
|---|---|
| DeTech | Distancia tecnológica entre universidad y empresa |
| ValorInc | Valor incremental de innovación |
| TPI | Stock acumulado de patentes |
| Experticia | Experiencia acumulada de colaboradores |
| Rank | Indicador de universidad prestigiosa |
| Inventores | Número de inventores asociados |

---

## Métodos estadísticos

Se utilizaron diferentes enfoques econométricos:

- Regresión OLS.
- Modelos Binomial Negativa NB2.
- Modelos polinomiales.
- Transformaciones Log-Log.
- Modelos moderados.
- Errores robustos HC3.
- Comparaciones AIC/BIC.
- Pseudo-R².
- Pruebas RESET, White y Breusch-Pagan.

---

# Tecnologías utilizadas

## Lenguaje y librerías

- Python
- Pandas
- NumPy
- Statsmodels
- Scikit-learn
- Matplotlib
- Seaborn
- SciPy
- OpenPyXL
