## Hipótesis 3 — Efecto Moderador de la Experticia en la Relación entre Distancia Tecnológica e Innovación



# NB2 – Moderación del Stock de Patentes del Equipo en la Relación DeTech-ValorInc

Modelos binomiales negativos para analizar si el stock acumulado de patentes del equipo (`SumPatents`) modera la relación entre diversificación tecnológica (`DeTech`) y valor incremental (`ValorInc`), controlando por `Experticia`, `Inventores`, `Rank`, `PatenUniv`, `PatenInd`.

## Contenido
- Modelo base (sin moderador) y modelo moderado (con `SumLog`, `SumLog²` e interacción `DeTech×SumLog`).
- Segmentación por `Rank` (prestigio universitario).
- Comparación de AIC/BIC entre modelos base y moderados.
- Efecto marginal del moderador sobre la pendiente de `DeTech`.
- 10 gráficas: descriptivas, correlaciones, curvas por nivel de moderador, comparativas, diagnóstico, panel resumen.

## Datos
- Archivo `Datos_2026.xlsx`, hoja `Matriz 1_`.
- `SumPatents` = suma de patentes de todos los inventores del equipo.

## Metodología
- NB2 con variables centradas (`DeTech_c`, `SumLog_c`, etc.).
- Transformación logarítmica del moderador (`SumLog = log1p(SumPatents)`).
- Interacción multiplicativa entre `DeTech_c` y `SumLog_c`.
- Estimación por máxima verosimilitud (BFGS, 5000 iteraciones).

## Resultados principales
- Tabla comparativa de coeficientes (modelo base, modelo moderado completo, por Rank).
- Punto de cruce del efecto marginal (si `b_interacción ≠ 0`).
- 10 figuras de alta calidad, listas para publicación.

## Requisitos
```bash
pip install pandas numpy statsmodels matplotlib seaborn scipy openpyxl
---
