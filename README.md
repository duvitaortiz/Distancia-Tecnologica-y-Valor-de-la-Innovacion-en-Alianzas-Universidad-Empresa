# NB2 – Diversificación Tecnológica y Valor de la Innovación

Análisis de la relación entre diversificación tecnológica (`DeTech`) y el valor incremental de la innovación (`ValorInc`), utilizando modelos binomiales negativos (NB2) con variables de control (`Experticia`, `Inventores`, `Rank`, `PatenUniv`, `PatenInd`).

## Contenido
- Detección y exclusión de outliers.
- Transformación logarítmica de `DeTech`.
- Segmentación por prestigio universitario (`Rank`).
- Interacciones `Rank × DeTech` y `Experticia × DeTech`.
- Generación de 10 gráficas de diagnóstico y resultados.

## Datos
- Archivo: `Datos_2026_1.xlsx`, hoja `Matriz 1_`.
- Variables clave: `DeTech`, `ValorInc`, `Rank`, `Experticia`, `Inventores`, `PatenUniv`, `PatenInd`.

## Metodología
- Modelo NB2 con enlace logarítmico (`loglike_method='nb2'`).
- Estimación por máxima verosimilitud (BFGS, 5000 iteraciones).
- Errores estándar robustos.
- Variables centradas para reducir multicolinealidad.

## Resultados principales
- Tabla comparativa de modelos (base vs moderado, por segmentos de Rank).
- Punto de inflexión de la U invertida (si aplica).
- 10 figuras: distribuciones, correlaciones, curvas predichas por nivel del moderador (stock de patentes del equipo), efecto marginal, diagnóstico de residuos, panel resumen.

## Requisitos
```bash
pip install pandas numpy statsmodels matplotlib seaborn scipy openpyxl

