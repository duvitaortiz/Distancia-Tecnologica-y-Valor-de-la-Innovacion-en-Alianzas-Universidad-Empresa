

---

## Hipótesis 2 — Efecto de la Experticia sobre el Valor de la Innovación Conjunta)


# OLS – Stock de Patentes de Inventores vs Valor de la Innovación

Modelos de regresión lineal (OLS) para evaluar el efecto del stock acumulado de patentes de los inventores (`Total_Patentes_Inventores`) sobre `ValorInc`, incluyendo controles (`DeTech`, `Experticia`, `Inventores`, `Rank`, `PatenUniv`, `PatenInd`).

## Contenido
- Cuatro especificaciones: lineal, log‑log, polinómica original, polinómica log‑log.
- Winsorización al 1% en colas.
- Errores estándar robustos HC3.
- Diagnósticos: Breusch‑Pagan, White, Durbin‑Watson, RESET, Jarque‑Bera.
- Generación de 5 figuras (distribuciones, curva predicha, residuos, barras por decil, panel resumen).

## Datos
- Mismo archivo `Datos_2026.xlsx`, hoja `Matriz 1_`.
- La variable `Total_Patentes_Inventores` se calcula sumando las patentes de los miembros del equipo.

## Metodología
- Transformación logarítmica de la variable dependiente e independiente principal.
- Términos polinómicos centrados para evitar multicolinealidad.
- Comparación de AIC/BIC entre modelos.

## Resultados principales
- Coeficientes significativos y punto de inflexión (U invertida) en el modelo log‑log polinómico.
- Tabla comparativa de bondad de ajuste.
- 5 figuras guardadas en `graficas_patentes_con_controles/`.

## Requisitos
```bash
pip install pandas numpy statsmodels matplotlib seaborn scipy openpyxl
