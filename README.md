# Hipótesis 2 — Efecto de la Experticia sobre el Valor de la Innovación Conjunta

Esta rama contiene el desarrollo completo de la **Hipótesis 2** de la investigación:

> **La experticia de los colaboradores de la alianza universidad–empresa tiene un efecto en el valor de la innovación conjunta.**

El análisis busca determinar cómo la acumulación de experiencia tecnológica y conocimiento previo influye sobre el desempeño innovador de las alianzas universidad–empresa.

---

# Objetivo de la hipótesis

Evaluar el efecto de la experticia acumulada sobre el valor de la innovación conjunta utilizando modelos econométricos lineales y no lineales.

---

# Pregunta de investigación

¿El aumento del conocimiento tecnológico acumulado y la experiencia previa de los colaboradores mejora el valor de la innovación o existe un punto de saturación donde el efecto comienza a disminuir?

---

# Variables utilizadas

| Variable | Descripción |
|---|---|
| ValorInc_w | Valor incremental de innovación (winsorizado) |
| TPI_w | Stock acumulado de patentes |
| log_V | Transformación logarítmica de ValorInc |
| log_P | Transformación logarítmica de TPI |
| TPI_c | Variable TPI centrada |
| TPI_c² | Término cuadrático |
| log_P_c | Variable log centrada |
| log_P_c² | Término cuadrático logarítmico |

---

# Preparación y procesamiento de datos

Antes de estimar los modelos econométricos se realizaron diferentes procesos de limpieza y preparación de datos:

- Eliminación de inconsistencias.
- Winsorización al 1% en ambas colas.
- Transformaciones logarítmicas.
- Centrado de variables.
- Validación de distribución.
- Revisión de valores extremos.

---
