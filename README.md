# Hipótesis 3 — Efecto Moderador de la Experticia en la Relación entre Distancia Tecnológica e Innovación

Esta rama contiene el desarrollo de la **Hipótesis 3** de la investigación:

> **La experticia de los colaboradores de la alianza universidad–empresa modera la relación entre la distancia tecnológica y el valor de la innovación conjunta.**

El análisis evalúa cómo la acumulación de conocimiento tecnológico y experiencia modifica el efecto de la distancia tecnológica (DeTech) sobre el desempeño innovador de las colaboraciones universidad–empresa.

---

# Objetivo

Determinar si la experticia acumulada fortalece o debilita la relación entre distancia tecnológica y valor de la innovación conjunta.

---

# Pregunta de investigación

¿La experiencia tecnológica acumulada permite aprovechar mejor la diversidad tecnológica o reduce el impacto positivo de la colaboración?

---

# Variables utilizadas

| Variable | Descripción |
|---|---|
| ValorInc | Valor incremental de innovación |
| DeTech_c | Distancia tecnológica centrada |
| DeTech_c_sq | Término cuadrático |
| SumLog_c | Stock tecnológico acumulado centrado |
| SumLog_c_sq | Término cuadrático del stock |
| Interaction | Interacción DeTech × SumLog |
| Experticia | Experiencia acumulada |
| Inventores | Número de inventores |
| Rank | Indicador de universidad prestigiosa |

---

# Metodología

Se utilizaron modelos:

- Binomial Negativa (NB2)
- Modelos polinomiales moderados
- Interacciones cuadráticas
- Segmentación institucional
- Comparación AIC/BIC
- Máxima verosimilitud (MLE)

Las variables fueron centradas respecto a su media para reducir multicolinealidad.

---

# Segmentación de análisis

La muestra se dividió en:

| Segmento | Descripción |
|---|---|
| Completa | Total de observaciones |
| Rank = 1 | Universidades prestigiosas |
| Rank = 0 | Universidades no prestigiosas |

---
