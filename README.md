# Hipótesis 1 — Distancia Tecnológica y Valor de la Innovación Conjunta

Esta rama contiene el desarrollo completo de la **Hipótesis 1** de la investigación:

> : La distancia tecnológica tiene un efecto en forma de U invertida en el valor de la innovación conjunta Universidad–Empresa.

El análisis busca determinar si la diversidad tecnológica entre los colaboradores favorece o limita la generación de innovación de alto valor.

---

# Objetivo

Evaluar el efecto de la distancia tecnológica (DeTech) sobre el valor incremental de la innovación conjunta universidad–empresa mediante modelos econométricos de conteo.

---

# Pregunta de investigación

¿La distancia tecnológica entre universidad y empresa mejora el valor de la innovación o existe un punto óptimo después del cual la colaboración pierde efectividad?

---

# Variables utilizadas

| Variable | Descripción |
|---|---|
| ValorInc | Valor incremental de innovación |
| DeTech | Distancia tecnológica |
| DeTech² | Término cuadrático de distancia tecnológica |
| Rank | Indicador de universidad prestigiosa |
| Experticia | Experiencia acumulada de colaboradores |
| Inventores | Número de inventores |
| alpha | Parámetro de dispersión del modelo NB2 |

---

# Metodología

Debido a la naturaleza discreta y sobredispersa de la variable dependiente, se utilizaron modelos:

- Binomial Negativa (NB2)
- Modelos polinomiales
- Segmentación por prestigio universitario
- Análisis de interacción
- Pruebas de robustez

