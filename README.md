# Scripts Auxiliares y Herramientas de Complemento

Esta rama contiene una colección de herramientas especializadas y scripts de procesamiento de datos desarrollados para dar soporte a las ramas principales del proyecto. Aquí se agrupan utilidades para limpieza de bases de datos, filtrado de rankings y cálculos métricos específicos.

##  Índice de Herramientas

### 1. Comparación de Perfiles Tecnológicos (DeTech)
Script avanzado para medir la proximidad tecnológica entre conjuntos de datos.
*   **Funcionalidad:** Extrae prefijos de códigos tecnológicos (Columna G) de archivos Excel particionados.
*   **Métrica:** Calcula la distancia tecnológica mediante la fórmula:  
    $$DeTech_{ij} = 1 - \cos(X_i, X_j)$$
*   **Salida:** Genera comparativas individuales y un maestro consolidado (`detech_resumen.xlsx`).

### 2. Limpieza de Registros Duplicados (Cruce de Bases)
Herramienta de depuración para evitar redundancia entre bases de datos de distinto tamaño.
*   **Lógica:** Toma una "Base Grande" y elimina automáticamente cualquier registro (por título) que ya se encuentre presente en una "Base Pequeña".
*   **Uso común:** Consolidación de bases de datos tras procesos de recolección manual.

### 3. Filtro de Prestigio Universitario (Top 20 por Año)
Script de validación histórica de instituciones.
*   **Funcionalidad:** Cruza una base de datos con nombres de universidades y fechas frente a una base de referencia de rankings.
*   **Criterio:** Solo conserva los registros donde la universidad pertenecía al **Top 20** en el año específico del registro. Los que no cumplen con este estándar de prestigio para esa fecha son eliminados.

### 4. Extractor de Métricas de Inventores
Analizador de colaboración en patentes.
*   **Funcionalidad:** Procesa celdas que contienen múltiples nombres separados por punto y coma (`;`).
*   **Resultado:** Devuelve el conteo exacto de inventores por cada entrada, facilitando el análisis de densidad de colaboración.

---

## ⚙️ Requisitos del Entorno
La mayoría de estos scripts están diseñados para ejecutarse en entornos Python 3.x con las siguientes librerías:
*   `pandas` (Procesamiento de datos)
*   `openpyxl` / `xlrd` (Lectura de Excel)
*   `numpy` / `scipy` (Cálculos matemáticos y similitud de coseno)


