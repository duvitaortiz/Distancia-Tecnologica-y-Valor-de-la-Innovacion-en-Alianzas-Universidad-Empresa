# Instalar si falta (en Colab)
!pip -q install openpyxl

import os, glob, re, math
import numpy as np
import pandas as pd

# Carpeta con los archivos comparacion_*.xlsx
comp_dir = "/content/comparaciones"
assert os.path.isdir(comp_dir), f"No existe la carpeta: {comp_dir}"

def _extract_index(path):
    m = re.search(r"comparacion_(\d+)\.xlsx$", os.path.basename(path), re.IGNORECASE)
    return int(m.group(1)) if m else 10**9

results = []

paths = sorted(glob.glob(os.path.join(comp_dir, "comparacion_*.xlsx")), key=_extract_index)

for p in paths:
    nombre = os.path.basename(p)
    try:
        df = pd.read_excel(p, header=0)  # la primera fila son encabezados y se ignora como datos
    except Exception as e:
        print(f"⚠️ No se pudo leer {nombre}: {e}. Se omite.")
        continue

    # Comprobar columnas esperadas
    if not {"Total1", "Total2"}.issubset(df.columns):
        print(f"⚠️ {nombre} no tiene las columnas 'Total1' y 'Total2'. Se omite.")
        continue

    # Convertir a arrays numéricos (NaN -> 0)
    x = pd.to_numeric(df["Total1"], errors="coerce").fillna(0).to_numpy(dtype=float)
    y = pd.to_numeric(df["Total2"], errors="coerce").fillna(0).to_numpy(dtype=float)

    # Producto punto y normas
    dot = float(np.dot(x, y))
    normx = math.sqrt(float(np.dot(x, x)))
    normy = math.sqrt(float(np.dot(y, y)))

    # Casos borde
    if normx == 0.0 and normy == 0.0:
        detech = 0.0   # ambos vectores vacíos/ceros -> idénticos
    elif normx == 0.0 or normy == 0.0:
        detech = 1.0   # uno vacío -> máxima distancia
    else:
        cos = dot / (normx * normy)
        # Asegurar rango [0,1] (contadores no negativos implican cos>=0, pero por seguridad)
        cos = max(0.0, min(1.0, cos))
        detech = 1.0 - cos

    # Guardar resultado con 6 decimales
    results.append({
        "comparacion": nombre,
        "DeTech": round(detech, 6)
    })

# Crear DataFrame de salida y guardar
df_out = pd.DataFrame(results)
out_path = "/content/detech_resultados.xlsx"
df_out.to_excel(out_path, index=False)

print("✅ Cálculo completado. Resultados guardados en:", out_path)
