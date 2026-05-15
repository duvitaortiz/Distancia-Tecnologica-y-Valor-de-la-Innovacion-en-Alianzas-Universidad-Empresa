# ============================================
# LIMPIEZA Y FILTRADO DE BASES DE DATOS
# ============================================

import pandas as pd

# -----------------------------
# 1. RUTAS DE LOS ARCHIVOS
# -----------------------------
ruta_grande = "/content/ruta1.xlsx"
ruta_pequena = "/content/ruta2.xlsx"

# -----------------------------
# 2. LEER LOS ARCHIVOS
# -----------------------------
df_grande = pd.read_excel(ruta_grande)
df_pequena = pd.read_excel(ruta_pequena)

# -----------------------------
# 3. TOMAR LOS TITULOS
#    Grande -> columna G
#    Pequeña -> columna H
# -----------------------------

# Índices:
# G = 6
# H = 7

titulos_grande = df_grande.iloc[:, 6].astype(str).str.strip().str.lower()
titulos_pequena = df_pequena.iloc[:, 7].astype(str).str.strip().str.lower()

# -----------------------------
# 4. ELIMINAR DE LA BASE GRANDE
#    LOS TITULOS QUE ESTEN EN
#    LA BASE PEQUEÑA
# -----------------------------

# Crear conjunto para búsqueda rápida
titulos_pequena_set = set(titulos_pequena)

# Filtrar
df_filtrado = df_grande[
    ~titulos_grande.isin(titulos_pequena_set)
].copy()

print("Registros después de eliminar coincidencias:",
      len(df_filtrado))

# -----------------------------
# 5. FILTRAR COLUMNA J
#    SOLO DEJAR FILAS QUE
#    CONTENGAN "university"
# -----------------------------

# J = índice 9
columna_j = df_filtrado.iloc[:, 9].astype(str)

df_final = df_filtrado[
    columna_j.str.contains("university", case=False, na=False)
].copy()

print("Registros después del filtro 'university':",
      len(df_final))

# -----------------------------
# 6. GUARDAR RESULTADO
# -----------------------------

ruta_salida = "/content/ruta_Final_Filtrada.xlsx"

df_final.to_excel(ruta_salida, index=False)

print("\nArchivo guardado en:")
print(ruta_salida)
