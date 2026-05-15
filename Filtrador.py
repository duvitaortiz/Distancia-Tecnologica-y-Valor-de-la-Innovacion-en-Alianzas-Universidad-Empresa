

import pandas as pd
import re

# ----------------------------------------------------------
# 1. RUTAS
# ----------------------------------------------------------

ruta_base = "/content/ruta1.xlsx"
ruta_ranking = "/content/ruta2.xlsx"

# ----------------------------------------------------------
# 2. LEER ARCHIVOS
# ----------------------------------------------------------

df = pd.read_excel(ruta_base)
ranking_df = pd.read_excel(ruta_ranking)

# ----------------------------------------------------------
# 3. EXTRAER EL AÑO DE LA COLUMNA D
#    FORMATO: 15.10.2013
# ----------------------------------------------------------

# Columna D = índice 3
df["AÑO_EXTRAIDO"] = (
    df.iloc[:, 3]
    .astype(str)
    .str.extract(r'(\d{4})')[0]
)

# Convertir a número
df["AÑO_EXTRAIDO"] = pd.to_numeric(
    df["AÑO_EXTRAIDO"],
    errors="coerce"
)

# ----------------------------------------------------------
# 4. FUNCIÓN PARA LIMPIAR NOMBRES
# ----------------------------------------------------------

def limpiar_universidad(nombre):

    if pd.isna(nombre):
        return ""

    nombre = str(nombre).upper()

    # Quitar símbolos y puntuación
    nombre = re.sub(r'[^A-Z0-9\s]', ' ', nombre)

    # Expresiones comunes que sobran
    eliminar = [
        "THE TRUSTEES OF",
        "TRUSTEES OF",
        "BOARD OF REGENTS",
        "REGENTS OF",
        "THE REGENTS OF",
        "THE BOARD OF REGENTS",
        "INC",
        "CORPORATION",
        "SYSTEM",
    ]

    for e in eliminar:
        nombre = nombre.replace(e, " ")

    # Espacios dobles
    nombre = re.sub(r'\s+', ' ', nombre).strip()

    return nombre

# ----------------------------------------------------------
# 5. CREAR DICCIONARIO:
#    AÑO -> UNIVERSIDADES TOP 20
# ----------------------------------------------------------

ranking_por_año = {}

# Recorrer columnas de dos en dos:
# A-B, C-D, E-F ...

for i in range(0, ranking_df.shape[1], 2):

    try:
        col_año = ranking_df.iloc[:, i]
        col_univ = ranking_df.iloc[:, i + 1]

        # Tomar el año
        año = pd.to_numeric(col_año.dropna().iloc[0], errors="coerce")

        if pd.isna(año):
            continue

        año = int(año)

        universidades = set()

        for u in col_univ.dropna():

            nombre_limpio = limpiar_universidad(u)

            universidades.add(nombre_limpio)

        ranking_por_año[año] = universidades

    except:
        continue

# ----------------------------------------------------------
# 6. BUSCAR LA UNIVERSIDAD EN EL TOP 20
# ----------------------------------------------------------

# IMPORTANTE:
# Cambia este índice si la universidad
# NO está en la columna J

# Actualmente:
# J = índice 9

columna_universidad = 9

def pertenece_top20(fila):

    año = fila["AÑO_EXTRAIDO"]

    if pd.isna(año):
        return False

    año = int(año)

    # Si el año no existe en ranking
    if año not in ranking_por_año:
        return False

    universidad = fila.iloc[columna_universidad]

    universidad_limpia = limpiar_universidad(universidad)

    top20 = ranking_por_año[año]

    # Coincidencia parcial inteligente
    for uni_top in top20:

        if (
            universidad_limpia in uni_top
            or uni_top in universidad_limpia
        ):
            return True

    return False

# ----------------------------------------------------------
# 7. FILTRAR SOLO TOP 20
# ----------------------------------------------------------

df_top20 = df[
    df.apply(pertenece_top20, axis=1)
].copy()

# ----------------------------------------------------------
# 8. GUARDAR RESULTADO
# ----------------------------------------------------------

ruta_salida = "/content/rutaguarda.xlsx"

df_top20.to_excel(ruta_salida, index=False)

# ----------------------------------------------------------
# 9. RESULTADOS
# ----------------------------------------------------------

print("Total registros originales:", len(df))
print("Total registros TOP 20:", len(df_top20))

print("\nArchivo guardado en:")
print(ruta_salida)
