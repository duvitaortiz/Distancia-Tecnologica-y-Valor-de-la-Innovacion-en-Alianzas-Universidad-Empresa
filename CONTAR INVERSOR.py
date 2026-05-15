import pandas as pd

# Ruta del archivo
ruta = "/content/nombrearchivo.xlsx"

# Leer el archivo Excel
df = pd.read_excel(ruta)

# Contar inventores en la columna 'inventores'
# Separados por ';' y empezando desde la fila 2
df["# de inventores"] = (
    df["Inventors"]
    .fillna("")  # evitar errores con celdas vacías
    .apply(lambda x: len([i for i in str(x).split(";") if i.strip() != ""]))
)

# Guardar el nuevo archivo
salida = "/content/Base_Top20_Universidades_Conteo.xlsx"
df.to_excel(salida, index=False)

print("Archivo guardado en:")
print(salida)
