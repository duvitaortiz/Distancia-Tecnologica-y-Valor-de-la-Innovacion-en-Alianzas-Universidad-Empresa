import zipfile
import pandas as pd
import os
import re

# Ruta al archivo ZIP original
zip_path = '/content/perfiles 2.zip'

# Directorio donde se extraerán los archivos
extract_dir = 'extracted_files'
os.makedirs(extract_dir, exist_ok=True)

# Descomprimir el archivo ZIP
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# Lista para almacenar los archivos renombrados y ordenados
file_list = []

# Recorrer los archivos extraídos
for root, dirs, files in os.walk(extract_dir):
    for file in files:
        if file.endswith('.xlsx') or file.endswith('.xls'):
            file_path = os.path.join(root, file)

            # Leer el archivo Excel
            df = pd.read_excel(file_path)

            # Ignorar la primera fila y ordenar la primera columna alfabéticamente
            df_sorted = df.iloc[1:].sort_values(by=df.columns[0])

            # Generar nuevo nombre eliminando "resultados_"
            new_name = re.sub(r'^resultados_', '', file)
            new_path = os.path.join(root, new_name)

            # Guardar el archivo con el nuevo nombre
            df_sorted.to_excel(new_path, index=False)

            # Agregar el archivo a la lista con su número extraído para ordenarlo después
            match = re.search(r'\d+', new_name)  # Extrae el número del nombre del archivo
            file_number = int(match.group()) if match else float('inf')  # Si no tiene número, lo coloca al final
            file_list.append((file_number, new_path))

            # Eliminar el archivo original si su nombre cambió
            if file_path != new_path:
                os.remove(file_path)

            print(f"Archivo {new_name} ordenado y guardado.")

# Ordenar los archivos por número antes de comprimirlos
file_list.sort()

# Comprimir los archivos organizados en un nuevo ZIP
new_zip_path = 'perfiles_organizado_2.zip'
with zipfile.ZipFile(new_zip_path, 'w') as new_zip:
    for _, file_path in file_list:
        arcname = os.path.relpath(file_path, extract_dir)
        new_zip.write(file_path, arcname)

print(f"Archivos organizados comprimidos en {new_zip_path}.")
