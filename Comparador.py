# === 0) Dependencias e importaciones ===
!pip -q install xlrd openpyxl

import os, re, glob, zipfile, shutil
import pandas as pd

# === 1) Montar Google Drive ===
from google.colab import drive
drive.mount('/content/drive')

# === 2) Rutas a los ZIP en tu Drive (AJUSTA ESTAS DOS) ===
zip1_gdrive_path = "/content/drive/MyDrive/Perfil_1.zip"
zip2_gdrive_path = "/content/drive/MyDrive/perfil_2.zip"

assert os.path.exists(zip1_gdrive_path), f"No encuentro: {zip1_gdrive_path}"
assert os.path.exists(zip2_gdrive_path), f"No encuentro: {zip2_gdrive_path}"

# === 3) Extraer los ZIP a carpetas temporales ===
extract_root = "/content/extract"
extract1 = os.path.join(extract_root, "perfil_1")
extract2 = os.path.join(extract_root, "perfil_2")
os.makedirs(extract1, exist_ok=True)
os.makedirs(extract2, exist_ok=True)

with zipfile.ZipFile(zip1_gdrive_path, 'r') as z:
    z.extractall(extract1)
with zipfile.ZipFile(zip2_gdrive_path, 'r') as z:
    z.extractall(extract2)

print("✅ ZIPs extraídos en:", extract1, "y", extract2)

# === 4) Funciones utilitarias ===
def build_index_map(root_dir):
    """
    Crea un diccionario {indice:int -> path_del_archivo} buscando recursivamente .xls/.xlsx
    Elige el primer archivo encontrado para cada índice "N_" en el nombre.
    """
    idx_map = {}
    for path in glob.glob(os.path.join(root_dir, "**", "*.xls*"), recursive=True):
        base = os.path.basename(path)
        m = re.match(r"^(\d+)_", base)  # ejemplo: 123_2002_University_Ori.xls
        if m:
            idx = int(m.group(1))
            if idx not in idx_map:  # toma el primero
                idx_map[idx] = path
    return idx_map

def leer_columna_g(path_xls):
    """
    Devuelve una Serie (str) con los valores de la columna G.
    Intenta varias estrategias para soportar .xls y .xlsx con distintos formatos.
    """
    intentos = [
        dict(usecols=[6], header=None, dtype=str),  # por índice (G = 6)
        dict(usecols="G", header=None, dtype=str),  # por letra
        dict(header=None, dtype=str),               # todo y luego tomo la 7ª
    ]
    for kwargs in intentos:
        try:
            df = pd.read_excel(path_xls, **kwargs)
            if df.shape[1] == 0:
                continue
            if "usecols" in kwargs:
                col = df.iloc[:, 0]
            else:
                if df.shape[1] <= 6:
                    continue
                col = df.iloc[:, 6]
            return col.dropna().astype(str)
        except Exception:
            continue
    # Si no se pudo leer, retorna serie vacía
    return pd.Series([], dtype=str)

_prefijo_regex = re.compile(r"([A-Za-z]\d{2})")

def extraer_prefijos_de_texto(texto):
    """
    De un string como 'C12N 5/08; A61L 27/24' devuelve ['c12','a61'].
    - Separa por ';' o saltos de línea.
    - Toma prefijo letra+2 dígitos desde el inicio de cada código.
    - Fallback: primeros 3 alfanuméricos si no matchea.
    """
    prefijos = []
    for parte in re.split(r"[;\n]+", texto):
        parte = parte.strip()
        if not parte:
            continue
        m = _prefijo_regex.match(parte)
        if m:
            prefijos.append(m.group(1).lower())
        else:
            alnum = re.sub(r"[^A-Za-z0-9]", "", parte)
            if len(alnum) >= 3:
                prefijos.append(alnum[:3].lower())
    return prefijos

def contar_prefijos_en_archivo(path_xls):
    """
    Lee la columna G y cuenta los prefijos (en minúscula) en toda la columna.
    Ignora el prefijo 'ipc'.
    """
    serie = leer_columna_g(path_xls)
    conteo = {}
    for cell in serie:
        for pref in extraer_prefijos_de_texto(cell):
            pref = pref.lower()
            if pref == "ipc":   # 👈 filtro para excluir IPC
                continue
            conteo[pref] = conteo.get(pref, 0) + 1
    return conteo


# === 5) Mapear archivos por índice en cada ZIP extraído ===
map1 = build_index_map(extract1)
map2 = build_index_map(extract2)

print(f"Archivos detectados: perfil_1 -> {len(map1)} indices, perfil_2 -> {len(map2)} indices")

# === 6) Procesar índices 1..348 y generar comparaciones ===
out_dir = "/content/comparaciones"
os.makedirs(out_dir, exist_ok=True)

faltantes_zip1 = []
faltantes_zip2 = []

for i in range(1, 349):
    path1 = map1.get(i, None)
    path2 = map2.get(i, None)

    if path1 is None:
        faltantes_zip1.append(i)
    if path2 is None:
        faltantes_zip2.append(i)

    conteo1 = contar_prefijos_en_archivo(path1) if path1 else {}
    conteo2 = contar_prefijos_en_archivo(path2) if path2 else {}

    prefijos = sorted(set(conteo1) | set(conteo2))
    data = {
        "Prefijo": prefijos,
        "Total1": [conteo1.get(p, 0) for p in prefijos],
        "Total2": [conteo2.get(p, 0) for p in prefijos],
    }
    df_out = pd.DataFrame(data)
    out_path = os.path.join(out_dir, f"comparacion_{i}.xlsx")
    df_out.to_excel(out_path, index=False)

print("✅ Comparaciones generadas en:", out_dir)
if faltantes_zip1:
    print("ℹ️ Indices sin archivo en perfil_1:", faltantes_zip1[:20], "..." if len(faltantes_zip1) > 20 else "")
if faltantes_zip2:
    print("ℹ️ Indices sin archivo en perfil_2:", faltantes_zip2[:20], "..." if len(faltantes_zip2) > 20 else "")

# === 7) Crear ZIP final de la carpeta 'comparaciones' ===
zip_base = "/content/comparaciones"  # base_name sin extensión
zip_path = shutil.make_archive(zip_base, 'zip', out_dir)
print("📦 ZIP creado en:", zip_path)

# (Opcional) Copiar el ZIP final a tu Drive:
# shutil.copy(zip_path, "/content/drive/MyDrive/comparaciones.zip")
# print("📁 Copiado a Drive:", "/content/drive/MyDrive/comparaciones.zip")
