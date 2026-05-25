import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN GLOBAL DE ESTILO
# =============================================================================
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

PALETTE = {
    'prestigiosa': '#C0392B',
    'no_prestigiosa': '#2980B9',
    'original': '#2C3E50',
    'sin_outliers': '#27AE60',
    'outlier': '#E74C3C',
    'accent': '#F39C12',
    'neutral': '#7F8C8D'
}

OUTPUT_DIR = "graficas_adicionales_con_controles"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# CARGA Y PREPARACIÓN BASE (CON NUEVAS VARIABLES)
# =============================================================================
ruta_archivo = "/content/Datos_2026_1.xlsx"
df = pd.read_excel(ruta_archivo, sheet_name="Matriz 1_", header=0)
df = df.rename(columns={
    'Detech': 'DeTech',
    'Exprt. Conjunta': 'Experticia',
    '# de inventores': 'Inventores',
    'ValorInc': 'ValorInc',
    'Rank': 'Rank',
    'PatenUniv': 'PatenUniv',
    'PatenInd': 'PatenInd'
})
variables = ['ValorInc', 'DeTech', 'Rank', 'Experticia', 'Inventores', 'PatenUniv', 'PatenInd']
df_modelo = df[variables].dropna().copy()
df_modelo['DeTech_sq'] = df_modelo['DeTech'] ** 2
df_modelo['log_PatenUniv'] = np.log1p(df_modelo['PatenUniv'])   # log(1+x)
df_modelo['log_PatenInd'] = np.log1p(df_modelo['PatenInd'])

# =============================================================================
# FUNCIÓN AUXILIAR PARA AJUSTAR NB2 (MANEJA CONSTANTES Y CONVERGENCIA)
# =============================================================================
def estimar_nb2(y, X, metodo='bfgs', maxiter=3000):
    """Estima NegativeBinomial NB2 con manejo robusto."""
    if 'const' not in X.columns:
        X = sm.add_constant(X)
    # Eliminar columnas con varianza cero (excepto constante)
    var = X.var()
    const_cols = var[var == 0].index.tolist()
    if 'const' in const_cols:
        const_cols.remove('const')
    if const_cols:
        print(f"  [!] Eliminando predictores constantes: {const_cols}")
        X = X.drop(columns=const_cols)
    try:
        modelo = NegativeBinomial(y, X, loglike_method='nb2')
        res = modelo.fit(method=metodo, maxiter=maxiter, disp=0, tol=1e-6)
        if not res.mle_retvals['converged']:
            print(f"  [Advertencia] Modelo no convergió después de {maxiter} iteraciones.")
        return res
    except Exception as e:
        print(f"  [Error] No se pudo estimar el modelo: {e}")
        return None

# =============================================================================
# 1. ANÁLISIS DE OUTLIERS (CON CONTROLES)
# =============================================================================
print("=" * 70)
print("1. ANÁLISIS DE OUTLIERS (con PatenUniv y PatenInd)")
print("=" * 70)

y = df_modelo['ValorInc']
X_base = df_modelo[['DeTech', 'DeTech_sq', 'Rank', 'Experticia', 'Inventores', 'log_PatenUniv', 'log_PatenInd']]
modelo_original = estimar_nb2(y, X_base)
if modelo_original is None:
    raise RuntimeError("No se pudo estimar el modelo base.")

residuos = modelo_original.resid_pearson
umbral = 3
outliers_idx = np.where(np.abs(residuos) > umbral)[0]
print(f"Outliers detectados (|residuo| > {umbral}): {len(outliers_idx)}")

df_out = df_modelo.copy()
df_out['residuo'] = residuos

if len(outliers_idx) > 0:
    df_sin_out = df_modelo.drop(index=df_modelo.index[outliers_idx])
    y_sin = df_sin_out['ValorInc']
    X_sin = df_sin_out[['DeTech', 'DeTech_sq', 'Rank', 'Experticia', 'Inventores', 'log_PatenUniv', 'log_PatenInd']]
    mod_sin_out = estimar_nb2(y_sin, X_sin)
    print("\nModelo sin outliers:")
    print(mod_sin_out.summary2(float_format="%.4f"))
else:
    mod_sin_out = modelo_original

# =============================================================================
# 2. TRANSFORMACIÓN LOGARÍTMICA DE DeTech (CON CONTROLES)
# =============================================================================
print("\n" + "=" * 70)
print("2. TRANSFORMACIÓN LOGARÍTMICA DE DeTech")
print("=" * 70)

epsilon = 1e-6
df_modelo['log_DeTech'] = np.log(df_modelo['DeTech'] + epsilon)
df_modelo['log_DeTech_sq'] = df_modelo['log_DeTech'] ** 2

y_log = df_modelo['ValorInc']
X_log = df_modelo[['log_DeTech', 'log_DeTech_sq', 'Rank', 'Experticia', 'Inventores', 'log_PatenUniv', 'log_PatenInd']]
mod_log = estimar_nb2(y_log, X_log)
if mod_log is not None:
    print(mod_log.summary2(float_format="%.4f"))
    param_log = mod_log.params
    if 'log_DeTech_sq' in param_log and param_log['log_DeTech_sq'] < 0:
        turning_log = -param_log['log_DeTech'] / (2 * param_log['log_DeTech_sq'])
        print(f"U invertida en log(DeTech). Máximo en log(DeTech) = {turning_log:.4f}")
        print(f"Equivalente en DeTech original: {np.exp(turning_log) - epsilon:.4f}")

# =============================================================================
# 3. SEGMENTACIÓN POR PRESTIGIO (CON CONTROLES)
# =============================================================================
print("\n" + "=" * 70)
print("3. SEGMENTACIÓN POR PRESTIGIO (Rank=1 vs Rank=0)")
print("=" * 70)

modelos_seg = {}
for rank_val, etiqueta in [(1, "Prestigiosa"), (0, "No prestigiosa")]:
    sub = df_modelo[df_modelo['Rank'] == rank_val]
    if len(sub) < 20:
        print(f"\n{etiqueta}: muestra muy pequeña (n={len(sub)}). Se omite.")
        continue
    print(f"\n{etiqueta} (n={len(sub)})")
    y_sub = sub['ValorInc']
    X_sub = sub[['DeTech', 'DeTech_sq', 'Experticia', 'Inventores', 'log_PatenUniv', 'log_PatenInd']]
    mod_sub = estimar_nb2(y_sub, X_sub)
    if mod_sub is not None:
        modelos_seg[etiqueta] = (mod_sub, sub)
        print(mod_sub.summary2(float_format="%.4f"))
        if 'DeTech_sq' in mod_sub.params and mod_sub.params['DeTech_sq'] < 0:
            print("  => Posible U invertida en este subgrupo.")
        else:
            print("  => No hay U invertida en este subgrupo.")

# =============================================================================
# 4. INTERACCIONES (CON CONTROLES)
# =============================================================================
print("\n" + "=" * 70)
print("4. INTERACCIONES (Rank × DeTech, Experticia × DeTech)")
print("=" * 70)

df_modelo['DeTech_Rank'] = df_modelo['DeTech'] * df_modelo['Rank']
df_modelo['DeTech_sq_Rank'] = df_modelo['DeTech_sq'] * df_modelo['Rank']
df_modelo['DeTech_Exp'] = df_modelo['DeTech'] * df_modelo['Experticia']
df_modelo['DeTech_sq_Exp'] = df_modelo['DeTech_sq'] * df_modelo['Experticia']

# Interacción con Rank
X_int_rank = df_modelo[['DeTech', 'DeTech_sq', 'Rank', 'DeTech_Rank', 'DeTech_sq_Rank',
                        'Experticia', 'Inventores', 'log_PatenUniv', 'log_PatenInd']]
mod_int_rank = estimar_nb2(y, X_int_rank)
if mod_int_rank is not None:
    print("Modelo con interacción Rank:")
    print(mod_int_rank.summary2(float_format="%.4f"))

# Interacción con Experticia
X_int_exp = df_modelo[['DeTech', 'DeTech_sq', 'Rank', 'DeTech_Exp', 'DeTech_sq_Exp',
                       'Experticia', 'Inventores', 'log_PatenUniv', 'log_PatenInd']]
mod_int_exp = estimar_nb2(y, X_int_exp)
if mod_int_exp is not None:
    print("\nModelo con interacción Experticia:")
    print(mod_int_exp.summary2(float_format="%.4f"))

# =============================================================================
#  FUNCIÓN AUXILIAR PARA PREDECIR DE FORMA SEGURA (MANEJA NOMBRES DE COLUMNAS)
# =============================================================================
def predecir_seguro(modelo, DeTech_vals, Rank_val, Experticia_val, Inventores_val,
                    log_PatenUniv_val, log_PatenInd_val, DeTech_sq_vals=None):
    if DeTech_sq_vals is None:
        DeTech_sq_vals = DeTech_vals ** 2
    exog_names = modelo.model.exog_names
    n = len(DeTech_vals)
    X_pred = np.zeros((n, len(exog_names)))
    for i, name in enumerate(exog_names):
        if name == 'const':
            X_pred[:, i] = 1.0
        elif name == 'DeTech':
            X_pred[:, i] = DeTech_vals
        elif name == 'DeTech_sq':
            X_pred[:, i] = DeTech_sq_vals
        elif name == 'Rank':
            X_pred[:, i] = Rank_val
        elif name == 'Experticia':
            X_pred[:, i] = Experticia_val
        elif name == 'Inventores':
            X_pred[:, i] = Inventores_val
        elif name == 'log_PatenUniv':
            X_pred[:, i] = log_PatenUniv_val
        elif name == 'log_PatenInd':
            X_pred[:, i] = log_PatenInd_val
        else:
            X_pred[:, i] = 0.0  # interacciones u otras variables extra
    return modelo.predict(X_pred)

# =============================================================================
# GRÁFICAS (TODAS ACTUALIZADAS CON LOS NUEVOS CONTROLES)
# =============================================================================
DeTech_grid = np.linspace(df_modelo['DeTech'].min(), df_modelo['DeTech'].max(), 200)
# Valores medios de los controles
medias = df_modelo[['Experticia', 'Inventores', 'log_PatenUniv', 'log_PatenInd']].median()
Rank_graf = 1  # base

# Para evitar errores si algún modelo es None
modelo_graf = mod_sin_out if mod_sin_out is not None else modelo_original
if modelo_graf is None:
    print("No hay modelo válido para graficar. Abortando.")
    exit()

# ---------------------------------------------------------------------
# FIGURA 1: Diagnóstico de residuos (igual, pero con controles)
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Figura 1: Diagnóstico de Residuos de Pearson (con controles)", fontsize=14, fontweight='bold', y=1.02)
ax = axes[0]
ax.hist(residuos, bins=30, color=PALETTE['original'], edgecolor='white', alpha=0.85)
ax.axvline(umbral, color=PALETTE['outlier'], linestyle='--', linewidth=1.8, label=f'Umbral ±{umbral}')
ax.axvline(-umbral, color=PALETTE['outlier'], linestyle='--', linewidth=1.8)
ax.set_xlabel("Residuo de Pearson"); ax.set_ylabel("Frecuencia")
ax.set_title("Distribución de Residuos de Pearson"); ax.legend()
ax = axes[1]
mu_fit = modelo_original.predict()
colors_res = [PALETTE['outlier'] if abs(r) > umbral else PALETTE['original'] for r in residuos]
ax.scatter(mu_fit, residuos, c=colors_res, alpha=0.65, edgecolors='none', s=40)
ax.axhline(umbral, color=PALETTE['outlier'], linestyle='--', linewidth=1.5, label=f'|residuo| = {umbral}')
ax.axhline(-umbral, color=PALETTE['outlier'], linestyle='--', linewidth=1.5)
ax.axhline(0, color='black', linewidth=0.8)
outlier_patch = mpatches.Patch(color=PALETTE['outlier'], label=f'Outliers (n={len(outliers_idx)})')
normal_patch = mpatches.Patch(color=PALETTE['original'], label='Observaciones normales')
ax.legend(handles=[outlier_patch, normal_patch])
ax.set_xlabel("Valor ajustado (μ̂)"); ax.set_ylabel("Residuo de Pearson")
ax.set_title("Residuos vs Valores Ajustados")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig1_diagnostico_residuos.png")
plt.show()
print(f"✓ Guardada: {OUTPUT_DIR}/fig1_diagnostico_residuos.png")

# ---------------------------------------------------------------------
# FIGURA 2: Dispersión en barras (con cuartiles de DeTech)
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Figura 2: Distribución de ValorInc por Cuartil de DeTech y Prestigio", fontsize=14, fontweight='bold', y=1.02)
# Recalcular cuartiles (puede fallar si muchos iguales)
_q_temp = pd.qcut(df_modelo['DeTech'], q=4, duplicates='drop')
_n_bins = _q_temp.cat.categories.nunique()
_labels_all = ['Q1\n(Baja)', 'Q2\n(Mod-Baja)', 'Q3\n(Mod-Alta)', 'Q4\n(Alta)']
_labels_use = _labels_all[:_n_bins]
df_modelo['Q_DeTech'] = pd.qcut(df_modelo['DeTech'], q=4, labels=_labels_use, duplicates='drop')
ax = axes[0]
estadísticas = df_modelo.groupby('Q_DeTech', observed=True)['ValorInc'].agg(['mean', 'std', 'count'])
estadísticas['se'] = estadísticas['std'] / np.sqrt(estadísticas['count'])
bars = ax.bar(estadísticas.index, estadísticas['mean'],
              yerr=estadísticas['se'] * 1.96,
              color=[plt.cm.Blues_r(i / _n_bins + 0.2) for i in range(_n_bins)],
              edgecolor='white', capsize=5, linewidth=0.8)
ax.set_xlabel("Cuartil de DeTech"); ax.set_ylabel("Media de ValorInc")
ax.set_title("Media de ValorInc por Cuartil de DeTech (IC 95%)")
for bar, (_, row) in zip(bars, estadísticas.iterrows()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + row['se']*1.96 + 0.5,
            f'n={int(row["count"])}', ha='center', va='bottom', fontsize=9, color='gray')
ax = axes[1]
df_cross = df_modelo.groupby(['Q_DeTech', 'Rank'], observed=True)['ValorInc'].mean().unstack()
df_cross.columns = ['No Prestigiosa', 'Prestigiosa']
x = np.arange(len(df_cross))
width = 0.35
ax.bar(x - width/2, df_cross['No Prestigiosa'], width, label='No Prestigiosa', color=PALETTE['no_prestigiosa'], alpha=0.85, edgecolor='white')
ax.bar(x + width/2, df_cross['Prestigiosa'], width, label='Prestigiosa', color=PALETTE['prestigiosa'], alpha=0.85, edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(df_cross.index)
ax.set_xlabel("Cuartil de DeTech"); ax.set_ylabel("Media de ValorInc")
ax.set_title("Media de ValorInc por Cuartil y Tipo"); ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig2_dispersion_barras.png")
plt.show()
print(f"✓ Guardada: {OUTPUT_DIR}/fig2_dispersion_barras.png")

# ---------------------------------------------------------------------
# FIGURA 3: Curva original vs sin outliers
# ---------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle("Figura 3: Curvas Predichas — Modelo Original vs Sin Outliers (con controles)", fontsize=14, fontweight='bold')
mu_orig = predecir_seguro(modelo_original, DeTech_grid, Rank_graf,
                          medias['Experticia'], medias['Inventores'],
                          medias['log_PatenUniv'], medias['log_PatenInd'])
ax.plot(DeTech_grid, mu_orig, color=PALETTE['original'], linewidth=2.5, label='Modelo original', zorder=3)
if len(outliers_idx) > 0 and mod_sin_out is not None:
    mu_sin = predecir_seguro(mod_sin_out, DeTech_grid, Rank_graf,
                            medias['Experticia'], medias['Inventores'],
                            medias['log_PatenUniv'], medias['log_PatenInd'])
    ax.plot(DeTech_grid, mu_sin, color=PALETTE['sin_outliers'], linewidth=2.5, linestyle='--', label='Sin outliers', zorder=3)
    ax.fill_between(DeTech_grid, mu_orig, mu_sin, alpha=0.15, color=PALETTE['accent'], label='Diferencia entre modelos')
out_detech = df_modelo.iloc[outliers_idx]['DeTech'].values if len(outliers_idx) > 0 else []
ax.scatter(out_detech, np.full_like(out_detech, mu_orig.min()*0.97), marker='|', s=200, color=PALETTE['outlier'], linewidth=2, label=f'Outliers (n={len(outliers_idx)})', zorder=4)
ax.set_xlabel("DeTech (Índice de Diversificación Tecnológica)")
ax.set_ylabel("Valor Esperado de ValorInc")
ax.legend(framealpha=0.9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig3_original_vs_sin_outliers.png")
plt.show()
print(f"✓ Guardada: {OUTPUT_DIR}/fig3_original_vs_sin_outliers.png")

# ---------------------------------------------------------------------
# FIGURA 4: Prestigiosas (si existe)
# ---------------------------------------------------------------------
if "Prestigiosa" in modelos_seg:
    mod_prest, sub_prest = modelos_seg["Prestigiosa"]
    med_prest = sub_prest[['Experticia', 'Inventores', 'log_PatenUniv', 'log_PatenInd']].median()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figura 4: Universidades PRESTIGIOSAS — Relación DeTech vs ValorInc", fontsize=13, fontweight='bold', y=1.02)
    ax = axes[0]
    mu_prest = predecir_seguro(mod_prest, DeTech_grid, 1,
                              med_prest['Experticia'], med_prest['Inventores'],
                              med_prest['log_PatenUniv'], med_prest['log_PatenInd'])
    idx_max = np.argmax(mu_prest)
    dt_max = DeTech_grid[idx_max]
    mu_max = mu_prest[idx_max]
    ax.scatter(sub_prest['DeTech'], sub_prest['ValorInc'], color=PALETTE['prestigiosa'], alpha=0.5, s=50, zorder=2, label='Observaciones')
    ax.plot(DeTech_grid, mu_prest, color=PALETTE['prestigiosa'], linewidth=2.5, zorder=3, label='Curva predicha NB2')
    ax.axvline(dt_max, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.scatter([dt_max], [mu_max], color='black', s=100, zorder=5, label=f'Máximo: DeTech ≈ {dt_max:.2f}')
    ax.set_xlabel("DeTech"); ax.set_ylabel("Valor Esperado de ValorInc")
    ax.set_title(f"Curva Predicha (n={len(sub_prest)})"); ax.legend(fontsize=9)
    ax = axes[1]
    ax.hist(sub_prest['ValorInc'], bins=15, color=PALETTE['prestigiosa'], edgecolor='white', alpha=0.85)
    ax.axvline(sub_prest['ValorInc'].mean(), color='black', linestyle='--', linewidth=1.8, label=f"Media = {sub_prest['ValorInc'].mean():.1f}")
    ax.axvline(sub_prest['ValorInc'].median(), color='gray', linestyle=':', linewidth=1.8, label=f"Mediana = {sub_prest['ValorInc'].median():.1f}")
    ax.set_xlabel("ValorInc"); ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de ValorInc\nUniversidades Prestigiosas"); ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig4_curva_prestigiosa.png")
    plt.show()
    print(f"✓ Guardada: {OUTPUT_DIR}/fig4_curva_prestigiosa.png")

# ---------------------------------------------------------------------
# FIGURA 5: No prestigiosas
# ---------------------------------------------------------------------
if "No prestigiosa" in modelos_seg:
    mod_noprest, sub_noprest = modelos_seg["No prestigiosa"]
    med_noprest = sub_noprest[['Experticia', 'Inventores', 'log_PatenUniv', 'log_PatenInd']].median()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figura 5: Universidades NO PRESTIGIOSAS — Relación DeTech vs ValorInc", fontsize=13, fontweight='bold', y=1.02)
    ax = axes[0]
    mu_noprest = predecir_seguro(mod_noprest, DeTech_grid, 0,
                                med_noprest['Experticia'], med_noprest['Inventores'],
                                med_noprest['log_PatenUniv'], med_noprest['log_PatenInd'])
    ax.scatter(sub_noprest['DeTech'], sub_noprest['ValorInc'], color=PALETTE['no_prestigiosa'], alpha=0.4, s=30, zorder=2, label='Observaciones')
    ax.plot(DeTech_grid, mu_noprest, color=PALETTE['no_prestigiosa'], linewidth=2.5, zorder=3, label='Curva predicha NB2')
    ax.set_xlabel("DeTech"); ax.set_ylabel("Valor Esperado de ValorInc")
    ax.set_title(f"Curva Predicha (n={len(sub_noprest)})"); ax.legend(fontsize=9)
    ax = axes[1]
    ax.hist(sub_noprest['ValorInc'], bins=30, color=PALETTE['no_prestigiosa'], edgecolor='white', alpha=0.85)
    ax.axvline(sub_noprest['ValorInc'].mean(), color='black', linestyle='--', linewidth=1.8, label=f"Media = {sub_noprest['ValorInc'].mean():.1f}")
    ax.axvline(sub_noprest['ValorInc'].median(), color='gray', linestyle=':', linewidth=1.8, label=f"Mediana = {sub_noprest['ValorInc'].median():.1f}")
    ax.set_xlabel("ValorInc"); ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de ValorInc\nUniversidades No Prestigiosas"); ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig5_curva_no_prestigiosa.png")
    plt.show()
    print(f"✓ Guardada: {OUTPUT_DIR}/fig5_curva_no_prestigiosa.png")

# ---------------------------------------------------------------------
# FIGURA 6: Comparativa ambos segmentos
# ---------------------------------------------------------------------
if "Prestigiosa" in modelos_seg and "No prestigiosa" in modelos_seg:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figura 6: Comparativa de Curvas Predichas por Tipo de Universidad", fontsize=13, fontweight='bold', y=1.02)
    ax = axes[0]
    ax.plot(DeTech_grid, mu_prest, color=PALETTE['prestigiosa'], linewidth=2.5, label='Prestigiosa (U invertida)')
    ax.plot(DeTech_grid, mu_noprest, color=PALETTE['no_prestigiosa'], linewidth=2.5, label='No Prestigiosa (decreciente)')
    ax.scatter([dt_max], [mu_max], color=PALETTE['prestigiosa'], s=120, zorder=5, edgecolors='black', linewidth=1)
    ax.annotate(f'Óptimo\n≈{dt_max:.2f}', xy=(dt_max, mu_max), xytext=(dt_max+0.1, mu_max+0.5),
                fontsize=9, color=PALETTE['prestigiosa'], arrowprops=dict(arrowstyle='->', color=PALETTE['prestigiosa']))
    ax.set_xlabel("DeTech"); ax.set_ylabel("Valor Esperado de ValorInc")
    ax.set_title("Curvas Predichas Superpuestas"); ax.legend()
    ax = axes[1]
    diferencia = mu_prest - mu_noprest
    ax.fill_between(DeTech_grid, diferencia, 0, where=(diferencia>0), color=PALETTE['prestigiosa'], alpha=0.4, label='Ventaja Prestigiosa')
    ax.fill_between(DeTech_grid, diferencia, 0, where=(diferencia<=0), color=PALETTE['no_prestigiosa'], alpha=0.4, label='Ventaja No Prestigiosa')
    ax.plot(DeTech_grid, diferencia, color='black', linewidth=1.5)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel("DeTech"); ax.set_ylabel("Diferencia en ValorInc esperado\n(Prestigiosa − No Prestigiosa)")
    ax.set_title("Diferencia entre Curvas por Tipo"); ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig6_comparativa_segmentos.png")
    plt.show()
    print(f"✓ Guardada: {OUTPUT_DIR}/fig6_comparativa_segmentos.png")

# ---------------------------------------------------------------------
# FIGURA 7: Coeficientes comparativos (ahora incluye log_PatenUniv y log_PatenInd)
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Figura 7: Comparativa de Coeficientes NB2 entre Especificaciones (con controles)", fontsize=13, fontweight='bold', y=1.02)
modelos_comp = {
    'Original': modelo_original,
    'Sin Outliers': mod_sin_out,
    'Log-DeTech': mod_log
}
vars_comunes = ['Inventores', 'Rank', 'Experticia', 'log_PatenUniv', 'log_PatenInd']
for ax_i, (nombre, mod) in enumerate(modelos_comp.items()):
    ax = axes[ax_i]
    if mod is None:
        ax.text(0.5, 0.5, 'Modelo no disponible', ha='center', va='center')
        continue
    params_sel = {v: mod.params.get(v, np.nan) for v in vars_comunes if v in mod.params}
    bse_sel = {v: mod.bse.get(v, np.nan) for v in vars_comunes if v in mod.bse}
    names = list(params_sel.keys())
    coefs = np.array([params_sel[v] for v in names])
    errors = np.array([bse_sel[v] * 1.96 for v in names])
    colors_bar = [PALETTE['original'] if c > 0 else PALETTE['no_prestigiosa'] for c in coefs]
    ax.barh(names, coefs, xerr=errors, color=colors_bar, edgecolor='white', capsize=4, alpha=0.85)
    ax.axvline(0, color='black', linewidth=1.2)
    ax.set_title(nombre, fontweight='bold')
    ax.set_xlabel("Coeficiente (β ± 1.96·SE)")
    if ax_i == 0:
        ax.set_ylabel("Variable")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig7_coeficientes_comparativa.png")
plt.show()
print(f"✓ Guardada: {OUTPUT_DIR}/fig7_coeficientes_comparativa.png")

# ---------------------------------------------------------------------
# FIGURA 8: Heatmap de correlación y scatter
# ---------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Figura 8: Estructura de Correlación y Relaciones Bivariadas (con controles)", fontsize=13, fontweight='bold', y=1.02)
ax = axes[0]
corr_matrix = df_modelo[['ValorInc', 'DeTech', 'Rank', 'Experticia', 'Inventores', 'log_PatenUniv', 'log_PatenInd']].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, ax=ax, annot=True, fmt=".2f", cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            linewidths=0.5, annot_kws={"size": 9}, cbar_kws={"shrink": 0.8})
ax.set_title("Correlaciones de Pearson")
ax = axes[1]
for rank_val, color, label in [(0, PALETTE['no_prestigiosa'], 'No Prestigiosa'),
                               (1, PALETTE['prestigiosa'], 'Prestigiosa')]:
    sub = df_modelo[df_modelo['Rank'] == rank_val]
    ax.scatter(sub['DeTech'], sub['ValorInc'], c=color, alpha=0.5, s=30, label=f'{label} (n={len(sub)})', edgecolors='none')
ax.set_xlabel("DeTech"); ax.set_ylabel("ValorInc")
ax.set_title("ValorInc vs DeTech por Tipo de Universidad"); ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig8_correlacion_dispersion.png")
plt.show()
print(f"✓ Guardada: {OUTPUT_DIR}/fig8_correlacion_dispersion.png")

# ---------------------------------------------------------------------
# FIGURA RESUMEN: Panel 2×3 con principales visualizaciones
# ---------------------------------------------------------------------
fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("Resumen: Análisis NB2 de Diversificación Tecnológica y Valor de la Innovación (con controles)", fontsize=14, fontweight='bold')
ax_a = fig.add_subplot(gs[0, 0])
ax_a.hist(df_modelo['ValorInc'], bins=40, color=PALETTE['original'], edgecolor='white', alpha=0.85)
ax_a.set_xlabel("ValorInc"); ax_a.set_ylabel("Frecuencia"); ax_a.set_title("A. Distribución de ValorInc")
ax_b = fig.add_subplot(gs[0, 1])
ax_b.plot(DeTech_grid, mu_orig, color=PALETTE['original'], linewidth=2, label='Original')
if len(outliers_idx) > 0 and mod_sin_out is not None:
    ax_b.plot(DeTech_grid, mu_sin, color=PALETTE['sin_outliers'], linewidth=2, linestyle='--', label='Sin outliers')
ax_b.set_xlabel("DeTech"); ax_b.set_ylabel("μ̂"); ax_b.set_title("B. Original vs Sin Outliers"); ax_b.legend(fontsize=9)
ax_c = fig.add_subplot(gs[0, 2])
if "Prestigiosa" in modelos_seg and "No prestigiosa" in modelos_seg:
    ax_c.plot(DeTech_grid, mu_prest, color=PALETTE['prestigiosa'], linewidth=2, label='Prestigiosa')
    ax_c.plot(DeTech_grid, mu_noprest, color=PALETTE['no_prestigiosa'], linewidth=2, label='No prestigiosa')
    ax_c.legend(fontsize=9)
ax_c.set_xlabel("DeTech"); ax_c.set_ylabel("μ̂"); ax_c.set_title("C. Curvas por Prestigio")
ax_d = fig.add_subplot(gs[1, 0])
estadísticas = df_modelo.groupby('Q_DeTech', observed=True)['ValorInc'].mean()
estadísticas.plot(kind='bar', ax=ax_d, color=plt.cm.Blues_r(np.linspace(0.2, 0.8, len(estadísticas))), edgecolor='white')
ax_d.set_xlabel("Cuartil DeTech"); ax_d.set_ylabel("Media ValorInc"); ax_d.set_title("D. ValorInc por Cuartil DeTech"); ax_d.tick_params(axis='x', rotation=0)
ax_e = fig.add_subplot(gs[1, 1])
ax_e.scatter(mu_fit, residuos, c=colors_res, alpha=0.6, s=20, edgecolors='none')
ax_e.axhline(umbral, color=PALETTE['outlier'], linestyle='--', linewidth=1.5)
ax_e.axhline(-umbral, color=PALETTE['outlier'], linestyle='--', linewidth=1.5)
ax_e.axhline(0, color='black', linewidth=0.8)
ax_e.set_xlabel("μ̂"); ax_e.set_ylabel("Residuo Pearson"); ax_e.set_title("E. Residuos vs Ajustados")
ax_f = fig.add_subplot(gs[1, 2])
corr_sub = df_modelo[['ValorInc', 'DeTech', 'Experticia', 'Inventores', 'log_PatenUniv', 'log_PatenInd']].corr()
sns.heatmap(corr_sub, ax=ax_f, annot=True, fmt=".2f", cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            linewidths=0.5, annot_kws={"size": 9}, cbar_kws={"shrink": 0.8})
ax_f.set_title("F. Correlaciones")
plt.savefig(f"{OUTPUT_DIR}/fig_resumen_panel.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ Guardada: {OUTPUT_DIR}/fig_resumen_panel.png")

# =============================================================================
# TABLA RESUMEN DE MODELOS (INCLUYENDO LOS NUEVOS CONTROLES)
# =============================================================================
print("\n" + "=" * 70)
print("TABLA RESUMEN COMPARATIVA DE MODELOS (con PatenUniv y PatenInd)")
print("=" * 70)
resumen_modelos = {
    'Original (n=345)': modelo_original,
    'Sin Outliers (n=333)': mod_sin_out,
    'Log-DeTech (n=345)': mod_log,
}
if "Prestigiosa" in modelos_seg:
    resumen_modelos['Prestigiosas (n=31)'] = modelos_seg["Prestigiosa"][0]
if "No prestigiosa" in modelos_seg:
    resumen_modelos['No Prestigiosas (n=314)'] = modelos_seg["No prestigiosa"][0]

print(f"\n{'Modelo':<25} {'Log-Lik':>10} {'AIC':>10} {'BIC':>10} {'α':>8} {'Pseudo-R²':>10}")
print("-" * 75)
for nombre, mod in resumen_modelos.items():
    if mod is not None:
        print(f"{nombre:<25} {mod.llf:>10.1f} {mod.aic:>10.1f} {mod.bic:>10.1f} "
              f"{mod.params.get('alpha', np.nan):>8.4f} {mod.prsquared:>10.4f}")

print("\n✅ Análisis de sensibilidad completo con las variables de control PatenUniv y PatenInd.")
print(f"📁 Gráficas guardadas en: ./{OUTPUT_DIR}/")
