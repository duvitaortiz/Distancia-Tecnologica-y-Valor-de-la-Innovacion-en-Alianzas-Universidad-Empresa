import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import NegativeBinomial
from scipy import stats
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

OUTPUT_DIR = "graficas_adicionales"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# CARGA Y PREPARACIÓN BASE
# =============================================================================
ruta_archivo = "/content/ruta.xlsx"
df = pd.read_excel(ruta_archivo, sheet_name="Matriz 1_", header=0)
df = df.rename(columns={
    'Detech': 'DeTech',
    'Exprt. Conjunta': 'Experticia',
    '# de inventores': 'Inventores',
    'ValorInc': 'ValorInc',
    'Rank': 'Rank'
})
variables = ['ValorInc', 'DeTech', 'Rank', 'Experticia', 'Inventores']
df_modelo = df[variables].dropna().copy()
df_modelo['DeTech_sq'] = df_modelo['DeTech'] ** 2


# =============================================================================
# FUNCIÓN AUXILIAR PARA AJUSTAR NB2
# =============================================================================
def estimar_nb2(y, X, metodo='nm'):
    modelo = NegativeBinomial(y, X, loglike_method='nb2')
    res = modelo.fit(method=metodo, maxiter=2000, disp=0)
    return res


def intervalo_confianza_prediccion(modelo, X_pred, n_boot=200, alpha=0.05):
    """Calcula IC por bootstrap para las predicciones."""
    preds = []
    y_obs = modelo.model.endog
    X_obs = modelo.model.exog
    n = len(y_obs)
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        y_b = y_obs[idx]
        X_b = X_obs[idx]
        try:
            m_b = NegativeBinomial(y_b, X_b, loglike_method='nb2')
            r_b = m_b.fit(method='nm', maxiter=1000, disp=0)
            preds.append(r_b.predict(X_pred))
        except Exception:
            continue
    if len(preds) < 10:
        return None, None
    preds = np.array(preds)
    lb = np.percentile(preds, 100 * alpha / 2, axis=0)
    ub = np.percentile(preds, 100 * (1 - alpha / 2), axis=0)
    return lb, ub


# =============================================================================
# 1. ANÁLISIS DE OUTLIERS
# =============================================================================
print("=" * 70)
print("1. ANÁLISIS DE OUTLIERS")
print("=" * 70)

y = df_modelo['ValorInc']
X = sm.add_constant(df_modelo[['DeTech', 'DeTech_sq', 'Rank', 'Experticia', 'Inventores']])
modelo_original = estimar_nb2(y, X)
residuos = modelo_original.resid_pearson

umbral = 3
outliers_idx = np.where(np.abs(residuos) > umbral)[0]
print(f"Outliers detectados (|residuo| > {umbral}): {len(outliers_idx)}")

df_out = df_modelo.copy()
df_out['residuo'] = residuos

if len(outliers_idx) > 0:
    df_sin_out = df_modelo.drop(index=df_modelo.index[outliers_idx])
    y_sin = df_sin_out['ValorInc']
    X_sin = sm.add_constant(df_sin_out[['DeTech', 'DeTech_sq', 'Rank', 'Experticia', 'Inventores']])
    mod_sin_out = estimar_nb2(y_sin, X_sin)
    print("\nModelo sin outliers:")
    print(mod_sin_out.summary2(float_format="%.4f"))
else:
    mod_sin_out = modelo_original

# =============================================================================
# 2. TRANSFORMACIÓN LOGARÍTMICA
# =============================================================================
print("\n" + "=" * 70)
print("2. TRANSFORMACIÓN LOGARÍTMICA DE DeTech")
print("=" * 70)

epsilon = 1e-6
df_modelo['log_DeTech'] = np.log(df_modelo['DeTech'] + epsilon)
df_modelo['log_DeTech_sq'] = df_modelo['log_DeTech'] ** 2

y_log = df_modelo['ValorInc']
X_log = sm.add_constant(df_modelo[['log_DeTech', 'log_DeTech_sq', 'Rank', 'Experticia', 'Inventores']])
mod_log = estimar_nb2(y_log, X_log)
print(mod_log.summary2(float_format="%.4f"))

param_log = mod_log.params
if 'log_DeTech_sq' in param_log and param_log['log_DeTech_sq'] < 0:
    turning_log = -param_log['log_DeTech'] / (2 * param_log['log_DeTech_sq'])
    print(f"U invertida en log(DeTech). Máximo en log(DeTech) = {turning_log:.4f}")
    print(f"Equivalente en DeTech original: {np.exp(turning_log) - epsilon:.4f}")

# =============================================================================
# 3. SEGMENTACIÓN POR PRESTIGIO
# =============================================================================
print("\n" + "=" * 70)
print("3. SEGMENTACIÓN POR PRESTIGIO (Rank=1 vs Rank=0)")
print("=" * 70)

modelos_seg = {}
for rank_val, etiqueta in [(1, "Prestigiosa"), (0, "No prestigiosa")]:
    sub = df_modelo[df_modelo['Rank'] == rank_val]
    if len(sub) < 30:
        print(f"\n{etiqueta}: muestra insuficiente (n={len(sub)}). Se omite.")
        continue
    y_sub = sub['ValorInc']
    X_sub = sm.add_constant(sub[['DeTech', 'DeTech_sq', 'Experticia', 'Inventores']])
    mod_sub = estimar_nb2(y_sub, X_sub)
    modelos_seg[etiqueta] = (mod_sub, sub)
    print(f"\n{etiqueta} (n={len(sub)})")
    print(mod_sub.summary2(float_format="%.4f"))

# =============================================================================
# 4. INTERACCIONES
# =============================================================================
print("\n" + "=" * 70)
print("4. INTERACCIONES (Rank × DeTech, Experticia × DeTech)")
print("=" * 70)

df_modelo['DeTech_Rank'] = df_modelo['DeTech'] * df_modelo['Rank']
df_modelo['DeTech_sq_Rank'] = df_modelo['DeTech_sq'] * df_modelo['Rank']
df_modelo['DeTech_Exp'] = df_modelo['DeTech'] * df_modelo['Experticia']
df_modelo['DeTech_sq_Exp'] = df_modelo['DeTech_sq'] * df_modelo['Experticia']

X_int_rank = sm.add_constant(df_modelo[['DeTech', 'DeTech_sq', 'Rank',
                                         'DeTech_Rank', 'DeTech_sq_Rank',
                                         'Experticia', 'Inventores']])
mod_int_rank = estimar_nb2(y, X_int_rank)
print("Modelo con interacción Rank:")
print(mod_int_rank.summary2(float_format="%.4f"))

X_int_exp = sm.add_constant(df_modelo[['DeTech', 'DeTech_sq', 'Rank',
                                        'DeTech_Exp', 'DeTech_sq_Exp',
                                        'Experticia', 'Inventores']])
mod_int_exp = estimar_nb2(y, X_int_exp)
print("\nModelo con interacción Experticia:")
print(mod_int_exp.summary2(float_format="%.4f"))


# =============================================================================
#   VISUALIZACIONES MEJORADAS Y AMPLIADAS
# =============================================================================

DeTech_grid = np.linspace(df_modelo['DeTech'].min(), df_modelo['DeTech'].max(), 200)

# ─────────────────────────────────────────────────────────────────────
# FIGURA 1: Diagnóstico de outliers (histograma residuos + scatter)
# ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Figura 1: Diagnóstico de Residuos de Pearson", fontsize=14, fontweight='bold', y=1.02)

# 1a. Histograma de residuos
ax = axes[0]
ax.hist(residuos, bins=30, color=PALETTE['original'], edgecolor='white', alpha=0.85)
ax.axvline(umbral, color=PALETTE['outlier'], linestyle='--', linewidth=1.8, label=f'Umbral ±{umbral}')
ax.axvline(-umbral, color=PALETTE['outlier'], linestyle='--', linewidth=1.8)
ax.set_xlabel("Residuo de Pearson")
ax.set_ylabel("Frecuencia")
ax.set_title("Distribución de Residuos de Pearson")
ax.legend()

# 1b. Residuos vs valores ajustados
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
ax.set_xlabel("Valor ajustado (μ̂)")
ax.set_ylabel("Residuo de Pearson")
ax.set_title("Residuos vs Valores Ajustados")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig1_diagnostico_residuos.png")
plt.show()
print(f"✓ Guardada: {OUTPUT_DIR}/fig1_diagnostico_residuos.png")


# ─────────────────────────────────────────────────────────────────────
# FIGURA 2: Dispersión en barras — ValorInc por cuartil de DeTech y Rank
# ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Figura 2: Distribución de ValorInc por Cuartil de DeTech y Prestigio",
             fontsize=14, fontweight='bold', y=1.02)

# pd.qcut puede fallar si DeTech tiene muchos valores repetidos en los bordes.
# Usamos duplicates='drop' para eliminar bordes duplicados y recalculamos labels dinámicamente.
_q_temp = pd.qcut(df_modelo['DeTech'], q=4, duplicates='drop')
_n_bins = _q_temp.cat.categories.nunique()
_labels_all = ['Q1\n(Baja)', 'Q2\n(Mod-Baja)', 'Q3\n(Mod-Alta)', 'Q4\n(Alta)']
_labels_use = _labels_all[:_n_bins]
df_modelo['Q_DeTech'] = pd.qcut(df_modelo['DeTech'], q=4,
                                  labels=_labels_use, duplicates='drop')

# 2a. Barras de media y error estándar por cuartil
ax = axes[0]
estadísticas = df_modelo.groupby('Q_DeTech', observed=True)['ValorInc'].agg(['mean', 'std', 'count'])
estadísticas['se'] = estadísticas['std'] / np.sqrt(estadísticas['count'])
bars = ax.bar(estadísticas.index, estadísticas['mean'],
              yerr=estadísticas['se'] * 1.96,
              color=[plt.cm.Blues_r(i / _n_bins + 0.2) for i in range(_n_bins)],
              edgecolor='white', capsize=5, linewidth=0.8)
ax.set_xlabel("Cuartil de DeTech")
ax.set_ylabel("Media de ValorInc")
ax.set_title("Media de ValorInc por Cuartil de DeTech\n(IC 95%)")
for bar, (_, row) in zip(bars, estadísticas.iterrows()):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + row['se'] * 1.96 + 0.5,
            f'n={int(row["count"])}', ha='center', va='bottom', fontsize=9, color='gray')

# 2b. Barras comparativas por cuartil y Rank
ax = axes[1]
df_cross = df_modelo.groupby(['Q_DeTech', 'Rank'], observed=True)['ValorInc'].mean().unstack()
df_cross.columns = ['No Prestigiosa', 'Prestigiosa']
x = np.arange(len(df_cross))
width = 0.35
bars1 = ax.bar(x - width / 2, df_cross['No Prestigiosa'], width,
               label='No Prestigiosa', color=PALETTE['no_prestigiosa'], alpha=0.85, edgecolor='white')
bars2 = ax.bar(x + width / 2, df_cross['Prestigiosa'], width,
               label='Prestigiosa', color=PALETTE['prestigiosa'], alpha=0.85, edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(df_cross.index)
ax.set_xlabel("Cuartil de DeTech")
ax.set_ylabel("Media de ValorInc")
ax.set_title("Media de ValorInc por Cuartil de DeTech\ny Tipo de Universidad")
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig2_dispersion_barras.png")
plt.show()
print(f"✓ Guardada: {OUTPUT_DIR}/fig2_dispersion_barras.png")


# ─────────────────────────────────────────────────────────────────────
# FIGURA 3: Curva modelo original vs sin outliers (mejorada)
# ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
fig.suptitle("Figura 3: Curvas Predichas — Modelo Original vs Sin Outliers",
             fontsize=14, fontweight='bold')

X_pred_cols = ['DeTech', 'DeTech_sq', 'Rank', 'Experticia', 'Inventores']
X_pred_orig = pd.DataFrame({
    'const': 1.0,
    'DeTech': DeTech_grid,
    'DeTech_sq': DeTech_grid ** 2,
    'Rank': 1,
    'Experticia': df_modelo['Experticia'].median(),
    'Inventores': df_modelo['Inventores'].median()
})[modelo_original.params.index.drop('alpha')]

mu_orig = modelo_original.predict(X_pred_orig)
ax.plot(DeTech_grid, mu_orig, color=PALETTE['original'], linewidth=2.5, label='Modelo original', zorder=3)

if len(outliers_idx) > 0:
    mu_sin = mod_sin_out.predict(X_pred_orig)
    ax.plot(DeTech_grid, mu_sin, color=PALETTE['sin_outliers'], linewidth=2.5,
            linestyle='--', label='Sin outliers', zorder=3)
    ax.fill_between(DeTech_grid, mu_orig, mu_sin, alpha=0.15, color=PALETTE['accent'],
                    label='Diferencia entre modelos')

# Rug plot de outliers
out_detech = df_modelo.iloc[outliers_idx]['DeTech'].values
ax.scatter(out_detech, np.full_like(out_detech, mu_orig.min() * 0.97),
           marker='|', s=200, color=PALETTE['outlier'], linewidth=2,
           label=f'Outliers (n={len(outliers_idx)})', zorder=4)

ax.set_xlabel("DeTech (Índice de Diversificación Tecnológica)")
ax.set_ylabel("Valor Esperado de ValorInc")
ax.set_title("")
ax.legend(framealpha=0.9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig3_original_vs_sin_outliers.png")
plt.show()
print(f"✓ Guardada: {OUTPUT_DIR}/fig3_original_vs_sin_outliers.png")


# ─────────────────────────────────────────────────────────────────────
# FIGURA 4: Curva PRESTIGIOSA — separada con datos observados
# ─────────────────────────────────────────────────────────────────────
if "Prestigiosa" in modelos_seg:
    mod_prest, sub_prest = modelos_seg["Prestigiosa"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figura 4: Universidades PRESTIGIOSAS — Análisis de la Relación DeTech vs ValorInc",
                 fontsize=13, fontweight='bold', y=1.02)

    # 4a. Curva predicha
    ax = axes[0]
    X_pred_prest = pd.DataFrame({
        'const': 1.0,
        'DeTech': DeTech_grid,
        'DeTech_sq': DeTech_grid ** 2,
        'Experticia': sub_prest['Experticia'].median(),
        'Inventores': sub_prest['Inventores'].median()
    })[mod_prest.params.index.drop('alpha')]

    mu_prest = mod_prest.predict(X_pred_prest)

    # Localizar el máximo de la curva
    idx_max = np.argmax(mu_prest)
    dt_max = DeTech_grid[idx_max]
    mu_max = mu_prest[idx_max]

    ax.scatter(sub_prest['DeTech'], sub_prest['ValorInc'],
               color=PALETTE['prestigiosa'], alpha=0.5, s=50, zorder=2, label='Observaciones')
    ax.plot(DeTech_grid, mu_prest, color=PALETTE['prestigiosa'],
            linewidth=2.5, zorder=3, label='Curva predicha NB2')
    ax.axvline(dt_max, color='black', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.scatter([dt_max], [mu_max], color='black', s=100, zorder=5,
               label=f'Máximo: DeTech ≈ {dt_max:.2f}')
    ax.set_xlabel("DeTech")
    ax.set_ylabel("Valor Esperado de ValorInc")
    ax.set_title(f"Curva Predicha (n={len(sub_prest)})")
    ax.legend(fontsize=9)

    # 4b. Distribución de ValorInc en prestigiosas
    ax = axes[1]
    ax.hist(sub_prest['ValorInc'], bins=15, color=PALETTE['prestigiosa'],
            edgecolor='white', alpha=0.85)
    ax.axvline(sub_prest['ValorInc'].mean(), color='black', linestyle='--',
               linewidth=1.8, label=f"Media = {sub_prest['ValorInc'].mean():.1f}")
    ax.axvline(sub_prest['ValorInc'].median(), color='gray', linestyle=':',
               linewidth=1.8, label=f"Mediana = {sub_prest['ValorInc'].median():.1f}")
    ax.set_xlabel("ValorInc")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de ValorInc\nUniversidades Prestigiosas")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig4_curva_prestigiosa.png")
    plt.show()
    print(f"✓ Guardada: {OUTPUT_DIR}/fig4_curva_prestigiosa.png")


# ─────────────────────────────────────────────────────────────────────
# FIGURA 5: Curva NO PRESTIGIOSA — separada con datos observados
# ─────────────────────────────────────────────────────────────────────
if "No prestigiosa" in modelos_seg:
    mod_noprest, sub_noprest = modelos_seg["No prestigiosa"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figura 5: Universidades NO PRESTIGIOSAS — Análisis de la Relación DeTech vs ValorInc",
                 fontsize=13, fontweight='bold', y=1.02)

    # 5a. Curva predicha
    ax = axes[0]
    X_pred_noprest = pd.DataFrame({
        'const': 1.0,
        'DeTech': DeTech_grid,
        'DeTech_sq': DeTech_grid ** 2,
        'Experticia': sub_noprest['Experticia'].median(),
        'Inventores': sub_noprest['Inventores'].median()
    })[mod_noprest.params.index.drop('alpha')]

    mu_noprest = mod_noprest.predict(X_pred_noprest)

    ax.scatter(sub_noprest['DeTech'], sub_noprest['ValorInc'],
               color=PALETTE['no_prestigiosa'], alpha=0.4, s=30, zorder=2, label='Observaciones')
    ax.plot(DeTech_grid, mu_noprest, color=PALETTE['no_prestigiosa'],
            linewidth=2.5, zorder=3, label='Curva predicha NB2')
    ax.set_xlabel("DeTech")
    ax.set_ylabel("Valor Esperado de ValorInc")
    ax.set_title(f"Curva Predicha (n={len(sub_noprest)})")
    ax.legend(fontsize=9)

    # 5b. Distribución de ValorInc en no prestigiosas
    ax = axes[1]
    ax.hist(sub_noprest['ValorInc'], bins=30, color=PALETTE['no_prestigiosa'],
            edgecolor='white', alpha=0.85)
    ax.axvline(sub_noprest['ValorInc'].mean(), color='black', linestyle='--',
               linewidth=1.8, label=f"Media = {sub_noprest['ValorInc'].mean():.1f}")
    ax.axvline(sub_noprest['ValorInc'].median(), color='gray', linestyle=':',
               linewidth=1.8, label=f"Mediana = {sub_noprest['ValorInc'].median():.1f}")
    ax.set_xlabel("ValorInc")
    ax.set_ylabel("Frecuencia")
    ax.set_title("Distribución de ValorInc\nUniversidades No Prestigiosas")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig5_curva_no_prestigiosa.png")
    plt.show()
    print(f"✓ Guardada: {OUTPUT_DIR}/fig5_curva_no_prestigiosa.png")


# ─────────────────────────────────────────────────────────────────────
# FIGURA 6: Comparativa ambos segmentos en un solo panel (mejorada)
# ─────────────────────────────────────────────────────────────────────
if "Prestigiosa" in modelos_seg and "No prestigiosa" in modelos_seg:

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Figura 6: Comparativa de Curvas Predichas por Tipo de Universidad",
                 fontsize=13, fontweight='bold', y=1.02)

    # 6a. Curvas superpuestas
    ax = axes[0]
    ax.plot(DeTech_grid, mu_prest, color=PALETTE['prestigiosa'],
            linewidth=2.5, label='Prestigiosa (U invertida)')
    ax.plot(DeTech_grid, mu_noprest, color=PALETTE['no_prestigiosa'],
            linewidth=2.5, label='No Prestigiosa (decreciente)')

    # Punto máximo de prestigiosas
    ax.scatter([dt_max], [mu_max], color=PALETTE['prestigiosa'], s=120,
               zorder=5, edgecolors='black', linewidth=1)
    ax.annotate(f'Óptimo\n≈{dt_max:.2f}', xy=(dt_max, mu_max),
                xytext=(dt_max + 0.1, mu_max + 0.5),
                fontsize=9, color=PALETTE['prestigiosa'],
                arrowprops=dict(arrowstyle='->', color=PALETTE['prestigiosa']))

    ax.set_xlabel("DeTech (Índice de Diversificación Tecnológica)")
    ax.set_ylabel("Valor Esperado de ValorInc")
    ax.set_title("Curvas Predichas Superpuestas")
    ax.legend()

    # 6b. Diferencia entre curvas
    ax = axes[1]
    diferencia = mu_prest - mu_noprest
    ax.fill_between(DeTech_grid, diferencia, 0,
                    where=(diferencia > 0), color=PALETTE['prestigiosa'], alpha=0.4,
                    label='Ventaja Prestigiosa')
    ax.fill_between(DeTech_grid, diferencia, 0,
                    where=(diferencia <= 0), color=PALETTE['no_prestigiosa'], alpha=0.4,
                    label='Ventaja No Prestigiosa')
    ax.plot(DeTech_grid, diferencia, color='black', linewidth=1.5)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xlabel("DeTech")
    ax.set_ylabel("Diferencia en ValorInc esperado\n(Prestigiosa − No Prestigiosa)")
    ax.set_title("Diferencia entre Curvas por Tipo")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/fig6_comparativa_segmentos.png")
    plt.show()
    print(f"✓ Guardada: {OUTPUT_DIR}/fig6_comparativa_segmentos.png")


# ─────────────────────────────────────────────────────────────────────
# FIGURA 7: Efectos de las covariables (coeficientes con IC)
# ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Figura 7: Comparativa de Coeficientes NB2 entre Especificaciones",
             fontsize=13, fontweight='bold', y=1.02)

modelos_comp = {
    'Original': modelo_original,
    'Sin Outliers': mod_sin_out,
    'Log-DeTech': mod_log
}

vars_comunes = ['Inventores', 'Rank', 'Experticia']

for ax_i, (nombre, mod) in enumerate(modelos_comp.items()):
    ax = axes[ax_i]
    params_sel = {v: mod.params.get(v, np.nan) for v in vars_comunes if v in mod.params}
    bse_sel = {v: mod.bse.get(v, np.nan) for v in vars_comunes if v in mod.bse}

    names = list(params_sel.keys())
    coefs = np.array([params_sel[v] for v in names])
    errors = np.array([bse_sel[v] * 1.96 for v in names])

    colors_bar = [PALETTE['original'] if c > 0 else PALETTE['no_prestigiosa'] for c in coefs]
    bars = ax.barh(names, coefs, xerr=errors, color=colors_bar,
                   edgecolor='white', capsize=4, alpha=0.85)
    ax.axvline(0, color='black', linewidth=1.2)
    ax.set_title(nombre, fontweight='bold')
    ax.set_xlabel("Coeficiente (β ± 1.96·SE)")
    if ax_i == 0:
        ax.set_ylabel("Variable")

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig7_coeficientes_comparativa.png")
plt.show()
print(f"✓ Guardada: {OUTPUT_DIR}/fig7_coeficientes_comparativa.png")


# ─────────────────────────────────────────────────────────────────────
# FIGURA 8: Heatmap de correlación y dispersión bivariada
# ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Figura 8: Estructura de Correlación y Relaciones Bivariadas",
             fontsize=13, fontweight='bold', y=1.02)

# 8a. Heatmap de correlación
ax = axes[0]
corr_matrix = df_modelo[['ValorInc', 'DeTech', 'Rank', 'Experticia', 'Inventores']].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
sns.heatmap(corr_matrix, ax=ax, annot=True, fmt=".2f", cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, linewidths=0.5,
            annot_kws={"size": 10}, cbar_kws={"shrink": 0.8})
ax.set_title("Correlaciones de Pearson")
ax.tick_params(labelsize=10)

# 8b. Scatter ValorInc vs DeTech coloreado por Rank
ax = axes[1]
for rank_val, color, label in [(0, PALETTE['no_prestigiosa'], 'No Prestigiosa'),
                                 (1, PALETTE['prestigiosa'], 'Prestigiosa')]:
    sub = df_modelo[df_modelo['Rank'] == rank_val]
    ax.scatter(sub['DeTech'], sub['ValorInc'], c=color, alpha=0.5,
               s=30, label=f'{label} (n={len(sub)})', edgecolors='none')

ax.set_xlabel("DeTech (Índice de Diversificación Tecnológica)")
ax.set_ylabel("ValorInc")
ax.set_title("ValorInc vs DeTech por Tipo de Universidad")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/fig8_correlacion_dispersion.png")
plt.show()
print(f"✓ Guardada: {OUTPUT_DIR}/fig8_correlacion_dispersion.png")


# ─────────────────────────────────────────────────────────────────────
# FIGURA RESUMEN: Panel 2×3 con las principales visualizaciones
# ─────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
fig.suptitle("Resumen: Análisis NB2 de Diversificación Tecnológica y Valor de la Innovación",
             fontsize=14, fontweight='bold')

# Panel A: Distribución ValorInc
ax_a = fig.add_subplot(gs[0, 0])
ax_a.hist(df_modelo['ValorInc'], bins=40, color=PALETTE['original'], edgecolor='white', alpha=0.85)
ax_a.set_xlabel("ValorInc"); ax_a.set_ylabel("Frecuencia")
ax_a.set_title("A. Distribución de ValorInc")

# Panel B: Curvas originales
ax_b = fig.add_subplot(gs[0, 1])
ax_b.plot(DeTech_grid, mu_orig, color=PALETTE['original'], linewidth=2, label='Original')
if len(outliers_idx) > 0:
    ax_b.plot(DeTech_grid, mu_sin, color=PALETTE['sin_outliers'], linewidth=2, linestyle='--', label='Sin outliers')
ax_b.set_xlabel("DeTech"); ax_b.set_ylabel("μ̂")
ax_b.set_title("B. Original vs Sin Outliers"); ax_b.legend(fontsize=9)

# Panel C: Comparativa segmentos
ax_c = fig.add_subplot(gs[0, 2])
if "Prestigiosa" in modelos_seg and "No prestigiosa" in modelos_seg:
    ax_c.plot(DeTech_grid, mu_prest, color=PALETTE['prestigiosa'], linewidth=2, label='Prestigiosa')
    ax_c.plot(DeTech_grid, mu_noprest, color=PALETTE['no_prestigiosa'], linewidth=2, label='No prestigiosa')
    ax_c.legend(fontsize=9)
ax_c.set_xlabel("DeTech"); ax_c.set_ylabel("μ̂")
ax_c.set_title("C. Curvas por Prestigio")

# Panel D: Dispersión barras
ax_d = fig.add_subplot(gs[1, 0])
estadísticas = df_modelo.groupby('Q_DeTech', observed=True)['ValorInc'].mean()
estadísticas.plot(kind='bar', ax=ax_d, color=plt.cm.Blues_r(np.linspace(0.2, 0.8, len(estadísticas))),
                  edgecolor='white')
ax_d.set_xlabel("Cuartil DeTech"); ax_d.set_ylabel("Media ValorInc")
ax_d.set_title("D. ValorInc por Cuartil DeTech"); ax_d.tick_params(axis='x', rotation=0)

# Panel E: Residuos diagnóstico
ax_e = fig.add_subplot(gs[1, 1])
ax_e.scatter(mu_fit, residuos, c=colors_res, alpha=0.6, s=20, edgecolors='none')
ax_e.axhline(umbral, color=PALETTE['outlier'], linestyle='--', linewidth=1.5)
ax_e.axhline(-umbral, color=PALETTE['outlier'], linestyle='--', linewidth=1.5)
ax_e.axhline(0, color='black', linewidth=0.8)
ax_e.set_xlabel("μ̂"); ax_e.set_ylabel("Residuo Pearson")
ax_e.set_title("E. Residuos vs Ajustados")

# Panel F: Heatmap correlación reducida
ax_f = fig.add_subplot(gs[1, 2])
corr_sub = df_modelo[['ValorInc', 'DeTech', 'Experticia', 'Inventores']].corr()
sns.heatmap(corr_sub, ax=ax_f, annot=True, fmt=".2f", cmap='RdBu_r',
            center=0, vmin=-1, vmax=1, linewidths=0.5,
            annot_kws={"size": 9}, cbar_kws={"shrink": 0.8})
ax_f.set_title("F. Correlaciones")

plt.savefig(f"{OUTPUT_DIR}/fig_resumen_panel.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ Guardada: {OUTPUT_DIR}/fig_resumen_panel.png")


# ─────────────────────────────────────────────────────────────────────
# TABLA RESUMEN DE MODELOS
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("TABLA RESUMEN COMPARATIVA DE MODELOS")
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
    print(f"{nombre:<25} {mod.llf:>10.1f} {mod.aic:>10.1f} {mod.bic:>10.1f} "
          f"{mod.params.get('alpha', np.nan):>8.4f} {mod.prsquared:>10.4f}")

print("\n✅ Análisis de sensibilidad completo.")
print(f"📁 Gráficas guardadas en: ./{OUTPUT_DIR}/")
