# -*- coding: utf-8 -*-
"""
Análisis de la relación entre Stock de Patentes de Inventores y Valor
Incremental de la Innovación — Modelos OLS con variables de control
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white, linear_reset
from statsmodels.stats.stattools import durbin_watson
from scipy.stats.mstats import winsorize
from scipy import stats
import warnings
import os

warnings.filterwarnings('ignore')

# ─── Estilo global ────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family'      : 'serif',
    'font.size'        : 11,
    'axes.titlesize'   : 13,
    'axes.labelsize'   : 11,
    'xtick.labelsize'  : 10,
    'ytick.labelsize'  : 10,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'figure.dpi'       : 150,
    'savefig.dpi'      : 150,
    'savefig.bbox'     : 'tight',
    'axes.grid'        : True,
    'grid.alpha'       : 0.25,
    'grid.linestyle'   : '--',
})

PAL = {
    'azul'    : '#1A5276',
    'rojo'    : '#C0392B',
    'verde'   : '#1E8449',
    'naranja' : '#D35400',
    'gris'    : '#626567',
    'amarillo': '#D4AC0D',
    'claro'   : '#AED6F1',
    'salmon'  : '#F1948A',
}

OUTPUT = "graficas_patentes_con_controles"
os.makedirs(OUTPUT, exist_ok=True)

# =============================================================================
# 1. CARGA Y PREPROCESAMIENTO (con variables de control)
# =============================================================================
print("=" * 70)
print("1. CARGA Y PREPROCESAMIENTO (con controles)")
print("=" * 70)

ruta  = "/content/Datos_2026_1.xlsx"
sheet = "Matriz 1_"
df    = pd.read_excel(ruta, sheet_name=sheet)

# Renombrar para consistencia
df = df.rename(columns={
    'Detech': 'DeTech',
    'Exprt. Conjunta': 'Experticia',
    '# de inventores': 'Inventores',
    'Rank': 'Rank',
    'PatenUniv': 'PatenUniv',
    'PatenInd': 'PatenInd'
})

# Stock acumulado de patentes de los inventores (suma por equipo)
num_inv_cols = [f'Inventor {i}' for i in range(1, 28)]

def sumar_patentes(fila):
    n = fila['Inventores']
    if pd.isna(n) or n <= 0:
        return np.nan
    n = int(n)
    return fila[num_inv_cols[:n]].sum(skipna=True)

df['Total_Patentes_Inventores'] = df.apply(sumar_patentes, axis=1)

# Selección de variables: dependiente, independiente principal + controles
vars_controles = ['DeTech', 'Experticia', 'Inventores', 'Rank', 'PatenUniv', 'PatenInd']
vars_modelo = ['ValorInc', 'Total_Patentes_Inventores'] + vars_controles
df_model = df[vars_modelo].dropna().copy()

# Winsorización al 1% en colas para variables continuas (para robustez)
for col in ['ValorInc', 'Total_Patentes_Inventores'] + vars_controles:
    if df_model[col].dtype in ['float64', 'int64']:
        df_model[f'{col}_w'] = winsorize(df_model[col], limits=(0.01, 0.01))
    else:
        df_model[f'{col}_w'] = df_model[col]  # variables dummy o enteras pequeñas

# Transformaciones logarítmicas (solo para variables no negativas)
df_model['log_V'] = np.log(df_model['ValorInc_w'] + 1)
df_model['log_P'] = np.log(df_model['Total_Patentes_Inventores_w'] + 1)

# Variables centradas (evitar multicolinealidad en polinomios)
df_model['TPI_c']    = df_model['Total_Patentes_Inventores_w'] - df_model['Total_Patentes_Inventores_w'].mean()
df_model['TPI_c2']   = df_model['TPI_c'] ** 2
df_model['log_P_c']  = df_model['log_P'] - df_model['log_P'].mean()
df_model['log_P_c2'] = df_model['log_P_c'] ** 2

# Para los controles, también centramos las continuas (opcional, mejora interpretación)
for var in ['DeTech', 'Experticia', 'Inventores', 'PatenUniv', 'PatenInd']:
    df_model[f'{var}_c'] = df_model[f'{var}_w'] - df_model[f'{var}_w'].mean()

# Estadísticas descriptivas
print(f"\n  Muestra final: n = {len(df_model)}")
print("\n  Estadísticas descriptivas (variables winzorizadas):")
print(df_model[['ValorInc_w', 'Total_Patentes_Inventores_w'] +
               [f'{v}_w' for v in vars_controles if v != 'Rank']].describe().round(3).to_string())

# =============================================================================
# 2. ESTIMACIÓN DE LOS CUATRO MODELOS OLS (con controles)
# =============================================================================
print("\n" + "=" * 70)
print("2. ESTIMACIÓN DE MODELOS OLS CON CONTROLES")
print("=" * 70)

# Definir los sets de variables independientes para cada modelo
# M1: Lineal (TPI + controles)
X1_vars = ['Total_Patentes_Inventores_w'] + [f'{v}_c' for v in vars_controles if v != 'Rank'] + ['Rank']
X1 = sm.add_constant(df_model[X1_vars])
m1 = sm.OLS(df_model['ValorInc_w'], X1).fit(cov_type='HC3')

# M2: Log-Log (log_P + controles)
X2_vars = ['log_P'] + [f'{v}_c' for v in vars_controles if v != 'Rank'] + ['Rank']
X2 = sm.add_constant(df_model[X2_vars])
m2 = sm.OLS(df_model['log_V'], X2).fit(cov_type='HC3')

# M3: Polinómico original (TPI_c + TPI_c2 + controles)
X3_vars = ['TPI_c', 'TPI_c2'] + [f'{v}_c' for v in vars_controles if v != 'Rank'] + ['Rank']
X3 = sm.add_constant(df_model[X3_vars])
m3 = sm.OLS(df_model['ValorInc_w'], X3).fit(cov_type='HC3')

# M4: Polinómico Log-Log (log_P_c + log_P_c2 + controles)
X4_vars = ['log_P_c', 'log_P_c2'] + [f'{v}_c' for v in vars_controles if v != 'Rank'] + ['Rank']
X4 = sm.add_constant(df_model[X4_vars])
m4 = sm.OLS(df_model['log_V'], X4).fit(cov_type='HC3')

modelos = {
    'M1 — Lineal'             : m1,
    'M2 — Log-Log'            : m2,
    'M3 — Polinómico original': m3,
    'M4 — Polinómico Log-Log' : m4,
}

for nombre, mod in modelos.items():
    print(f"\n{'─'*60}")
    print(f"  {nombre}")
    print(f"{'─'*60}")
    print(mod.summary2(float_format="%.4f"))

# Punto de inflexión M4 (U invertida)
if 'log_P_c2' in m4.params and m4.params['log_P_c2'] < 0:
    b1 = m4.params['log_P_c']
    b2 = m4.params['log_P_c2']
    tp_c  = -b1 / (2 * b2)
    tp_lP = tp_c + df_model['log_P'].mean()
    tp_P  = np.exp(tp_lP) - 1
    print(f"\n  ▶ M4 — U invertida: máximo en log(TPI+1) = {tp_lP:.4f} "
          f"→ TPI ≈ {tp_P:.1f} patentes")
else:
    tp_P = np.nan

# =============================================================================
# 3. DIAGNÓSTICOS ESTADÍSTICOS (sobre M4)
# =============================================================================
print("\n" + "=" * 70)
print("3. DIAGNÓSTICOS ESTADÍSTICOS — M4 (Polinómico Log-Log con controles)")
print("=" * 70)

resid4  = m4.resid
fitted4 = m4.fittedvalues

bp_stat, bp_pval, _, _ = het_breuschpagan(resid4, m4.model.exog)
print(f"\n  Breusch-Pagan   LM = {bp_stat:.4f}, p = {bp_pval:.4f}")

wh_stat, wh_pval, _, _ = het_white(resid4, m4.model.exog)
print(f"  White           LM = {wh_stat:.4f}, p = {wh_pval:.4f}")

dw = durbin_watson(resid4)
print(f"  Durbin-Watson   DW = {dw:.4f}")

reset_res = linear_reset(m4, power=3, use_f=True)
print(f"  RESET           F  = {reset_res.statistic:.4f}, p = {reset_res.pvalue:.4f}")

jb_stat, jb_pval = stats.jarque_bera(resid4)
print(f"  Jarque-Bera     JB = {jb_stat:.4f}, p = {jb_pval:.4f}")

# =============================================================================
# 4. TABLA COMPARATIVA PARA PAPER
# =============================================================================
print("\n" + "=" * 70)
print("4. TABLA COMPARATIVA DE MODELOS (formato paper)")
print("=" * 70)

def sig_stars(p):
    if p < 0.01:  return '***'
    if p < 0.05:  return '**'
    if p < 0.10:  return '*'
    return ''

def fila_modelo(mod, nombre, var_principal, var_cuad=None):
    coef1  = mod.params.get(var_principal, np.nan)
    se1    = mod.bse.get(var_principal, np.nan)
    p1     = mod.pvalues.get(var_principal, np.nan)
    s1     = sig_stars(p1)
    row = {
        'Especificación'   : nombre,
        'β₁ (coef.)'       : f"{coef1:.4f}{s1}",
        'SE₁'              : f"({se1:.4f})",
        'p₁'               : f"{p1:.4f}",
    }
    if var_cuad:
        coef2 = mod.params.get(var_cuad, np.nan)
        se2   = mod.bse.get(var_cuad, np.nan)
        p2    = mod.pvalues.get(var_cuad, np.nan)
        s2    = sig_stars(p2)
        row['β₂ (cuad.)'] = f"{coef2:.4f}{s2}"
        row['SE₂']        = f"({se2:.4f})"
        row['p₂']         = f"{p2:.4f}"
    else:
        row['β₂ (cuad.)'] = '—'
        row['SE₂']        = '—'
        row['p₂']         = '—'
    row['R²']      = f"{mod.rsquared:.4f}"
    row['R² adj.'] = f"{mod.rsquared_adj:.4f}"
    row['AIC']     = f"{mod.aic:.1f}"
    row['BIC']     = f"{mod.bic:.1f}"
    row['n']       = len(mod.model.endog)
    return row

tabla_rows = [
    fila_modelo(m1, 'M1 — Lineal',              'Total_Patentes_Inventores_w'),
    fila_modelo(m2, 'M2 — Log-Log',             'log_P'),
    fila_modelo(m3, 'M3 — Polinómico original', 'TPI_c',   'TPI_c2'),
    fila_modelo(m4, 'M4 — Polinómico Log-Log',  'log_P_c', 'log_P_c2'),
]
tabla_paper = pd.DataFrame(tabla_rows)
print("\n" + tabla_paper.to_string(index=False))
print("\n  Nota: SE robustos HC3. Controles incluyen: DeTech, Experticia, Inventores, Rank, PatenUniv, PatenInd.")
print("  * p<0.10, ** p<0.05, *** p<0.01.")

# =============================================================================
#  VISUALIZACIONES — Mantenemos solo las que tienen sentido multivariado
# =============================================================================

# Grilla para efectos marginales de TPI (manteniendo controles en medias)
# Para M4 (polinómico log-log)
logP_c_grid  = np.linspace(df_model['log_P_c'].min(),
                            df_model['log_P_c'].max(), 300)
logP_grid    = logP_c_grid + df_model['log_P'].mean()
TPI_grid     = np.exp(logP_grid) - 1

# Valores medios de los controles (centrados, por lo tanto media = 0)
controles_medios = {f'{v}_c': 0 for v in vars_controles if v != 'Rank'}
controles_medios['Rank'] = df_model['Rank'].mean()  # media de la dummy

# Construir matriz de predicción para M4 (incluye const, log_P_c, log_P_c2 y controles)
X_m4_pred = pd.DataFrame({'const': 1.0,
                          'log_P_c': logP_c_grid,
                          'log_P_c2': logP_c_grid**2,
                          **controles_medios})
# Reordenar columnas igual que en el modelo
X_m4_pred = X_m4_pred[m4.model.exog_names]
logV_pred_m4 = m4.predict(X_m4_pred)
ValV_pred_m4 = np.exp(logV_pred_m4) - 1

# Intervalos de confianza para la predicción
pred_obj = m4.get_prediction(X_m4_pred)
pred_df  = pred_obj.summary_frame(alpha=0.05)
lb_m, ub_m = pred_df['mean_ci_lower'], pred_df['mean_ci_upper']

# ─────────────────────────────────────────────────────────────────
# FIGURA 1: Distribuciones originales y transformadas (2×2)
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(" Distribución de Variables — Original y Transformación Logarítmica",
             fontsize=13, fontweight='bold')

axes[0, 0].hist(df_model['Total_Patentes_Inventores_w'], bins=50,
                color=PAL['azul'], edgecolor='white', alpha=0.85)
axes[0, 0].set_title("Stock de Patentes (winsorizado)")
axes[0, 0].set_xlabel("Total_Patentes_Inventores")
axes[0, 0].set_ylabel("Frecuencia")
_med = df_model['Total_Patentes_Inventores_w'].median()
_mn  = df_model['Total_Patentes_Inventores_w'].mean()
axes[0, 0].axvline(_mn,  color=PAL['rojo'],   linestyle='--', lw=1.8,
                   label=f"Media={_mn:.0f}")
axes[0, 0].axvline(_med, color=PAL['naranja'],linestyle=':',  lw=1.8,
                   label=f"Mediana={_med:.0f}")
axes[0, 0].legend(fontsize=9)

axes[0, 1].hist(df_model['ValorInc_w'], bins=50,
                color=PAL['rojo'], edgecolor='white', alpha=0.85)
axes[0, 1].set_title("ValorInc (winsorizado)")
axes[0, 1].set_xlabel("ValorInc")
axes[0, 1].set_ylabel("Frecuencia")
axes[0, 1].axvline(df_model['ValorInc_w'].mean(),   color='black',
                   linestyle='--', lw=1.8,
                   label=f"Media={df_model['ValorInc_w'].mean():.1f}")
axes[0, 1].axvline(df_model['ValorInc_w'].median(), color=PAL['gris'],
                   linestyle=':', lw=1.8,
                   label=f"Mediana={df_model['ValorInc_w'].median():.1f}")
axes[0, 1].legend(fontsize=9)

axes[1, 0].hist(df_model['log_P'], bins=50,
                color=PAL['azul'], edgecolor='white', alpha=0.85)
axes[1, 0].set_title("log(TPI + 1)")
axes[1, 0].set_xlabel("log(Total_Patentes_Inventores + 1)")
axes[1, 0].set_ylabel("Frecuencia")
mu_lp, sd_lp = df_model['log_P'].mean(), df_model['log_P'].std()
x_n = np.linspace(df_model['log_P'].min(), df_model['log_P'].max(), 200)
y_n = stats.norm.pdf(x_n, mu_lp, sd_lp)
ax_twin = axes[1, 0].twinx()
ax_twin.plot(x_n, y_n, color=PAL['naranja'], lw=2, linestyle='--',
             label='Distribución normal')
ax_twin.set_ylabel("Densidad", color=PAL['naranja'])
ax_twin.tick_params(axis='y', colors=PAL['naranja'])
ax_twin.spines['right'].set_visible(True)

axes[1, 1].hist(df_model['log_V'], bins=50,
                color=PAL['rojo'], edgecolor='white', alpha=0.85)
axes[1, 1].set_title("log(ValorInc + 1)")
axes[1, 1].set_xlabel("log(ValorInc + 1)")
axes[1, 1].set_ylabel("Frecuencia")
mu_lv, sd_lv = df_model['log_V'].mean(), df_model['log_V'].std()
x_n2 = np.linspace(df_model['log_V'].min(), df_model['log_V'].max(), 200)
y_n2 = stats.norm.pdf(x_n2, mu_lv, sd_lv)
ax_twin2 = axes[1, 1].twinx()
ax_twin2.plot(x_n2, y_n2, color=PAL['naranja'], lw=2, linestyle='--')
ax_twin2.set_ylabel("Densidad", color=PAL['naranja'])
ax_twin2.tick_params(axis='y', colors=PAL['naranja'])
ax_twin2.spines['right'].set_visible(True)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig1_distribuciones.png")
plt.show()
print(f"✓ {OUTPUT}/fig1_distribuciones.png")

# ─────────────────────────────────────────────────────────────────
# FIGURA 2: Efecto marginal de TPI (M4) con controles en medias
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(logP_grid, logV_pred_m4, color=PAL['azul'], lw=2.5,
        label='Curva predicha')
ax.fill_between(logP_grid, lb_m, ub_m, color=PAL['azul'], alpha=0.15,
                label='IC 95 %')
if not np.isnan(tp_P):
    idx_tp = np.argmax(logV_pred_m4)
    ax.scatter([logP_grid[idx_tp]], [logV_pred_m4[idx_tp]],
               color=PAL['rojo'], s=120, zorder=5, edgecolors='black', lw=1,
               label=f'Máximo: TPI ≈ {tp_P:.0f}')
    ax.annotate(f'Óptimo\n≈ {logP_grid[idx_tp]:.2f}',
                xy=(logP_grid[idx_tp], logV_pred_m4[idx_tp]),
                xytext=(logP_grid[idx_tp] + 0.3, logV_pred_m4[idx_tp] - 0.1),
                fontsize=9, color=PAL['rojo'],
                arrowprops=dict(arrowstyle='->', color=PAL['rojo']))
ax.set_xlabel("log(Total_Patentes_Inventores + 1)")
ax.set_ylabel("log(ValorInc + 1) predicho")
ax.set_title("Efecto marginal de TPI (controles en valores medios)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig2_efecto_marginal_TPI.png")
plt.show()
print(f"✓ {OUTPUT}/fig2_efecto_marginal_TPI.png")

# ─────────────────────────────────────────────────────────────────
# FIGURA 3: Diagnóstico de residuos — M4 (3 paneles)
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle(" Diagnóstico de Residuos — M4 (Polinómico Log-Log con controles)",
             fontsize=13, fontweight='bold')

# Residuos vs ajustados
ax = axes[0]
ax.scatter(fitted4, resid4, color=PAL['azul'], alpha=0.45, s=30, edgecolors='none')
ax.axhline(0, color=PAL['rojo'], linestyle='--', lw=1.8)
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    lw4 = lowess(resid4, fitted4, frac=0.5)
    ax.plot(lw4[:, 0], lw4[:, 1], color=PAL['verde'], lw=2, label='LOESS')
    ax.legend(fontsize=9)
except Exception:
    pass
ax.set_xlabel("Valores ajustados"); ax.set_ylabel("Residuos")
ax.set_title("Residuos vs Ajustados")

# Histograma con curva normal
ax = axes[1]
ax.hist(resid4, bins=30, color=PAL['azul'], edgecolor='white', alpha=0.8,
        density=True, label='Residuos')
x_res = np.linspace(resid4.min(), resid4.max(), 200)
ax.plot(x_res, stats.norm.pdf(x_res, resid4.mean(), resid4.std()),
        color=PAL['rojo'], lw=2, label='Normal teórica')
ax.set_xlabel("Residuos"); ax.set_ylabel("Densidad")
ax.set_title("Histograma de Residuos"); ax.legend(fontsize=9)

# Q-Q plot
ax = axes[2]
(osm, osr), (slope, intercept, _) = stats.probplot(resid4, dist="norm")
ax.scatter(osm, osr, color=PAL['azul'], alpha=0.5, s=30, edgecolors='none')
x_qq = np.array([min(osm), max(osm)])
ax.plot(x_qq, slope * x_qq + intercept, color=PAL['rojo'], lw=2)
ax.set_xlabel("Cuantiles teóricos"); ax.set_ylabel("Cuantiles observados")
ax.set_title("Q-Q Plot de Residuos")

plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig3_diagnostico_residuos.png")
plt.show()
print(f"✓ {OUTPUT}/fig3_diagnostico_residuos.png")

# ─────────────────────────────────────────────────────────────────
# FIGURA 4: Dispersión de ValorInc por decil de TPI (barras)
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Distribución de ValorInc por Decil de Stock de Patentes",
             fontsize=13, fontweight='bold')

df_model['decil_TPI'] = pd.qcut(df_model['Total_Patentes_Inventores_w'], q=10,
                                  labels=[f'D{i}' for i in range(1, 11)],
                                  duplicates='drop')
est_dec = df_model.groupby('decil_TPI', observed=True)['ValorInc_w'].agg(
    ['mean', 'std', 'count', 'median'])
est_dec['se'] = est_dec['std'] / np.sqrt(est_dec['count'])

colors_dec = plt.cm.Blues(np.linspace(0.35, 0.85, len(est_dec)))

ax = axes[0]
bars = ax.bar(range(len(est_dec)), est_dec['mean'],
              yerr=est_dec['se'] * 1.96, color=colors_dec,
              edgecolor='white', capsize=4, linewidth=0.8)
ax.set_xticks(range(len(est_dec)))
ax.set_xticklabels(est_dec.index, fontsize=9)
ax.set_xlabel("Decil de Total_Patentes_Inventores")
ax.set_ylabel("Media de ValorInc (winsorizado)")
ax.set_title("Media ± IC 95% por decil")
for bar, (_, row) in zip(bars, est_dec.iterrows()):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + row['se'] * 1.96 + 0.1,
            f"n={int(row['count'])}", ha='center', fontsize=7.5, color='gray')

ax = axes[1]
ax.bar(range(len(est_dec)), est_dec['median'],
       color=colors_dec, edgecolor='white', linewidth=0.8)
ax.set_xticks(range(len(est_dec)))
ax.set_xticklabels(est_dec.index, fontsize=9)
ax.set_xlabel("Decil de Total_Patentes_Inventores")
ax.set_ylabel("Mediana de ValorInc (winsorizado)")
ax.set_title("Mediana por decil")

plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig4_barras_deciles.png")
plt.show()
print(f"✓ {OUTPUT}/fig4_barras_deciles.png")

# ─────────────────────────────────────────────────────────────────
# FIGURA 5: Panel resumen (dashboard 2×3)
# ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
fig.suptitle("Resumen General: Stock de Patentes vs Valor de Innovación (con controles)",
             fontsize=14, fontweight='bold')

# A — Histograma log_P
ax_a = fig.add_subplot(gs[0, 0])
ax_a.hist(df_model['log_P'], bins=40, color=PAL['azul'],
          edgecolor='white', alpha=0.85)
ax_a.set_title("A. log(TPI+1)")
ax_a.set_xlabel("log(TPI+1)"); ax_a.set_ylabel("Frecuencia")

# B — Histograma log_V
ax_b = fig.add_subplot(gs[0, 1])
ax_b.hist(df_model['log_V'], bins=40, color=PAL['rojo'],
          edgecolor='white', alpha=0.85)
ax_b.set_title("B. log(ValorInc+1)")
ax_b.set_xlabel("log(ValorInc+1)"); ax_b.set_ylabel("Frecuencia")

# C — Efecto marginal de TPI (curva M4)
ax_c = fig.add_subplot(gs[0, 2])
ax_c.plot(logP_grid, logV_pred_m4, color=PAL['azul'], lw=2)
ax_c.fill_between(logP_grid, lb_m, ub_m, color=PAL['azul'], alpha=0.15)
if not np.isnan(tp_P):
    ax_c.scatter([logP_grid[idx_tp]], [logV_pred_m4[idx_tp]], color='black', s=80, zorder=5)
ax_c.set_title("C. Efecto marginal de TPI (M4)")
ax_c.set_xlabel("log(TPI+1)"); ax_c.set_ylabel("log(V+1) predicho")

# D — Curva en escala original
ax_d = fig.add_subplot(gs[1, 0])
ax_d.plot(TPI_grid, ValV_pred_m4, color=PAL['verde'], lw=2)
if not np.isnan(tp_P):
    ax_d.axvline(tp_P, color=PAL['rojo'], linestyle='--', lw=1.5,
                 label=f'Óptimo≈{tp_P:.0f}')
    ax_d.legend(fontsize=8)
ax_d.set_title("D. Curva escala original")
ax_d.set_xlabel("TPI"); ax_d.set_ylabel("ValorInc predicho")

# E — Residuos M4
ax_e = fig.add_subplot(gs[1, 1])
ax_e.scatter(fitted4, resid4, color=PAL['azul'], alpha=0.4, s=18, edgecolors='none')
ax_e.axhline(0, color=PAL['rojo'], lw=1.5, linestyle='--')
ax_e.set_title("E. Residuos vs Ajustados M4")
ax_e.set_xlabel("Ajustados"); ax_e.set_ylabel("Residuos")

# F — Barras media por decil
ax_f = fig.add_subplot(gs[1, 2])
ax_f.bar(range(len(est_dec)), est_dec['mean'],
         color=plt.cm.Blues(np.linspace(0.35, 0.85, len(est_dec))),
         edgecolor='white', linewidth=0.8)
ax_f.set_xticks(range(len(est_dec)))
ax_f.set_xticklabels(est_dec.index, fontsize=7.5)
ax_f.set_title("F. Media ValorInc por decil TPI")
ax_f.set_xlabel("Decil TPI"); ax_f.set_ylabel("Media ValorInc")

plt.savefig(f"{OUTPUT}/fig5_panel_resumen.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ {OUTPUT}/fig5_panel_resumen.png")

# =============================================================================
# 5. TABLA RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 70)
print("5. TABLA RESUMEN PARA PAPER")
print("=" * 70)

print(f"""
┌──────────────────────────────────────────────────────────────────────────────┐
│     TABLA 1. Estimaciones OLS — Variable dependiente: log(ValorInc+1)       │
│     (Errores estándar robustos HC3 entre paréntesis)                         │
├──────────────────────────────────────────────────────────────────────────────┤
│ Controles en todos los modelos: DeTech, Experticia, Inventores, Rank,       │
│ PatenUniv, PatenInd (centrados).                                             │
├──────────────────────────┬──────────┬──────────┬──────────┬──────────────────┤
│ Variable principal       │  M1      │  M2      │  M3      │  M4              │
│                          │ Lineal   │ Log-Log  │ Polinóm. │ Polinóm. Log-Log │
├──────────────────────────┼──────────┼──────────┼──────────┼──────────────────┤
""")
print(f"│ R² ajustado           │ {m1.rsquared_adj:.4f}   │ {m2.rsquared_adj:.4f}   │ {m3.rsquared_adj:.4f}   │ {m4.rsquared_adj:.4f}     │")
print(f"│ AIC                   │ {m1.aic:.1f}    │ {m2.aic:.1f}    │ {m3.aic:.1f}    │ {m4.aic:.1f}      │")
print(f"│ BIC                   │ {m1.bic:.1f}    │ {m2.bic:.1f}    │ {m3.bic:.1f}    │ {m4.bic:.1f}      │")
print(f"│ n                     │ {len(df_model)}       │ {len(df_model)}       │ {len(df_model)}       │ {len(df_model)}         │")
print(f"└──────────────────────────┴──────────┴──────────┴──────────┴──────────────────┘")
if not np.isnan(tp_P):
    print(f"\n  ▶ Punto de inflexión M4: TPI ≈ {tp_P:.0f} patentes acumuladas.")
print(f"\n✅ Análisis completo con variables de control.")
print(f"📁 Figuras guardadas en: ./{OUTPUT}/")
