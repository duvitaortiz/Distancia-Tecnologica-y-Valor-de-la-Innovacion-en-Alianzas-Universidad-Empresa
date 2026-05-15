
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.diagnostic import (het_breuschpagan, het_white,
                                           linear_reset)
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

OUTPUT = "graficas_patentes"
os.makedirs(OUTPUT, exist_ok=True)

# =============================================================================
# 1. CARGA Y PREPROCESAMIENTO
# =============================================================================
print("=" * 70)
print("1. CARGA Y PREPROCESAMIENTO")
print("=" * 70)

ruta  = "/content/ruta.xlsx"
sheet = "Matriz 1_"
df    = pd.read_excel(ruta, sheet_name=sheet)

# Stock acumulado de patentes de los inventores (suma por equipo)
num_inv_cols = [f'Inventor {i}' for i in range(1, 28)]

def sumar_patentes(fila):
    n = fila['# de inventores']
    if pd.isna(n) or n <= 0:
        return np.nan
    n = int(n)
    return fila[num_inv_cols[:n]].sum(skipna=True)

df['Total_Patentes_Inventores'] = df.apply(sumar_patentes, axis=1)

# Selección y winsorización (1 % en colas)
df_model = df[['ValorInc', 'Total_Patentes_Inventores']].dropna().copy()
df_model['ValorInc_w'] = winsorize(df_model['ValorInc'],
                                    limits=(0.01, 0.01))
df_model['TPI_w']      = winsorize(df_model['Total_Patentes_Inventores'],
                                    limits=(0.01, 0.01))

# Transformaciones logarítmicas
df_model['log_V'] = np.log(df_model['ValorInc_w'] + 1)
df_model['log_P'] = np.log(df_model['TPI_w']      + 1)

# Variables centradas (evitar multicolinealidad en polinomios)
df_model['TPI_c']    = df_model['TPI_w']   - df_model['TPI_w'].mean()
df_model['TPI_c2']   = df_model['TPI_c']  ** 2
df_model['log_P_c']  = df_model['log_P']  - df_model['log_P'].mean()
df_model['log_P_c2'] = df_model['log_P_c'] ** 2

# Estadísticas descriptivas
print(f"\n  Muestra final: n = {len(df_model)}")
print("\n  Estadísticas descriptivas (variables winzorizadas):")
print(df_model[['ValorInc_w', 'TPI_w', 'log_V', 'log_P']].describe().round(3).to_string())

# =============================================================================
# 2. ESTIMACIÓN DE LOS CUATRO MODELOS OLS
# =============================================================================
print("\n" + "=" * 70)
print("2. ESTIMACIÓN DE MODELOS OLS")
print("=" * 70)

# M1 — Lineal
m1 = sm.OLS(df_model['ValorInc_w'],
            sm.add_constant(df_model['TPI_w'])).fit(cov_type='HC3')

# M2 — Log-Log simple
m2 = sm.OLS(df_model['log_V'],
            sm.add_constant(df_model['log_P'])).fit(cov_type='HC3')

# M3 — Polinómico de segundo grado (escala original)
m3 = sm.OLS(df_model['ValorInc_w'],
            sm.add_constant(df_model[['TPI_c', 'TPI_c2']])).fit(cov_type='HC3')

# M4 — Polinómico Log-Log (especificación principal)
m4 = sm.OLS(df_model['log_V'],
            sm.add_constant(df_model[['log_P_c', 'log_P_c2']])).fit(cov_type='HC3')

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
b1 = m4.params['log_P_c']
b2 = m4.params['log_P_c2']
if b2 < 0:
    tp_c  = -b1 / (2 * b2)
    tp_lP = tp_c + df_model['log_P'].mean()
    tp_P  = np.exp(tp_lP) - 1
    print(f"\n  ▶ M4 — U invertida: máximo en log(TPI+1) = {tp_lP:.4f} "
          f"→ TPI ≈ {tp_P:.1f} patentes")

# =============================================================================
# 3. DIAGNÓSTICOS ESTADÍSTICOS
# =============================================================================
print("\n" + "=" * 70)
print("3. DIAGNÓSTICOS ESTADÍSTICOS — M4 (Polinómico Log-Log)")
print("=" * 70)

resid4  = m4.resid
fitted4 = m4.fittedvalues

# Breusch-Pagan (heterocedasticidad)
bp_stat, bp_pval, _, _ = het_breuschpagan(resid4, m4.model.exog)
print(f"\n  Breusch-Pagan   LM = {bp_stat:.4f}, p = {bp_pval:.4f}")

# White test
wh_stat, wh_pval, _, _ = het_white(resid4, m4.model.exog)
print(f"  White           LM = {wh_stat:.4f}, p = {wh_pval:.4f}")

# Durbin-Watson
dw = durbin_watson(resid4)
print(f"  Durbin-Watson   DW = {dw:.4f}")

# RESET (no linealidad omitida)
reset_res = linear_reset(m4, power=3, use_f=True)
print(f"  RESET           F  = {reset_res.statistic:.4f}, p = {reset_res.pvalue:.4f}")

# Normalidad de residuos
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
    fila_modelo(m1, 'M1 — Lineal',              'TPI_w'),
    fila_modelo(m2, 'M2 — Log-Log',             'log_P'),
    fila_modelo(m3, 'M3 — Polinómico original', 'TPI_c',   'TPI_c2'),
    fila_modelo(m4, 'M4 — Polinómico Log-Log',  'log_P_c', 'log_P_c2'),
]
tabla_paper = pd.DataFrame(tabla_rows)
print("\n" + tabla_paper.to_string(index=False))
print("\n  Nota: SE robustos HC3. * p<0.10, ** p<0.05, *** p<0.01.")


# =============================================================================
#   VISUALIZACIONES — 9 FIGURAS
# =============================================================================

# Grilla de predicción compartida
logP_c_grid  = np.linspace(df_model['log_P_c'].min(),
                            df_model['log_P_c'].max(), 300)
logP_grid    = logP_c_grid + df_model['log_P'].mean()
TPI_grid     = np.exp(logP_grid) - 1

X_m4_pred    = sm.add_constant(pd.DataFrame({
    'log_P_c' : logP_c_grid,
    'log_P_c2': logP_c_grid ** 2
}))[m4.params.index]
logV_pred_m4 = m4.predict(X_m4_pred)
ValV_pred_m4 = np.exp(logV_pred_m4) - 1            # Escala original

# IC analítico para la curva
pred_obj = m4.get_prediction(X_m4_pred)
pred_df  = pred_obj.summary_frame(alpha=0.05)
lb_m, ub_m = pred_df['mean_ci_lower'], pred_df['mean_ci_upper']

# ─────────────────────────────────────────────────────────────────
# FIGURA 1: Distribuciones originales y transformadas (2×2)
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(" Distribución de Variables — Original y Transformación Logarítmica",
             fontsize=13, fontweight='bold')

axes[0, 0].hist(df_model['TPI_w'], bins=50,
                color=PAL['azul'], edgecolor='white', alpha=0.85)
axes[0, 0].set_title("Stock de Patentes (winsorizado)")
axes[0, 0].set_xlabel("Total_Patentes_Inventores")
axes[0, 0].set_ylabel("Frecuencia")
_med = df_model['TPI_w'].median()
_mn  = df_model['TPI_w'].mean()
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

# Curva normal superpuesta
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
# FIGURA 2: Dispersión original con ajuste lineal (M1)
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(" Ajuste Lineal — Escala Original (M1)",
             fontsize=13, fontweight='bold')

# 2a. Scatter + línea
ax = axes[0]
ax.scatter(df_model['TPI_w'], df_model['ValorInc_w'],
           color=PAL['azul'], alpha=0.45, s=35, edgecolors='none',
           label='Observaciones')
x_rng = np.linspace(df_model['TPI_w'].min(), df_model['TPI_w'].max(), 200)
Xp = sm.add_constant(x_rng)
yp = m1.predict(Xp)
# IC 95 % para la recta
pred_m1 = m1.get_prediction(Xp).summary_frame(alpha=0.05)
ax.plot(x_rng, yp, color=PAL['rojo'], lw=2.5,
        label=f'Ajuste lineal (R²={m1.rsquared:.3f})')
ax.fill_between(x_rng, pred_m1['mean_ci_lower'], pred_m1['mean_ci_upper'],
                color=PAL['rojo'], alpha=0.12, label='IC 95 %')
ax.set_xlabel("Total_Patentes_Inventores (winsorizado)")
ax.set_ylabel("ValorInc (winsorizado)")
ax.set_title("Dispersión y ajuste lineal")
ax.legend(fontsize=9)

# 2b. Residuos vs ajustados M1
ax = axes[1]
ax.scatter(m1.fittedvalues, m1.resid,
           color=PAL['azul'], alpha=0.45, s=35, edgecolors='none')
ax.axhline(0, color=PAL['rojo'], lw=1.8, linestyle='--')
# Línea LOESS suavizada
try:
    from statsmodels.nonparametric.smoothers_lowess import lowess
    lw_fit = lowess(m1.resid, m1.fittedvalues, frac=0.5)
    ax.plot(lw_fit[:, 0], lw_fit[:, 1], color=PAL['verde'], lw=2, label='LOESS')
    ax.legend(fontsize=9)
except Exception:
    pass
ax.set_xlabel("Valores ajustados")
ax.set_ylabel("Residuos")
ax.set_title("Residuos vs Ajustados — M1")

plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig2_lineal_m1.png")
plt.show()
print(f"✓ {OUTPUT}/fig2_lineal_m1.png")

# ─────────────────────────────────────────────────────────────────
# FIGURA 3: Ajuste Log-Log simple (M2)
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df_model['log_P'], df_model['log_V'],
           color=PAL['azul'], alpha=0.4, s=35, edgecolors='none',
           label='Observaciones transformadas')

Xp2  = sm.add_constant(logP_grid)
yp2  = m2.predict(Xp2)
pred2 = m2.get_prediction(Xp2).summary_frame(alpha=0.05)
ax.plot(logP_grid, yp2, color=PAL['rojo'], lw=2.5,
        label=f'Log-Log lineal (R²={m2.rsquared:.3f})')
ax.fill_between(logP_grid, pred2['mean_ci_lower'], pred2['mean_ci_upper'],
                color=PAL['rojo'], alpha=0.12, label='IC 95 %')

ax.set_xlabel("log(Total_Patentes_Inventores + 1)")
ax.set_ylabel("log(ValorInc + 1)")
ax.set_title("Ajuste Log-Log Simple (M2)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig3_loglog_m2.png")
plt.show()
print(f"✓ {OUTPUT}/fig3_loglog_m2.png")

# ─────────────────────────────────────────────────────────────────
# FIGURA 4: Curva de U invertida — M4 (sin datos / con datos)
#           Separadas como dos paneles independientes
# ─────────────────────────────────────────────────────────────────
# ── 4a: Curva sola ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(logP_grid, logV_pred_m4, color=PAL['azul'], lw=2.5)
ax.fill_between(logP_grid, lb_m, ub_m, color=PAL['azul'], alpha=0.15,
                label='IC 95 %')

# Marcar el máximo
idx_tp = np.argmax(logV_pred_m4)
ax.scatter([logP_grid[idx_tp]], [logV_pred_m4[idx_tp]],
           color=PAL['rojo'], s=120, zorder=5, edgecolors='black', lw=1)
ax.annotate(f'Máximo\n≈ {logP_grid[idx_tp]:.2f}',
            xy=(logP_grid[idx_tp], logV_pred_m4[idx_tp]),
            xytext=(logP_grid[idx_tp] + 0.6, logV_pred_m4[idx_tp] - 0.1),
            fontsize=9, color=PAL['rojo'],
            arrowprops=dict(arrowstyle='->', color=PAL['rojo']))

ax.set_xlabel("log(Total_Patentes_Inventores + 1)")
ax.set_ylabel("log(ValorInc + 1) predicho")
ax.set_title(" Curva de U Invertida — M4 (sin puntos)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig4a_curva_sola_m4.png")
plt.show()
print(f"✓ {OUTPUT}/fig4a_curva_sola_m4.png")

# ── 4b: Curva con datos ───────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(df_model['log_P'], df_model['log_V'],
           color=PAL['azul'], alpha=0.38, s=32, edgecolors='none',
           label='Observaciones')
ax.plot(logP_grid, logV_pred_m4, color=PAL['rojo'], lw=2.5,
        label=f'M4 Polinómico (R²={m4.rsquared:.3f})')
ax.fill_between(logP_grid, lb_m, ub_m, color=PAL['rojo'], alpha=0.12,
                label='IC 95 %')
ax.scatter([logP_grid[idx_tp]], [logV_pred_m4[idx_tp]],
           color='black', s=100, zorder=5,
           label=f'Óptimo ≈ {np.exp(logP_grid[idx_tp]) - 1:.0f} patentes')

ax.set_xlabel("log(Total_Patentes_Inventores + 1)")
ax.set_ylabel("log(ValorInc + 1)")
ax.set_title(" Curva de U Invertida — M4 con Datos Observados")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig4b_curva_datos_m4.png")
plt.show()
print(f"✓ {OUTPUT}/fig4b_curva_datos_m4.png")

# ─────────────────────────────────────────────────────────────────
# FIGURA 5: Curva en escala ORIGINAL (retro-transformada)
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(" Curva Predicha en Escala Original (M4 retro-transformado)",
             fontsize=13, fontweight='bold')

ax = axes[0]
ax.plot(TPI_grid, ValV_pred_m4, color=PAL['verde'], lw=2.5)
ax.fill_between(TPI_grid,
                np.exp(lb_m) - 1, np.exp(ub_m) - 1,
                color=PAL['verde'], alpha=0.15, label='IC 95 %')
tp_tpi = np.exp(logP_grid[idx_tp]) - 1
ax.axvline(tp_tpi, color=PAL['rojo'], linestyle='--', lw=1.5,
           label=f'Óptimo ≈ {tp_tpi:.0f}')
ax.set_xlabel("Total_Patentes_Inventores")
ax.set_ylabel("ValorInc predicho")
ax.set_title("Escala original — curva completa")
ax.legend(fontsize=9)

ax = axes[1]
# Recortar a rango percentil 5-95 para mejor visualización
p5  = df_model['TPI_w'].quantile(0.05)
p95 = df_model['TPI_w'].quantile(0.95)
mask_zoom = (TPI_grid >= p5) & (TPI_grid <= p95)
ax.scatter(df_model['TPI_w'], df_model['ValorInc_w'],
           color=PAL['azul'], alpha=0.4, s=30, edgecolors='none',
           label='Observaciones')
ax.plot(TPI_grid[mask_zoom], ValV_pred_m4[mask_zoom],
        color=PAL['rojo'], lw=2.5, label='Curva predicha')
ax.set_xlabel("Total_Patentes_Inventores (winsorizado)")
ax.set_ylabel("ValorInc (winsorizado)")
ax.set_title("Con datos observados (zoom P5-P95)")
ax.legend(fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig5_escala_original.png")
plt.show()
print(f"✓ {OUTPUT}/fig5_escala_original.png")

# ─────────────────────────────────────────────────────────────────
# FIGURA 6: Diagnóstico de residuos — M4 (3 paneles)
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle(" Diagnóstico de Residuos — M4 (Polinómico Log-Log)",
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
plt.savefig(f"{OUTPUT}/fig6_diagnostico_residuos.png")
plt.show()
print(f"✓ {OUTPUT}/fig6_diagnostico_residuos.png")

# ─────────────────────────────────────────────────────────────────
# FIGURA 7: Comparación de curvas predichas — 4 modelos
# ─────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
fig.suptitle("Comparación de Curvas Predichas — 4 Especificaciones",
             fontsize=13, fontweight='bold')

# M1 en log-log (para comparar en misma escala)
log_P_c_g2 = logP_grid - df_model['log_P'].mean()

Xp1_ll = sm.add_constant(logP_grid)
yp1_ll = m1.predict(Xp1_ll)  # Escala original → no comparable directo

Xp3_orig = sm.add_constant(pd.DataFrame({
    'TPI_c' : TPI_grid - df_model['TPI_w'].mean(),
    'TPI_c2': (TPI_grid - df_model['TPI_w'].mean()) ** 2
}))[m3.params.index]
yp3_log = np.log(m3.predict(Xp3_orig).clip(0) + 1)  # retransf. para comparar

ax.plot(logP_grid, m2.predict(sm.add_constant(logP_grid)),
        color=PAL['naranja'], lw=2, linestyle='-.',
        label=f'M2 Log-Log (R²={m2.rsquared:.3f})')
ax.plot(logP_grid, logV_pred_m4,
        color=PAL['azul'], lw=2.5, linestyle='-',
        label=f'M4 Polinómico Log-Log (R²={m4.rsquared:.3f})')
ax.fill_between(logP_grid, lb_m, ub_m, color=PAL['azul'], alpha=0.12)

ax.scatter([logP_grid[idx_tp]], [logV_pred_m4[idx_tp]],
           color=PAL['rojo'], s=120, zorder=5, edgecolors='black', lw=1,
           label=f'Óptimo M4 ≈ {tp_tpi:.0f} patentes')

ax.set_xlabel("log(Total_Patentes_Inventores + 1)")
ax.set_ylabel("log(ValorInc + 1) predicho")
ax.set_title("")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTPUT}/fig7_comparacion_modelos.png")
plt.show()
print(f"✓ {OUTPUT}/fig7_comparacion_modelos.png")

# ─────────────────────────────────────────────────────────────────
# FIGURA 8: Dispersión de ValorInc por decil de TPI (barras)
# ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Distribución de ValorInc por Decil de Stock de Patentes",
             fontsize=13, fontweight='bold')

df_model['decil_TPI'] = pd.qcut(df_model['TPI_w'], q=10,
                                  labels=[f'D{i}' for i in range(1, 11)],
                                  duplicates='drop')
n_deciles = df_model['decil_TPI'].nunique()

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
plt.savefig(f"{OUTPUT}/fig8_barras_deciles.png")
plt.show()
print(f"✓ {OUTPUT}/fig8_barras_deciles.png")

# ─────────────────────────────────────────────────────────────────
# FIGURA 9: Panel resumen (dashboard 2×3)
# ─────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
fig.suptitle("Resumen General: Stock de Patentes vs Valor de Innovación",
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

# C — Scatter log-log + M4
ax_c = fig.add_subplot(gs[0, 2])
ax_c.scatter(df_model['log_P'], df_model['log_V'],
             color=PAL['azul'], alpha=0.35, s=20, edgecolors='none')
ax_c.plot(logP_grid, logV_pred_m4, color=PAL['rojo'], lw=2)
ax_c.scatter([logP_grid[idx_tp]], [logV_pred_m4[idx_tp]],
             color='black', s=80, zorder=5)
ax_c.set_title("C. Ajuste M4 (U invertida)")
ax_c.set_xlabel("log(TPI+1)"); ax_c.set_ylabel("log(V+1)")

# D — Curva escala original
ax_d = fig.add_subplot(gs[1, 0])
ax_d.plot(TPI_grid, ValV_pred_m4, color=PAL['verde'], lw=2)
ax_d.axvline(tp_tpi, color=PAL['rojo'], linestyle='--', lw=1.5,
             label=f'Óptimo≈{tp_tpi:.0f}')
ax_d.legend(fontsize=8)
ax_d.set_title("D. Curva escala original")
ax_d.set_xlabel("TPI"); ax_d.set_ylabel("ValorInc predicho")

# E — Residuos M4
ax_e = fig.add_subplot(gs[1, 1])
ax_e.scatter(fitted4, resid4, color=PAL['azul'], alpha=0.4,
             s=18, edgecolors='none')
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

plt.savefig(f"{OUTPUT}/fig9_panel_resumen.png", dpi=150, bbox_inches='tight')
plt.show()
print(f"✓ {OUTPUT}/fig9_panel_resumen.png")

# =============================================================================
# 5. TABLA RESUMEN FINAL PARA PAPER (impresión limpia)
# =============================================================================
print("\n" + "=" * 70)
print("5. TABLA RESUMEN PARA PAPER")
print("=" * 70)

print("""
┌──────────────────────────────────────────────────────────────────────────────┐
│          TABLA 1. Estimaciones OLS — Variable dependiente: log(ValorInc+1)  │
│          (Errores estándar robustos HC3 entre paréntesis)                   │
├──────────────────────────┬──────────┬──────────┬──────────┬──────────────────┤
│ Variable                 │  M1      │  M2      │  M3      │  M4              │
│                          │ Lineal   │ Log-Log  │ Polinóm. │ Polinóm. Log-Log │
├──────────────────────────┼──────────┼──────────┼──────────┼──────────────────┤""")

for nombre, mod in modelos.items():
    params_str = " | ".join(
        [f"{v}: {mod.params.get(v, np.nan):.4f}{sig_stars(mod.pvalues.get(v, 1))}"
         for v in mod.params.index if v != 'const']
    )
    print(f"│ {nombre:<24} | R²={mod.rsquared:.4f} | AIC={mod.aic:.1f} │")

print(f"""└──────────────────────────────────────────────────────────────────────────────┘
  Nota: * p<0.10, ** p<0.05, *** p<0.01.  Errores estándar HC3 robustos.
  n = {len(df_model)}.  Datos winsorizados al 1% en ambas colas.
""")

print(f"\n✅ Análisis completo.")
print(f"   R² M1={m1.rsquared:.4f} | M2={m2.rsquared:.4f} "
      f"| M3={m3.rsquared:.4f} | M4={m4.rsquared:.4f}")
print(f"   Óptimo (M4): TPI ≈ {tp_tpi:.0f} patentes acumuladas")
print(f"📁 Figuras guardadas en: ./{OUTPUT}/")
