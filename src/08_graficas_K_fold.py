"""
08_grafica_cv.py
Gráfica comparativa de métricas con barras de error (validación cruzada k=5)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Datos de la validación cruzada (de 07_validacion_cruzada.py) ───────────
modelos = ['Logistic\nRegression', 'Random\nForest', 'Gradient\nBoosting', 'Árbol\nDecisión']
metricas = ['Accuracy', 'AUC', 'F1', 'Precision', 'Recall']

# Medias
medias = {
    'Logistic Regression': [0.662, 0.582, 0.000, 0.000, 0.000],
    'Random Forest':       [0.877, 0.932, 0.821, 0.798, 0.849],
    'Gradient Boosting':   [0.824, 0.895, 0.742, 0.731, 0.760],
    'Árbol Decisión':      [0.833, 0.832, 0.769, 0.719, 0.830],
}

# Desviaciones estándar
stds = {
    'Logistic Regression': [0.005, 0.060, 0.000, 0.000, 0.000],
    'Random Forest':       [0.019, 0.013, 0.031, 0.029, 0.062],
    'Gradient Boosting':   [0.030, 0.018, 0.043, 0.058, 0.073],
    'Árbol Decisión':      [0.021, 0.015, 0.022, 0.044, 0.028],
}

colores = ['#3266ad', '#2a9d6e', '#e07b2a', '#8b5cf6']

# ── Gráfica de barras agrupadas con error bars ──────────────────────────────
fig, ax = plt.subplots(figsize=(13, 7))

x = np.arange(len(metricas))
ancho = 0.2

for i, (modelo, color) in enumerate(zip(medias.keys(), colores)):
    offset = (i - 1.5) * ancho
    valores = medias[modelo]
    errores = stds[modelo]

    ax.bar(
        x + offset, valores, ancho,
        yerr=errores, capsize=4,
        label=modelos[i], color=color, alpha=0.85,
        error_kw={'elinewidth': 1.2, 'ecolor': 'black', 'alpha': 0.6}
    )

ax.set_title('Validación cruzada estratificada (K=5) — Media ± Desviación estándar',
              fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metricas, fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_ylabel('Valor', fontsize=11)
ax.axhline(0.8, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax.legend(loc='upper left', fontsize=9, ncol=2)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/cv_barras_error.png', dpi=150, bbox_inches='tight')
plt.show()
print("Guardado: outputs/cv_barras_error.png")