import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# ── Cargar modelo final y su scaler correspondiente ─────────────────────────
with open('data/processed/modelo_rf.pkl', 'rb') as f:
    modelo = pickle.load(f)

with open('data/processed/scaler_final.pkl', 'rb') as f:
    scaler_final = pickle.load(f)

felinos = pd.read_csv('data/processed/felinos_limpio.csv')
print("Dimensiones del Dataframe para el mapa: ", felinos.shape)

# ── Límites del área de estudio ──────────────────────────────────────────────
lat_min = felinos['decimalLatitude'].min()
lat_max = felinos['decimalLatitude'].max()
lon_min = felinos['decimalLongitude'].min()
lon_max = felinos['decimalLongitude'].max()

# ── Crear grilla ─────────────────────────────────────────────────────────────
lats = np.arange(lat_min, lat_max, 0.02)
lons = np.arange(lon_min, lon_max, 0.02)
lon_grid, lat_grid = np.meshgrid(lons, lats)
lon_grid_flat = lon_grid.ravel()
lat_grid_flat = lat_grid.ravel()
print("Número de puntos en la grilla: ", len(lon_grid_flat))

df_grilla = pd.DataFrame({
    'decimalLatitude': lat_grid_flat,
    'decimalLongitude': lon_grid_flat
})

# ── ESCALAR la grilla con el MISMO scaler usado para entrenar el modelo ────
# Antes: df_grilla_valores = df_grilla.values (sin escalar) — esto ya no aplica
# porque el modelo ahora espera datos en la misma escala con la que se entrenó.
df_grilla_scaled = scaler_final.transform(df_grilla[['decimalLatitude', 'decimalLongitude']])

# ── Predecir probabilidades ──────────────────────────────────────────────────
probabilidades = modelo.predict_proba(df_grilla_scaled)[:, 1]
prob_grid = probabilidades.reshape(lat_grid.shape)

print("Min probabilidad:", probabilidades.min())
print("Max probabilidad:", probabilidades.max())

# ── Visualizar: mapa de calor de probabilidad + puntos de presencia ────────
puma = felinos[felinos['species'] == 'Puma concolor']
jaguar = felinos[felinos['species'] == 'Panthera onca']

fig, ax = plt.subplots(figsize=(13, 9))

# Mapa de calor de probabilidad de presencia
heat = ax.contourf(lon_grid, lat_grid, prob_grid, levels=20, cmap='YlOrRd', alpha=0.7)
cbar = plt.colorbar(heat, ax=ax)
cbar.set_label('Probabilidad de presencia (Random Forest)', fontsize=10)

# Puntos de presencia real
ax.scatter(puma['decimalLongitude'], puma['decimalLatitude'],
           c='blue', marker='o', s=60, label=f'Puma concolor ({len(puma)})',
           alpha=0.8, edgecolors='darkblue', zorder=5)
ax.scatter(jaguar['decimalLongitude'], jaguar['decimalLatitude'],
           c='black', marker='*', s=130, label=f'Panthera onca ({len(jaguar)})',
           alpha=0.9, edgecolors='white', zorder=5)

ax.set_title('Distribución potencial — Modelo final Random Forest (reentrenado 100% datos)\n'
             'Llanos Orientales, Colombia 2011',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Longitud')
ax.set_ylabel('Latitud')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('outputs/mapa_distribucion_potencial_final.png', dpi=150, bbox_inches='tight')
plt.show()
print("Guardado: outputs/mapa_distribucion_potencial_final.png")

# ── Solapamiento (se mantiene igual, no depende del modelo) ────────────────
DISTANCIA = 0.05
solapamiento = 0
for _, j in jaguar.iterrows():
    for _, p in puma.iterrows():
        dist = np.sqrt((j['decimalLatitude'] - p['decimalLatitude'])**2 +
                       (j['decimalLongitude'] - p['decimalLongitude'])**2)
        if dist < DISTANCIA:
            solapamiento += 1
            break

pct = (solapamiento / len(jaguar)) * 100
print(f"Jaguares con pumas cercanos: {solapamiento} de {len(jaguar)} ({pct:.1f}%)")