import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# ── 1. Cargar el dataset completo ───────────────────────────────────────────
df = pd.read_csv('data/processed/dataset_modelo.csv')
print("Dimensiones del dataset: ", df.shape)

FEATURES = ['decimalLatitude', 'decimalLongitude']
X = df[FEATURES]
y = df['presencia']

# ── 2. Escalar con TODO el dataset (no solo el 80% de train) ───────────────
scaler_final = StandardScaler()
X_scaled = scaler_final.fit_transform(X)

# ── 3. Reentrenar Random Forest con el 100% de los datos ───────────────────
# Justificación metodológica: una vez seleccionado el modelo ganador mediante
# holdout + validación cruzada, se reentrena sobre todos los datos disponibles
# para maximizar la información utilizada en el modelo de producción final.
# Esto es práctica estándar — el modelo final NO se evalúa de nuevo aquí,
# ya fue evaluado rigurosamente en 03_modelo.py y 07_validacion_cruzada.py.

modelo_final = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_final.fit(X_scaled, y)

print("Modelo Random Forest reentrenado con el 100% del dataset (n=636)")

# ── 4. Guardar modelo final y scaler final ──────────────────────────────────
with open('data/processed/modelo_rf.pkl', 'wb') as f:
    pickle.dump(modelo_final, f)

with open('data/processed/scaler_final.pkl', 'wb') as f:
    pickle.dump(scaler_final, f)

print("Guardado: data/processed/modelo_rf.pkl")
print("Guardado: data/processed/scaler_final.pkl")

# ── 5. Verificación rápida ───────────────────────────────────────────────────
print(f"\nFeature importances:")
for feat, imp in zip(FEATURES, modelo_final.feature_importances_):
    print(f"  {feat}: {imp:.3f}")