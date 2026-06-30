import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

# ── 1. Cargar el dataset ya procesado ───────────────────────────────────────
df = pd.read_csv('data/processed/dataset_modelo.csv')
print("Dimensiones del dataset: ", df.shape)

# ── 2. Definir features y target (mismas que en 03_modelo.py) ──────────────
FEATURES = ['decimalLatitude', 'decimalLongitude']

X = df[FEATURES]
y = df['presencia']

print("Dimensiones de X: ", X.shape)
print("Dimensiones de y: ", y.shape)
print("Distribución de clases:\n", y.value_counts())

# ── 3. Escalar features (mismo StandardScaler del pipeline original) ───────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── 4. Definir los 4 modelos con los mismos hiperparámetros del proyecto ───
modelos = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Árbol de Decisión':   DecisionTreeClassifier(random_state=42)
}

# ── 5. Configurar K-Fold estratificado (k=5) ────────────────────────────────
# Estratificado: mantiene la proporción de clases (1 presencia : 2 pseudoausencias)
# en cada uno de los 5 folds, evitando folds con muy pocas presencias por azar.
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── 6. Métricas a calcular en cada fold ─────────────────────────────────────
scoring = {
    'accuracy':  'accuracy',
    'auc':       'roc_auc',
    'f1':        'f1',
    'precision': 'precision',
    'recall':    'recall',
}

# ── 7. Ejecutar validación cruzada para cada modelo ─────────────────────────
resultados_cv = {}

print("\n" + "=" * 65)
print("VALIDACIÓN CRUZADA ESTRATIFICADA (K=5)")
print("=" * 65)

for nombre, modelo in modelos.items():
    scores = cross_validate(
        modelo, X_scaled, y,
        cv=skf,
        scoring=scoring,
        return_train_score=False
    )

    resultados_cv[nombre] = {}
    print(f"\n{nombre}")
    print("-" * 40)

    for metric in scoring:
        mean = scores[f'test_{metric}'].mean()
        std  = scores[f'test_{metric}'].std()
        resultados_cv[nombre][f'{metric}_mean'] = round(mean, 3)
        resultados_cv[nombre][f'{metric}_std']  = round(std, 3)
        print(f"  {metric:10s}: {mean:.3f} ± {std:.3f}")

# ── 8. Consolidar resultados en un DataFrame legible ────────────────────────
filas = []
for nombre, metricas in resultados_cv.items():
    fila = {'Modelo': nombre}
    for metric in scoring:
        fila[metric.capitalize()] = f"{metricas[f'{metric}_mean']:.3f} ± {metricas[f'{metric}_std']:.3f}"
    filas.append(fila)

df_resultados = pd.DataFrame(filas)

print("\n" + "=" * 65)
print("RESUMEN COMPARATIVO — VALIDACIÓN CRUZADA (K=5)")
print("=" * 65)
print(df_resultados.to_string(index=False))

# ── 9. Guardar resultados ────────────────────────────────────────────────────
df_resultados.to_csv('outputs/cv_resultados.csv', index=False)
print("\nGuardado: outputs/cv_resultados.csv")

# ── 10. Comparación rápida: holdout (train/test único) vs cross-validation ──
print("\n" + "=" * 65)
print("NOTA METODOLÓGICA")
print("=" * 65)
print("""
El pipeline original (03_modelo.py) evaluó cada modelo sobre una única
partición train/test (80/20, random_state=42). Esta validación cruzada
estratificada con k=5 promedia el desempeño sobre 5 particiones distintas,
ofreciendo una estimación más robusta y con medida de variabilidad (±std)
que reduce la dependencia de una sola muestra aleatoria de test.
""")