import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score, 
                             f1_score, confusion_matrix,
                             matthews_corrcoef, precision_score, 
                             recall_score)
from sklearn.preprocessing import StandardScaler
import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree




df = pd.read_csv(r"C:\\Users\\diego\\Desktop\\Maestria\\Aprendizaje automatico\\Proyecto Puma-Jaguar\\data\\processed\\dataset_modelo.csv")
print("Dimensiones del Dataframe para el modelo: ", df.shape)


#Seleccionar las columas para usar en los tres modelos. por convención se llaman FEATURES
FEATURES = ['decimalLatitude', 'decimalLongitude']

#diligenciar valores nulos en las columnas seleccionadas para el modelo, y eliminar las filas con valores nulos, para tener un dataframe limpio para el analisis de SDM posteriormente.
# Normalizar las features, para que el modelo de SDM pueda trabajar con ellas, ya que algunas de las features tienen diferentes escalas, y esto puede afectar el rendimiento del modelo.
df = df.fillna(df.median(numeric_only=True))

# separar en X y y, donde X son las features y y es la variable objetivo (presencia/ausencia) Variable independfie: x, variable dependiente o de respuesta es y
X = df[FEATURES]
y = df['presencia']

print("Dimensiones de X: ", X.shape)
print("Dimensiones de y: ", y.shape)

scaler = StandardScaler() # Convierte todas las features para que tengan media 0 y desviacion estandar de 1, así, ninguna variable es dominante sobre otra
X_scaled = scaler.fit_transform(X)


#Dividir el dataset en entrenamiento y prueba, para evaluar el rendimiento del modelo posteriormente.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42) # 0.2 significa que con el 80% de los datos se hará el entrenamientno y con el 20% de ellos, se hará la pruea. el 42 es una convencion para reproducir el experimetno, pero se puede cambiar


print("Dimensiones de X_train: ", X_train.shape) # 649 filas equivale al 80%
print("Dimensiones de X_test: ", X_test.shape)  #163 filas equivale al 20% de los datos para la prueba

#Aplicar modelos
#Crear diccionario de modelos, para facilitar la iteración y evaluación de cada modelo posteriormente. ponerlos en diccionario nos facilita trabajar con un ciclo for y ahorrar la escritura de codigos
##Regresión logistica, es un modelo de clasificación que se utiliza para predecir la probabilidad de que una instancia pertenezca a una clase, en este caso, presencia o ausencia de la especie.
##Random Forest, es un modelo de clasificación que se basa en la construcción de múltiples árboles de decisión durante el entrenamiento, y la salida de la clase es la moda de las clases de los árboles individuales. Es un modelo robusto que puede manejar datos con muchas características y es menos propenso al sobreajuste. Es estandas en los SDM
##Gradient Boosting, es un modelo de clasificación que se basa en la construcción de múltiples árboles de decisión durante el entrenamiento, pero a diferencia de Random Forest, los árboles se construyen secuencialmente, donde cada árbol intenta corregir los errores del árbol anterior. Es un modelo potente que puede manejar datos con muchas características y es menos propenso al sobreajuste.
##Árbol de decisión: es un modelo de clasificación que se basa en la construcción de un árbol de decisión durante el entrenamiento, donde cada nodo del árbol representa una característica y cada rama representa una decisión basada en esa característica. Es un modelo interpretable que puede manejar datos con muchas características, pero es propenso al sobreajuste si no se poda adecuadamente.
##Cada modelo es mas robusto que el anterior
modelos = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'Árbol de Decisión': DecisionTreeClassifier(random_state=42)
    
}

#Correr los modelos y evaluar su rendimiento, utilizando las métricas de accuracy y AUC-ROC, que son métricas comunes para evaluar modelos de clasificación. Accuracy mide la proporción de predicciones correctas, mientras que AUC-ROC mide la capacidad del modelo para distinguir entre clases (presencia y ausencia) a través de diferentes umbrales de clasificación.

resultados = {}

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred  = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    accuracy  = accuracy_score(y_test, y_pred)
    auc       = roc_auc_score(y_test, y_proba)
    f1        = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    mcc       = matthews_corrcoef(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    sensibilidad = tp / (tp + fn) if (tp + fn) > 0 else 0
    especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0
    tss = sensibilidad + especificidad - 1

    resultados[nombre] = {
        'accuracy': accuracy, 'auc': auc, 'f1': f1,
        'precision': precision, 'recall': recall,
        'mcc': mcc, 'tss': tss
    }

    print(f"\n{nombre}")
    print(f"  Accuracy  : {accuracy:.3f}")
    print(f"  AUC-ROC   : {auc:.3f}")
    print(f"  F1-Score  : {f1:.3f}")
    print(f"  Precision : {precision:.3f}")
    print(f"  Recall    : {recall:.3f}")
    print(f"  MCC       : {mcc:.3f}")
    print(f"  TSS       : {tss:.3f}")


#Ver cual es el mejor modelo, que es el que tiene el mayor AUC-ROC, para usarlo posteriormente en el análisis de SDM.
mejor_modelo_nombre = max(resultados, key=lambda x: resultados[x]['auc'])   
# max( resultados) Busca el maximo numero deltro del diccionario en resultados
# Key Le dice a max con que criterio comparar los valores del diccionario, en este caso, se le dice que compare los valores de 'auc' de cada modelo, para determinar cual es el mejor modelo.
# Lambda significa que se aplica una función anónima a cada elemento del diccionario para obtener el valor de 'auc', es decir que x es cada clave del diccionario
# resultados [x] accede al valor del diccionario para la clave x, que es el nombre del modelo, y luego ['auc'] accede al valor de 'auc' dentro de ese diccionario, que es la métrica que se está utilizando para comparar los modelos. En resumen, esta línea

mejor_modelo = modelos[mejor_modelo_nombre]
print(f"\nEl mejor modelo es: {mejor_modelo_nombre} con AUC-ROC de {resultados[mejor_modelo_nombre]['auc']:.3f}")

#Guardar el mejor modelo en un archivo pickle, para luego usarlo en el análisis de SDM posteriormente.
with open ('data/processed/modelo_rf.pkl', 'wb') as f:
    pickle.dump(mejor_modelo, f) 
print("Modelo guardado exitosamente")

import pandas as pd

print("\n" + "="*65)
print("  RESUMEN COMPARATIVO DE MODELOS")
print("="*65)

df_resultados = pd.DataFrame(resultados).T
df_resultados = df_resultados.round(3)
print(df_resultados.to_string())
print("="*65)
print(f"\nMejor modelo: {mejor_modelo_nombre}")
print("Criterio de selección: mayor AUC-ROC")


#Graficas

# ── Entrenar todos los modelos ─────────────────────────────────────────────
modelos_entrenados = {}
for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    modelos_entrenados[nombre] = modelo
    print(f"✓ {nombre} entrenado")

# ══════════════════════════════════════════════════════════════════════════
# MATRICES DE CONFUSIÓN
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('Matrices de confusión — Pumas y Jaguares (GBIF 2011)', fontsize=14, fontweight='bold')

colores_cm = ['Blues', 'Greens', 'Oranges', 'Purples']
etiquetas  = [['VN', 'FP'], ['FN', 'VP']]

for ax, (nombre, modelo), cmap in zip(axes.flatten(), modelos_entrenados.items(), colores_cm):
    y_pred = modelo.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)

    sns.heatmap(
        cm, annot=True, fmt='d', cmap=cmap, ax=ax,
        xticklabels=['Ausencia', 'Presencia'],
        yticklabels=['Ausencia', 'Presencia'],
        linewidths=0.5, linecolor='white', cbar=False,
        annot_kws={'size': 14, 'weight': 'bold'}
    )

    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.78, etiquetas[i][j],
                    ha='center', va='center',
                    fontsize=8, color='gray', style='italic')

    vn, fp, fn, vp = cm.ravel()
    ax.set_title(f'{nombre}\nVP={vp}  VN={vn}  FP={fp}  FN={fn}', fontsize=10, fontweight='bold')
    ax.set_xlabel('Predicho',  fontsize=9)
    ax.set_ylabel('Real',      fontsize=9)

plt.tight_layout()
plt.savefig('outputs/matrices_confusion.png', dpi=150, bbox_inches='tight')
plt.show()
print("Guardado: outputs/matrices_confusion.png")


# ══════════════════════════════════════════════════════════════════════════
# PAIR PLOT ESPACIAL — Predicciones por modelo
# ══════════════════════════════════════════════════════════════════════════

# X_test está escalado — recuperar coordenadas originales
X_test_orig = scaler.inverse_transform(X_test)
X_test_df   = pd.DataFrame(X_test_orig, columns=FEATURES)
X_test_df['real'] = y_test.values

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Distribución espacial de predicciones — Pumas y Jaguares',
             fontsize=14, fontweight='bold')

for ax, (nombre, modelo) in zip(axes.flatten(), modelos_entrenados.items()):
    y_pred = modelo.predict(X_test)
    X_test_df['prediccion'] = y_pred

    vp = (X_test_df['real'] == 1) & (X_test_df['prediccion'] == 1)
    vn = (X_test_df['real'] == 0) & (X_test_df['prediccion'] == 0)
    fp = (X_test_df['real'] == 0) & (X_test_df['prediccion'] == 1)
    fn = (X_test_df['real'] == 1) & (X_test_df['prediccion'] == 0)

    ax.scatter(X_test_df.loc[vn, 'decimalLongitude'], X_test_df.loc[vn, 'decimalLatitude'],
               c='#b0c4de', s=25, alpha=0.6, label=f'VN ({vn.sum()})', marker='o')
    ax.scatter(X_test_df.loc[vp, 'decimalLongitude'], X_test_df.loc[vp, 'decimalLatitude'],
               c='#2a9d6e', s=60, alpha=0.9, label=f'VP ({vp.sum()})', marker='o')
    ax.scatter(X_test_df.loc[fp, 'decimalLongitude'], X_test_df.loc[fp, 'decimalLatitude'],
               c='#e07b2a', s=60, alpha=0.9, label=f'FP ({fp.sum()})', marker='^')
    ax.scatter(X_test_df.loc[fn, 'decimalLongitude'], X_test_df.loc[fn, 'decimalLatitude'],
               c='#e63946', s=80, alpha=0.9, label=f'FN ({fn.sum()})', marker='X')

    ax.set_title(nombre, fontsize=11, fontweight='bold')
    ax.set_xlabel('Longitud', fontsize=9)
    ax.set_ylabel('Latitud',  fontsize=9)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/pairplot_espacial.png', dpi=150, bbox_inches='tight')
plt.show()
print("Guardado: outputs/pairplot_espacial.png")



# ══════════════════════════════════════════════════════════════════════════
# ÁRBOL DE DECISIÓN — Visualización
# ══════════════════════════════════════════════════════════════════════════
from sklearn.tree import plot_tree

fig, ax = plt.subplots(figsize=(20, 10))

plot_tree(
    modelos_entrenados['Árbol de Decisión'],
    feature_names=FEATURES,
    class_names=['Ausencia', 'Presencia'],
    filled=True,
    rounded=True,
    fontsize=10,
    ax=ax,
    impurity=True,
    proportion=False,
    precision=3
)

ax.set_title('Árbol de Decisión — Pumas y Jaguares (GBIF 2011)',
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('outputs/arbol_decision.png', dpi=150, bbox_inches='tight')
plt.show()
print("Guardado: arbol_decision.png")


# ══════════════════════════════════════════════════════════════════════════
# PODA ÁRBOL DE DECISIÓN 
# ══════════════════════════════════════════════════════════════════════════


arbol_podado = DecisionTreeClassifier(
    random_state=42,
    max_depth=6,          # máximo 7 niveles de profundidad
    min_samples_leaf=15,  # un nodo hoja necesita mínimo 15 muestras
    min_samples_split=20  # un nodo se divide solo si tiene más de 20 muestras
)

arbol_podado.fit(X_train, y_train)

# ── Comparar rendimiento antes vs después ──────────────────────────────────
from sklearn.metrics import accuracy_score, roc_auc_score

y_pred_original = modelos_entrenados['Árbol de Decisión'].predict(X_test)
y_pred_podado   = arbol_podado.predict(X_test)

print("               Accuracy    AUC")
print(f"Sin podar:     {accuracy_score(y_test, y_pred_original):.3f}      "
      f"{roc_auc_score(y_test, modelos_entrenados['Árbol de Decisión'].predict_proba(X_test)[:,1]):.3f}")
print(f"Podado:        {accuracy_score(y_test, y_pred_podado):.3f}      "
      f"{roc_auc_score(y_test, arbol_podado.predict_proba(X_test)[:,1]):.3f}")

# ── Graficar árbol podado ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 8))

plot_tree(
    arbol_podado,
    feature_names=FEATURES,
    class_names=['Ausencia', 'Presencia'],
    filled=True,
    rounded=True,
    fontsize=11,
    ax=ax,
    precision=3
)

ax.set_title('Árbol de Decisión podado (max_depth=6, min_samples_leaf=15)\nPumas y Jaguares (GBIF 2011)',
             fontsize=13, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('outputs/arbol_podado.png', dpi=150, bbox_inches='tight')
plt.show()
print("Guardado: arbol_podado.png")

df_resultados.to_csv('data/processed/comparacion_modelos.csv')
print("Tabla guardada en data/processed/comparacion_modelos.csv")