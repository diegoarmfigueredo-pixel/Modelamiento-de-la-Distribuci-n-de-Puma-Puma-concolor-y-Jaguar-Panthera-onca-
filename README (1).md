# SDM Felinos Colombia — *Puma concolor* y *Panthera onca*

## Descripción

Este proyecto aplica técnicas de Machine Learning para modelar la distribución
espacial de dos grandes felinos colombianos: el puma (*Puma concolor*) y el
jaguar (*Panthera onca*) en los Llanos Orientales de Colombia.

Se trabajan dos objetivos concretos:

1. **Modelo de Distribución de Especies (SDM):** predecir en qué zonas del
   territorio es probable encontrar cada especie, a partir de registros de
   avistamiento georeferenciados.
2. **Análisis de coexistencia y conflicto humano-fauna:** identificar zonas
   donde ambas especies cohabitan y donde sus territorios se superponen con
   actividad humana, con implicaciones para la conservación.

---

## Especies estudiadas

| Especie | Nombre común | Estado IUCN | Rol ecológico |
|---|---|---|---|
| *Puma concolor* | Puma / León de montaña | LC (Preocupación menor) | Depredador tope, especie sombrilla |
| *Panthera onca* | Jaguar | NT (Casi amenazado) | Depredador tope, especie bandera y paraguas |

> **¿Qué es una especie sombrilla?** Es una especie cuya protección implica
> automáticamente la conservación de todo su ecosistema y de las demás especies
> que lo habitan. Proteger al jaguar equivale a proteger cientos de otras especies.

---

## Dataset

- **Fuente:Descarga directa: https://www.gbif.org
Búsqueda: species = "Puma concolor" OR "Panthera onca", country = Colombia, year = 2011.
- **Fuente:** GBIF (Global Biodiversity Information Facility) —
  plataforma internacional de datos de biodiversidad de acceso libre.
- **Registros:** 212 observaciones válidas (212 originales, 9 completadas
  por ausencia de coordenadas).
- **Cobertura temporal:** año 2011.
- **Cobertura espacial:** Llanos Orientales, Colombia
  (departamentos de Meta y Casanare).
- **Método de detección:** 100% observación humana directa — dato relevante
  para el análisis de conflicto humano-fauna.
- **Distribución por especie:** 152 registros de puma / 60 registros de jaguar.

**Columnas utilizadas:**

| Columna | Descripción |
|---|---|
| `species` | Nombre científico de la especie |
| `decimalLatitude` | Coordenada geográfica Y |
| `decimalLongitude` | Coordenada geográfica X |
| `year` | Año del registro |
| `basisOfRecord` | Método de detección |
| `habitat` | Tipo de ecosistema |
| `samplingProtocol` | Protocolo de muestreo |
| `locality` | Nombre del sitio exacto |
| `occurrenceStatus` | Confirmación de presencia |
| `iucnRedListCategory` | Categoría de amenaza IUCN |

---

## Pipeline

El proyecto sigue una secuencia de 6 scripts que se ejecutan en orden:

```
data/raw/occurrence.txt
        ↓
01_explorar.py       → limpieza y exploración
        ↓
02_pseudoausencias.py → construcción del dataset de entrenamiento
        ↓
03_modelo.py         → entrenamiento y comparación de 4 modelos
        ↓
04_mapa.py           → mapa de distribución con localidades
05_mapa_interactivo.py → mapa interactivo en navegador
06_mapa_geopandas.py  → mapa cartográfico oficial de Colombia
```

| Script | Descripción |
|---|---|
| `01_explorar.py` | Carga el dataset TSV de GBIF, selecciona las 10 columnas relevantes, elimina registros sin coordenadas y exporta el dataset limpio |
| `02_pseudoausencias.py` | Genera 424 pseudoausencias (puntos aleatorios dentro del área de estudio), aplica Label Encoding a variables categóricas y construye el dataset de entrenamiento balanceado (ratio 1:2) |
| `03_modelo.py` | Entrena y compara cuatro algoritmos de ML, evalúa con métricas AUC-ROC, Accuracy, F1, Precision, Recall, MCC, TSS, y guarda el mejor modelo |
| `04_mapa.py` | Visualiza los registros reales de cada especie con sus localidades y calcula el porcentaje de solapamiento territorial |
| `05_mapa_interactivo.py` | Genera un mapa HTML interactivo con folium sobre cartografía real |
| `06_mapa_geopandas.py` | Genera un mapa estático con el shapefile oficial de Colombia usando geopandas |

---

## Instalación

```bash
# 1. Crear entorno virtual
python -m venv .venv

# 2. Activar (Windows)
.venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt
```

**Dependencias principales:**

```
pandas
numpy
matplotlib
seaborn
scikit-learn
folium
geopandas
```

---

## Preprocesamiento aplicado

Se aplicaron tres metodologías de preprocesamiento:

1. **Limpieza de datos:** Completaron 9 registros sin coordenadas geográficas.
2. **Generación de pseudoausencias:** técnica estándar en SDMs para suplir la
   ausencia de registros negativos confirmados. Se generaron 424 puntos
   aleatorios dentro del bounding box del área de estudio (ratio 1:2).
3. **Normalización (StandardScaler):** estandarización de variables numéricas
   para que ninguna domine sobre otra por escala, especialmente necesario para
   la Regresión Logística.

---

## Resultados

### Comparación de modelos

| Modelo	         | Accuracy | AUC	   | F1	   | Precision	| Recall	| MCC	   | TSS   |
| Reg. Logística	| 0.771	  | 0.601	| 0.000	| 0.000	   | 0.000	| 0.000	| 0.000 |
| Random Forest	| 0.906	  | 0.919	| 0.795	| 0.795	   | 0.795	| 0.734	| 0.734 |
| Grad. Boosting	| 0.871	  | 0.921	| 0.656	| 0.840	   | 0.538	| 0.603	| 0.508 |
| Árbol Decisión	| 0.906	  | 0.885	| 0.805	| 0.767	   | 0.846	| 0.745	| 0.770 |




> **¿Qué es el AUC-ROC?** Es la métrica principal en SDMs. Mide qué tan bien
> el modelo distingue entre presencias reales y pseudoausencias. Un valor de
> 1.0 es perfecto y 0.5 equivale a una predicción aleatoria. Random Forest
> con 0.919 indica una capacidad discriminativa excelente.

**Random Forest** fue seleccionado como modelo final — consistente con la
literatura científica que lo establece como estándar en SDMs modernos por su
robustez con pocos datos de presencia y su capacidad para capturar relaciones
no lineales.

### Solapamiento territorial

El **93.3% de los registros de jaguar** tienen al menos un registro de puma
en un radio de 5 km. Esto indica que ambas especies son **simpátricas** —
coexisten en el mismo territorio — y sugiere un mecanismo de
**partición de nicho**: el jaguar caza presas más grandes (tapir, caimán) y
el puma presas medianas (venado, chigüiro), evitando así la competencia directa.

### Implicaciones para la conservación

- **Efecto paraguas:** proteger el territorio del jaguar protege
  automáticamente al puma y a todas las especies del ecosistema.
- **Asimetría de vulnerabilidad:** ante degradación del hábitat, el jaguar
  (NT) desaparece primero; el puma (LC) es más tolerante a la perturbación.
- **Todos los avistamientos son observaciones humanas directas**, lo que
  sugiere zonas de contacto frecuente entre personas y grandes felinos —
  potenciales puntos de conflicto humano-fauna.

### Implicaciones económicas

- **Ecoturismo:** la presencia de jaguares y pumas convierte estas zonas en
  destinos de alto valor turístico, alternativa económica a la ganadería extensiva.
- **Servicios ecosistémicos:** los felinos regulan poblaciones de herbívoros,
  protegiendo pastizales — beneficio directo para los ganaderos locales.
- **Pagos por Servicios Ambientales (PSA - Decreto-Ley 870 de 2017) y Guía de Valoración economica ambiental:** este tipo de mapas sustenta
  solicitudes de compensación económica a propietarios de fincas por conservar
  hábitat crítico para especies amenazadas. "Vale mas el ejemplar Vivo que muerto"

---

## Limitaciones

El modelo predijo una probabilidad constante de 0.54 para todos los puntos
de la grilla de predicción.

La causa principal es la homogeneidad del dataset: todos los registros
provienen de una región geográfica pequeña y de un único año, lo que no
genera suficiente variación espacial para que el modelo aprenda patrones
diferenciables.

Para mejorar el modelo en el futuro se requiere:
- Variables ambientales reales (temperatura, precipitación, NDVI,
  cobertura vegetal) desde IDEAM o Google Earth Engine
- Registros de una región más amplia (todo Colombia o Suramérica)
- Datos de múltiples años para capturar variación temporal

---

## Autores

Natali Charrupi Riascos · Diego Figueredo Gomez · Luis Alberto Torres Cardona  
Universidad Autónoma de Occidente — Maestría en IA y Ciencia de datos   
Curso: Aprendizaje Automático · 2026
