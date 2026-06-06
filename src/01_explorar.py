import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter #pip uninstall RateLimiter -y pip install geopy

import time
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('C:\\Users\\diego\\Desktop\\Maestria\\Aprendizaje automatico\\Proyecto Puma-Jaguar\\data\\raw\\occurrence.txt', sep='\t')
print("Dimensiones del Dataframe: ", df.shape)
print("Columnas del Dataframe: ", df.columns.tolist())
print("Tipos de datos de cada columna:")
print(df.select_dtypes(include='object').columns)
print("Primeras filas del Dataframe:")
print(df.head())

print("Información del Dataframe:")
print(df.info())

print("Estadísticas descriptivas del Dataframe:")
print(df.describe())    

COLUMNAS = [
    'species',
    'decimalLatitude',
    'decimalLongitude',
    'year',
    'basisOfRecord',
    'habitat',
    'samplingProtocol',
    'locality',
    'occurrenceStatus',
    'iucnRedListCategory'
]



df = df[COLUMNAS]
print("las dimensiones del nuevo data saet seleccionanod columnas es : ",df.shape)
print(df.isnull().sum()) # Verificar valores nulos en cada columna
# despues de verificar los valores nulos, se eliminan las columnas con muchos valores nulos, y se repite el proceso hasta tener un buen df para el analisis.

# Opcion 1, Como decimal latitude y decimal longitude tienen solo 9 valores nulos, los eliminamos con un dropna
# df = df.dropna(subset=['decimalLatitude', 'decimalLongitude'])
# print("Dimensiones del Dataframe después de eliminar filas con valores nulos en decimalLatitude y decimalLongitude: ", df.shape)


# Opcion 2. evaluar si en los datos nulos de latitud y longitud hay información de especies
nulos = df[df['decimalLatitude'].isna()]
print(nulos[['species', 'locality', 'habitat']].to_string())

#Como si hay información, vamos a asigna georreferenciación según la localidad

# Extraer coordenadas desde registros válidos del mismo dataset
coords_ref = (
    df.dropna(subset=['decimalLatitude', 'decimalLongitude'])
    .groupby('locality')[['decimalLatitude', 'decimalLongitude']]
    .first()
)

for idx in nulos.index:
    locality = df.loc[idx, 'locality']
    if locality in coords_ref.index:
        lat = coords_ref.loc[locality, 'decimalLatitude'].item()
        lon = coords_ref.loc[locality, 'decimalLongitude'].item()
        df.loc[idx, 'decimalLatitude']  = lat
        df.loc[idx, 'decimalLongitude'] = lon
        print(f"✓ [{idx}] {locality} → {lat:.4f}, {lon:.4f}")
    else:
        print(f"✗ [{idx}] {locality} → localidad no encontrada en otros registros")


#Como solo puso completar datos de una localidad, se va a tomar el dato del municipio
# Registros que aún tienen nulos
aun_nulos = df[df['decimalLatitude'].isna()].index.tolist()

for idx in aun_nulos:
    locality = df.loc[idx, 'locality']
    municipio = str(locality).split(',')[0].strip()  # "El Tropezon", "Melúa", "El Tigre"
    
    # Buscar registros con coordenadas que contengan ese municipio
    mask = (
        df['locality'].str.contains(municipio, case=False, na=False) &
        df['decimalLatitude'].notna()
    )
    candidatos = df[mask]
    
    if not candidatos.empty:
        lat = candidatos['decimalLatitude'].mean()
        lon = candidatos['decimalLongitude'].mean()
        df.loc[idx, 'decimalLatitude']  = round(lat, 6)
        df.loc[idx, 'decimalLongitude'] = round(lon, 6)
        print(f"✓ [{idx}] {municipio} → {lat:.4f}, {lon:.4f}  ({len(candidatos)} registros promediados)")
    else:
        print(f"✗ [{idx}] {municipio} → no encontrado")

print(f"\nNulos restantes: {df['decimalLatitude'].isna().sum()}")


#Como se asignaron coordenadas a 7 datos, poero quedaron 2 sin diligenciar, vamos a asignar georreferencia a los dos datos restantes

coordenadas_manuales = {
    1:  (4.0290, -71.9460),   # Gabanes — zona río Meta, Puerto Gaitán
    18: (4.1500, -73.6000),   # Vereda Santa María Alta — zona Villavicencio
}

for idx, (lat, lon) in coordenadas_manuales.items():
    df.loc[idx, 'decimalLatitude']  = lat
    df.loc[idx, 'decimalLongitude'] = lon
    print(f"✓ [{idx}] asignado manualmente → {lat}, {lon}")

print(f"\nNulos restantes: {df['decimalLatitude'].isna().sum()}")


#Verificar las especies
print("Especies únicas en el Dataframe: ", df['species'].nunique())
print( "La riqueza de especies que tenemos es: ", df['species'].value_counts())

#Verificar los hábitats
print("Hábitats únicos en el Dataframe: ", df['habitat'].nunique())
print("Los hábitats que tenemos son: ", df['habitat'].value_counts())
print("Los lugares que tenemos son: ", df['locality'].value_counts().head(10))

# Modalidad de avistamiento
print("La modalidad de avistamiento que tenemos es: ", df['basisOfRecord'].value_counts())

# 1. Separar por especie
puma = df[df['species'] == 'Puma concolor']
jaguar = df[df['species'] == 'Panthera onca']

# 2. Imprimir shapes
print("las dimensiones de la matriz de pumas es : ", puma.shape)
print("las dimensiones de la matriz de jaguar es : ", jaguar.shape)

# 3. Guardar en processed
df.to_csv('data/processed/felinos_limpio.csv', index=False)