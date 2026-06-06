import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder #pasa de texto a numeros, para que el modelo de SDM pueda trabajar con ellas, pero antes de hacer el encoding, es importante eliminar las filas con valores nulos en esas columnas, para evitar errores en el proceso de encoding.

felinos = pd.read_csv("C:\\Users\\diego\\Desktop\\Maestria\\Aprendizaje automatico\\Proyecto Puma-Jaguar\\data\\processed\\felinos_limpio.csv")
puma = felinos[felinos['species'] == 'Puma concolor']
jaguar = felinos[felinos['species'] == 'Panthera onca']

print("Dimensiones del Dataframe de Puma: ", puma.shape)
print("Dimensiones del Dataframe de Jaguar: ", jaguar.shape)   

#definir matriz de latitud y longitud (maximos y minimos) para el data set de felinos
#definir esta matriz de coordenadas, es util para establecer los limites del area de estudio, y para realizar analisis espaciales posteriormente.
lat_max = felinos['decimalLatitude'].max()
lat_min = felinos['decimalLatitude'].min()
lon_max = felinos['decimalLongitude'].max()
lon_min = felinos['decimalLongitude'].min()

print("Latitud máxima: ", lat_max)
print("Latitud mínima: ", lat_min)      
print("Longitud máxima: ", lon_max)
print("Longitud mínima: ", lon_min)

#Como hay muchos mas datos de puma que de jaguar (razon de 1:2.4, que aproximado a 2), para evitar sesgos y que el SDm funcione, haremos un analisis de pseudoausencias con 3 veces el total de registros ( en total tenemos 212 registros, entonces, se analilzaran 636 psudoausencias)
N_PSEUDO = 424

pseudo_lat = np.random.uniform(lat_min, lat_max, N_PSEUDO)
pseudo_lon = np.random.uniform(lon_min, lon_max, N_PSEUDO)

#Crear un DataFrame para las pseudoausencias, quitando los valores nulos de las columnas que no se van a usar para el analisis de SDM, y agregando una columna de presencia/ausencia, donde 1 es presencia y 0 es ausencia, para luego unir este DataFrame con el DataFrame de felinos, y tener un solo DataFrame con las presencias y las pseudoausencias, para el analisis de SDM posteriormente.
pseudoausencia = pd.DataFrame({
    'decimalLatitude': pseudo_lat,
    'decimalLongitude': pseudo_lon,
    'year': np.nan,
    'basisOfRecord': np.nan,
    'habitat': np.nan,
    'samplingProtocol': np.nan,
    'locality': np.nan,
    'occurrenceStatus': np.nan,
    'iucnRedListCategory': np.nan,
    'presencia': 0
})

print(pseudoausencia.shape)
felinos['presencia'] = 1 #Agrega una nueva columna 'presencia' al DataFrame puma y le asigna el valor 1 a todas las filas, indicando que estas filas representan presencias de la especie.
pseudoausencia['presencia'] = 0

print("las dimensiones del data frame de pseudoausencias es : ", pseudoausencia.shape)

#Unir las el dataframe de pseudoausencias con el dataframe de felinos, para tener un solo dataframe con las presencias y las pseudoausencias, esto es util para el analisis de SDM posteriormente.
df_modelo = pd.concat([felinos, pseudoausencia], ignore_index=True)
print("Dimensiones del Dataframe combinado de felinos y pseudoausencias: ", df_modelo.shape)

#En las dimesiones del dataframe combinado ( df_,modelo), 848 filas = 212 registros reales + 636 pseudoausencias. 11 columnas.

#Seleccionar columnas que requiere el modelo de SDM, y eliminar las filas con valores nulos en esas columnas, para tener un dataframe limpio para el analisis de SDM posteriormente.
#Como hay columnas que son tipo texto (str), debemos hacerun encoding, que convierte textos en numeros para que el modelo de SDM pueda trabajar con ellas, pero antes de hacer el encoding, es importante eliminar las filas con valores nulos en esas columnas, para evitar errores en el proceso de encoding.
# Los nan se qjuitaron arriba al hacer las columna de pseudoausencias


le = LabelEncoder() #creando un objeto listo para trabajar.

columnas_texto = ['basisOfRecord', 'habitat', 'samplingProtocol', 
                  'locality', 'occurrenceStatus', 'iucnRedListCategory']

for columna in columnas_texto:
    df_modelo[columna] = df_modelo[columna].fillna('desconocido')
    df_modelo[columna] = le.fit_transform(df_modelo[columna].astype(str)) # type: ignore #fit aprende todas las categorías únicas de la columna, mientras transform aplica esa transformación y devuelve la columna convertida a números.

print ("las column as del dataframe son: ", df_modelo.head())
print ("Dimensiones del Dataframe final para el analisis de SDM: ", df_modelo.shape)

#Guardar el DataFrame final para el análisis de SDM en un archivo CSV, para luego usarlo en el análisis de SDM posteriormente.
df_modelo.to_csv('data/processed/dataset_modelo.csv', index=False)