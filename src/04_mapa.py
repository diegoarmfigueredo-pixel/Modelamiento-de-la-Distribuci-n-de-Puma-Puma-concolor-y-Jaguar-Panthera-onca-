import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
with open('data/processed/modelo_rf.pkl', 'rb') as f:
    modelo = pickle.load(f)
felinos = pd.read_csv('C:\\Users\\diego\\Desktop\\Maestria\\Aprendizaje automatico\Proyecto Puma-Jaguar\\data\\processed\\felinos_limpio.csv')

df = pd.read_csv(r"C:\\Users\\diego\\Desktop\\Maestria\\Aprendizaje automatico\Proyecto Puma-Jaguar\\data\\processed\\felinos_limpio.csv")
print("Dimensiones del Dataframe para el mapa: ", df.shape)
print("Columnas del Dataframe: ", df.columns.tolist())

#Crear una grilla de latitud y longitud, para visualizar la distribución espacial de los felinos, esto es útil para el analisis de SDM posteriormente.
#Coordenadas máximas y mínimas de latitud y longitud, para establecer los límites del área de estudio, y para realizar análisis espaciales posteriormente.
lat_min = df['decimalLatitude'].min()
lat_max = df['decimalLatitude'].max()
lon_min = df['decimalLongitude'].min()
lon_max = df['decimalLongitude'].max()
print("Latitud mínima: ", lat_min)
print("Latitud máxima: ", lat_max)  
print("Longitud mínima: ", lon_min)
print("Longitud máxima: ", lon_max)

#Crear el arreglo de la grilla con np.arange
lats = np.arange(lat_min, lat_max, 0.02) # 0.02 es el paso de la grilla, se puede ajustar según la resolución deseada
lons = np.arange(lon_min, lon_max, 0.02)
print("Número de latitudes en la grilla: ", len(lats))
print("Número de longitudes en la grilla: ", len(lons))

#Aplanar grilla con .ravel() para crear un arreglo de coordenadas, esto es útil para el analisis de SDM posteriormente.pues elimina redundancia y evita sobreentrenamiento de modelo
lon_grid, lat_grid = np.meshgrid(lons, lats)
lon_grid_flat = lon_grid.ravel()    
lat_grid_flat = lat_grid.ravel()
print("Número de puntos en la grilla: ", len(lon_grid_flat))

#Crear df_grilla con las dos columnas seleccionadas ( decimalLatitude, decimalLongitud), para tener un dataframe limpio para el analisis de SDM posteriormente.
df_grilla = pd.DataFrame({
    'decimalLatitude': lat_grid_flat,   
    'decimalLongitude': lon_grid_flat
})
print("Dimensiones del Dataframe de la grilla: ", df_grilla.shape)
print("Columnas del Dataframe de la grilla: ", df_grilla.columns.tolist())  


#Agrergar nombre de columas
df_grilla_valores = df_grilla.values  # convierte a numpy sin nombres


#Predecir modelo
# predict_proba toma las características de la grilla y devuelve las probabilidades 
# de predicción para cada clase (presencia y ausencia). Se selecciona la columna [:, 1]
# que corresponde a la clase positiva (presencia) — todas las filas (:) y segunda columna (1).
probabilidades = modelo.predict_proba(df_grilla_valores)[:, 1]

# Visualizar
# Para visualizar, se debe convertir el vector de probabilidades a grilla de nuevo.
# reshape hace lo contrario de ravel — toma un vector unidimensional y lo convierte 
# en una matriz con la misma forma que lat_grid y lon_grid.
prob_grid = probabilidades.reshape(lat_grid.shape)
print("Dimensiones de prob_grid: ", prob_grid.shape)

print("Min probabilidad:", probabilidades.min())
print("Max probabilidad:", probabilidades.max())
print("Primeros 10 valores:", probabilidades[:10])

#Rangos de grilla
print("Grilla lat min/max:", df_grilla['decimalLatitude'].min(), df_grilla['decimalLatitude'].max())
print("Train lat min/max:", felinos['decimalLatitude'].min(), felinos['decimalLatitude'].max())



#Visualizar el mapa de probabilidades con plt.imshow, que muestra la grilla de probabilidades como una imagen, donde cada píxel representa la probabilidad de presencia de los felinos en esa ubicación geográfica. Se utiliza el cmap 'viridis' para asignar colores a las diferentes probabilidades, y se agrega una barra de color para interpretar los valores de probabilidad.

# Visualizar el mapa de distribución
#Cargar de nuevo jaguar y puma
puma = felinos[felinos['species'] == 'Puma concolor']
jaguar = felinos[felinos['species'] == 'Panthera onca']



plt.figure(figsize=(12, 8))
plt.scatter(puma['decimalLongitude'], puma['decimalLatitude'],
            c='blue', marker='o', s=80, label='Puma concolor (152)', 
            alpha=0.7, edgecolors='darkblue')
plt.scatter(jaguar['decimalLongitude'], jaguar['decimalLatitude'],
            c='red', marker='*', s=150, label='Panthera onca (60)', 
            alpha=0.7, edgecolors='darkred')
plt.title('Distribución de felinos - Llanos Orientales, Colombia 2011')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.legend()
# Mostrar solo localidades únicas sin repetir
localidades_vistas = set()
for _, row in felinos.iterrows():
    loc = row['locality'].split(',')[0].strip()  # toma solo la primera parte
    if loc not in localidades_vistas:
        plt.annotate(loc,
                    xy=(row['decimalLongitude'], row['decimalLatitude']),
                    fontsize=6, alpha=0.7,
                    xytext=(3, 3), textcoords='offset points')
        localidades_vistas.add(loc)
plt.grid(True, alpha=0.3)
plt.savefig('outputs/mapa_distribucion.png', dpi=150)
plt.show()



# Calcular el solapamiento de las especies ( tener importado numpy as np)

# Calcular solapamiento

# Distancia mínima para considerar solapamiento (0.05 grados ~ 5km)
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



