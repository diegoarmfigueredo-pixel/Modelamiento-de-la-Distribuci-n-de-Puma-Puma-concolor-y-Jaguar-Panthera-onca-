#Intentar mapa con geopandas
#instalar geopandas en terminal

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

# Cargar shapefile de Colombia
colombia = gpd.read_file('C:\\Users\\diego\\Desktop\\Maestria\\Aprendizaje automatico\\Proyecto Puma-Jaguar\\data\\raw\\SHP_MGN_ANM_MPIOCL')

# Cargar datos de felinos
felinos = pd.read_csv('data/processed/felinos_limpio.csv')
puma = felinos[felinos['species'] == 'Puma concolor']
jaguar = felinos[felinos['species'] == 'Panthera onca']

# Crear mapa
fig, ax = plt.subplots(figsize=(12, 10))

# Dibujar Colombia
colombia.plot(ax=ax, color='lightblue', edgecolor='gray', linewidth=0.5)

# Puntos de cada especie
ax.scatter(puma['decimalLongitude'], puma['decimalLatitude'],
           c='blue', marker='o', s=50, label=r'Puma($Puma\ concolor$) (150)', 
           alpha=0.7, zorder=5)
ax.scatter(jaguar['decimalLongitude'], jaguar['decimalLatitude'],
           c='red', marker='*', s=150, label=r'Jaguar ($Panthera \ onca$) (62)', 
           alpha=0.7, zorder=5)

ax.set_title(r'Distribución de Puma ($Puma\ concolor$)  y Jaguar ($Panthera\ onca$)' +  
             '\nen los Llanos Orientales de Colombia (2011)')
ax.set_xlabel('Longitud')
ax.set_ylabel('Latitud')
ax.legend()
plt.figtext(0.5, 0.01, 
            'Fuente: GBIF (Global Biodiversity Information Facility), 2011. Elaboración propia.', 
            ha='center', fontsize=8, style='italic', color='gray')
plt.savefig('outputs/mapa_geopandas.png', dpi=150)
plt.show()