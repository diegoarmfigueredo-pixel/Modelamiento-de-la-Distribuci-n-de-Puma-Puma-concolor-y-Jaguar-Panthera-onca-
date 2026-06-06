#Intentaremos poner estos datos en un mapa real con el paquete folium, que se instala en la terminal
import folium
import pandas as pd

# Cargar datos
felinos = pd.read_csv('data/processed/felinos_limpio.csv')
puma = felinos[felinos['species'] == 'Puma concolor']
jaguar = felinos[felinos['species'] == 'Panthera onca']

# Crear mapa centrado en los Llanos
mapa = folium.Map(location=[4.0, -72.5], zoom_start=8)

# Agregar puntos de puma
for _, row in puma.iterrows():
    folium.CircleMarker(
        location=[row['decimalLatitude'], row['decimalLongitude']],
        radius=6,
        color='blue',
        fill=True,
        fill_opacity=0.7,
        tooltip=f"Puma concolor - {row['locality']}"
    ).add_to(mapa)

# Agregar puntos de jaguar
for _, row in jaguar.iterrows():
    folium.Marker(
        location=[row['decimalLatitude'], row['decimalLongitude']],
        icon=folium.Icon(color='red', icon='star'),
        tooltip=f"Panthera onca - {row['locality']}"
    ).add_to(mapa)

# Guardar
mapa.save('outputs/mapa_interactivo.html')
print("Mapa guardado en outputs/mapa_interactivo.html")


