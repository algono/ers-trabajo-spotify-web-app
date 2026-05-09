import streamlit as st
import pandas as pd
import requests
import re
import os
import gdown
import time
from collections import Counter
import nltk
from nltk.corpus import stopwords
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURACIÓN DE RECURSOS ---

@st.cache_resource
def setup_environment():
    """Descarga de recursos NLTK y el dataset pesado de Google Drive."""
    # Descarga de stopwords para NLP
    nltk.download('stopwords', quiet=True)
    
    # Configuración de descarga del dataset (329MB)
    DATA_FILE = 'tracks_features.csv'
    if not os.path.exists(DATA_FILE):
        file_id = '1jsXTNtGhOrsCApQctYx-hRxAQASAcPlI'
        gdown.download(id=file_id, output=DATA_FILE, quiet=False)
    return DATA_FILE

# --- LÓGICA DE PROCESAMIENTO DE DATOS ---

@st.cache_data
def load_and_preprocess_data(data_path):
    """Carga, filtrado de artista y limpieza de títulos para la API."""
    df = pd.read_csv(data_path)
    
    # ID de Fleetwood Mac en este dataset específico
    TARGET_ARTIST_ID = '08GQAI4eElDnROBrJRGE0X'

    # Filtrado por artista y álbumes de estudio oficiales
    
    # NOTA: Por algún motivo que desconozco, el ejemplo usaba una forma para filtrar MUY ineficiente,
    # ya que convertía cada uno de los artist_id de TODAS las filas (que son muchísimas, el CSV son 300 MB)
    # a un string de python para poder compararlo con nuestro TARGET_ARTIST_ID, lo cual era horrible.
    # Solo en Colab tardaba 13 segundos en hacer esto,
    # y al llevarlo a la web app en Streamlit directamente ni cargaba, se quedaba dando vueltas eternamente.
    # Esta versión usa directamente la función contains() de pandas que está en C por debajo,
    # con lo cual lo hace rapidísimo (en menos de 1 segundo en Colab)
    mask = df['artist_ids'].str.contains(TARGET_ARTIST_ID, na=False)

    artist_df = df[mask].copy()
    
    STUDIO_ALBUMS = [
        'Fleetwood Mac', 'Then Play On', 'Kiln House', 'Future Games',
        'Bare Trees', 'Penguin', 'Mystery to Me', 'Heroes Are Hard to Find',
        'Rumours', 'Tusk', 'Mirage', 'Tango in the Night',
        'Behind the Mask', 'Time', 'Say You Will'
    ]
    
    artist_df['short_album_name'] = artist_df['album'].str.split('(').str[0].str.strip()
    artist_df = artist_df[artist_df['short_album_name'].isin(STUDIO_ALBUMS)]
    
    # Limpieza de títulos (quitar remasterizaciones y caracteres especiales para la API)
    artist_df['clean_name'] = artist_df['name'].str.split(' - ').str[0].str.split(' \(').str[0].str.strip()
    # Filtro de seguridad: solo nombres alfanuméricos para maximizar el hit rate de la API
    safe_songs_df = artist_df[artist_df['clean_name'].str.match(r'^[a-zA-Z0-9\s]+$', na=False)]
    
    return safe_songs_df.drop_duplicates(subset=['clean_name'])

# --- INICIALIZACIÓN ---

st.set_page_config(page_title="Fleetwood Mac Analytics", layout="wide")
st.title("🎸 Dashboard de Análisis: Fleetwood Mac")
st.markdown("Estudio de evolución sonora y análisis léxico mediante el dataset Spotify 1.2M+ y la API lyrics.ovh.")

data_path = setup_environment()
df_mac = load_and_preprocess_data(data_path)

# --- INTERFAZ ---

tab1, tab2 = st.tabs(["📊 Evolución Acústica", "🎤 Análisis NLP de Letras"])

with tab1:
    st.subheader("Características de Audio por Álbum")
    
    features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']
    album_means = df_mac.groupby('short_album_name')[features].mean()
    
    # Normalización para el gráfico de radar
    album_norm = (album_means - album_means.min()) / (album_means.max() - album_means.min() + 1e-9)
    
    album_options = list(album_norm.index)
    selected = st.multiselect("Comparar álbumes:", album_options, default=album_options)
    
    if selected:
        fig = go.Figure()
        
        # Plotly necesita que el último valor sea igual al primero para "cerrar" el polígono
        categories_closed = features + [features[0]]
        
        for album in selected:
            # Extraemos los valores y repetimos el primero al final
            values = album_norm.loc[album].tolist()
            values_closed = values + [values[0]]
            
            fig.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill='toself',
                name=album,
                hoverinfo='text',
                text=[f"{val:.2f}" for val in values_closed]
            ))

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            margin=dict(l=40, r=40, t=40, b=40)
        )
        # Renderizamos el gráfico interactivo en Streamlit
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Procesamiento de Lenguaje Natural")
    st.info("Se utiliza la Seed 14 para extraer una muestra de 15 canciones, garantizando la consistencia con el análisis del Notebook.")
    
    if st.button("Ejecutar análisis léxico"):
        # Usamos la semilla validada en el Colab
        sample = df_mac.sample(n=15, random_state=14)['clean_name'].tolist()
        all_text = ""
        
        progress = st.progress(0)
        status = st.empty()
        
        for i, song in enumerate(sample):
            status.text(f"Solicitando API: {song}...")
            try:
                # Consultamos la API lyrics.ovh
                r = requests.get(f"https://api.lyrics.ovh/v1/Fleetwood Mac/{song}", timeout=5)
                if r.status_code == 200:
                    all_text += " " + r.json().get('lyrics', '')
            except:
                pass
            progress.progress((i + 1) / 15)
            time.sleep(0.5) # Delay preventivo para la API pública
            
        status.success("Análisis completado.")
        
        # Tokenización y limpieza con NLTK
        words = re.findall(r'\b[a-z]{3,}\b', all_text.lower())
        stop_words = set(stopwords.words('english'))
        
        # Dejamos en custom_stops solo palabras reales
        custom_stops = {'chorus', 'verse', 'like', 'know', 'get', 'got', 'way', 'see', 'well'}
        
        # Filtramos uniendo stop words y aplicando Regex para palabras que se pueden alargar hasta el infinito (ooh, yeah, aah...)
        filtered = [
            w for w in words
            if w not in stop_words.union(custom_stops)
            and not re.match(r'^o+h+$', w)      # ooh, oooh, oooooh...
            and not re.match(r'^a+h+$', w)      # aah, aaah, aaaaah...
            and not re.match(r'^y+e+a+h+$', w)  # yeah, yeaaaah...
        ]
        
        counts = Counter(filtered).most_common(15)
        
        if counts:
            words_df = pd.DataFrame(counts, columns=['Palabra', 'Frecuencia'])
            
            # Gráfico de barras interactivo
            fig_bar = px.bar(
                words_df, 
                x='Frecuencia', 
                y='Palabra', 
                orientation='h',
                color='Frecuencia',
                color_continuous_scale='viridis',
                title="Palabras más frecuentes en la muestra (Stop-words filtradas)"
            )
            
            # Ordenamos para que la más frecuente quede arriba del todo
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'})
            
            st.plotly_chart(fig_bar, use_container_width=True)