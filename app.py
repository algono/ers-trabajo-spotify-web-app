import streamlit as st
import pandas as pd
import ast
import requests
import re
import os
import gdown
import time
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import nltk
from nltk.corpus import stopwords

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
    
    selected = st.multiselect("Comparar álbumes:", list(album_norm.index), default=['Rumours', 'Tusk'])
    
    if selected:
        fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist() + [0]
        
        for album in selected:
            values = album_norm.loc[album].tolist() + [album_norm.loc[album].iloc[0]]
            ax.plot(angles, values, linewidth=2, label=album)
            ax.fill(angles, values, alpha=0.1)
            
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        st.pyplot(fig)

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
        tokens = re.findall(r'\b[a-z]{3,}\b', all_text.lower())
        stop_words = set(stopwords.words('english'))
        # Añadimos muletillas comunes en letras de canciones
        custom_stops = {'yeah', 'ooh', 'chorus', 'verse', 'like', 'know', 'get', 'got', 'way', 'oh', 'ah', 'hey', 'see', 'well'}
        filtered = [t for t in tokens if t not in stop_words.union(custom_stops)]
        
        counts = Counter(filtered).most_common(15)
        
        if counts:
            words_df = pd.DataFrame(counts, columns=['Palabra', 'Frecuencia'])
            fig_bar, ax_bar = plt.subplots(figsize=(10, 5))
            sns.barplot(data=words_df, x='Frecuencia', y='Palabra', hue='Palabra', palette='viridis', legend=False, ax=ax_bar)
            st.pyplot(fig_bar)