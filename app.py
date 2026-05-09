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
    
    # ID de Fleetwood Mac en el dataset de Spotify (comprobado en Colab)
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
    
    # Ordenar por año, quitar duplicados de álbum/canción y resetear el índice
    artist_df = (
        artist_df
        .sort_values('year')
        .drop_duplicates(subset=['short_album_name', 'name'], keep='first')
        .sort_values('year')
        .reset_index(drop=True)
    )

    # Limpieza de títulos (quitar remasterizaciones y caracteres especiales para la API)
    artist_df['clean_name'] = artist_df['name'].str.split(' - ').str[0].str.split(' \(').str[0].str.strip()
    # Filtro de seguridad: solo nombres alfanuméricos
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
        st.plotly_chart(fig, width='stretch')

with tab2:
    st.subheader("Explorador Interactivo de Letras (NLP)")
    st.markdown("""
        1. Obtenemos una muestra representativa de 15 canciones (Seed 14).
        2. Puedes combinar y filtrar en tiempo real las letras extraídas para ver cómo cambia el vocabulario.
    """)
    
    # Tokenización y limpieza con NLTK y regex
    def clean_lyrics(text):
        # Cogemos solo palabras de al menos 3 letras y convertimos a minúsculas
        tokens = re.findall(r'\b[a-z]{3,}\b', text.lower())
        # Usamos las stopwords de NLTK en inglés
        stop_words = set(stopwords.words('english'))
        # Añadimos algunas palabras extra que suelen salir en letras de canciones y no aportan mucho
        custom_stops = {
            'chorus', 'verse', 'like', 'know', 'get', 'got', 'way', 'see', 'well', 
            'back', 'take', 'make', 'could', 'would', 'time', 'come', 'tell', 'one'
        }
        # Filtramos uniendo stop words
        # y aplicando Regex para palabras que se pueden alargar hasta el infinito (ooh, yeah, aah...)
        # (esas no se pueden definir en la lista de antes porque el número de vocales es indeterminado)
        filtered = [
            t for t in tokens 
            if t not in stop_words.union(custom_stops)
            and not re.match(r'^o+h+$', t) # ooh, oooh, oooooh...
            and not re.match(r'^a+h+$', t) # aah, aaah, aaaaah...
            and not re.match(r'^y+e+a+h+$', t) # yeah, yeaaaah...
        ]
        return filtered

    # Inicializamos la "memoria" de Streamlit si no existe (session storage del navegador)
    # NOTA: Nos estamos guardando las letras que encontramos en el navegador
    # para que cuando haces filtrado, como streamlit volvería a llamar a este código,
    # no queremos que cada vez que filtres tenga que llamar a la API otra vez (mala UX y no queremos spamear la API),
    # así que al guardarla en session_state, se queda en el navegador del usuario durante esa sesión
    # y reutiliza la lista en llamadas posteriores para que el filtrado sea rápido y directo.
    if 'lyrics_dict' not in st.session_state:
        st.session_state.lyrics_dict = {}

    # Botón de extracción (solo hace falta pulsarlo una vez)
    if st.button("📥 Descargar y procesar letras") or st.session_state.lyrics_dict:
        
        # Si el diccionario está vacío, llamamos a la API
        if not st.session_state.lyrics_dict:
            sample = df_mac.sample(n=15, random_state=14)['clean_name'].tolist()
            progress = st.progress(0)
            status = st.empty()
            
            for i, song in enumerate(sample):
                status.text(f"Extrayendo ({i+1}/15): {song}...")
                try:
                    r = requests.get(f"https://api.lyrics.ovh/v1/Fleetwood Mac/{song}", timeout=5)
                    if r.status_code == 200:
                        letra = r.json().get('lyrics', '')
                        if letra:
                            # Limpiamos la letra al momento y la guardamos en el diccionario
                            st.session_state.lyrics_dict[song] = clean_lyrics(letra)
                except:
                    pass
                progress.progress((i + 1) / 15)
                time.sleep(0.5) 
                
            status.empty()
            progress.empty()
            
            if st.session_state.lyrics_dict:
                st.success(f"¡{len(st.session_state.lyrics_dict)} letras descargadas y cacheadas con éxito!")
            else:
                st.error("Falló la conexión con la API.")

        # Si el diccionario se descargó bien (o ya lo teníamos), mostramos la gráfica con opciones de filtrado
        if st.session_state.lyrics_dict:
            canciones_disponibles = list(st.session_state.lyrics_dict.keys())
            
            seleccionadas = st.multiselect(
                "Selecciona qué canciones combinar en el análisis:",
                options=canciones_disponibles,
                default=canciones_disponibles # Por defecto mostramos todas las encontradas
            )
            
            if seleccionadas:
                # Concatenamos las listas de palabras de las canciones seleccionadas
                palabras_combinadas = []
                for c in seleccionadas:
                    palabras_combinadas.extend(st.session_state.lyrics_dict[c])
                
                counts = Counter(palabras_combinadas).most_common(15)
                
                if counts:
                    words_df = pd.DataFrame(counts, columns=['Palabra', 'Frecuencia'])
                    
                    fig_bar = px.bar(
                        words_df, 
                        x='Frecuencia', 
                        y='Palabra', 
                        orientation='h',
                        color='Frecuencia',
                        color_continuous_scale='viridis',
                        title="Frecuencia de palabras en la selección actual"
                    )
                    
                    fig_bar.update_layout(
                        yaxis={'categoryorder':'total ascending'},
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    
                    st.plotly_chart(fig_bar, width='stretch')
                else:
                    st.warning("No hay suficientes palabras significativas en esta selección.")
            else:
                st.info("Selecciona al menos una canción arriba para ver el gráfico.")