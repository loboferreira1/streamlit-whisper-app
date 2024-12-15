import streamlit as st
from io import BytesIO
from faster_whisper import WhisperModel

@st.cache_resource
def load_data():
    return WhisperModel("small", device="cpu", compute_type="int8")

st.set_page_config(page_title="Transcrição de Áudios", page_icon= '🔊', layout="wide")

st.markdown('# Áudio 🔊 para Texto 📝')
st.markdown("---")

buffer = st.file_uploader("Envie um arquivo de áudio 🔊 para ser transcrito 📝", accept_multiple_files=False)

model = load_data()

if buffer is not None:
    st.spinner("Aguarde um instante...")
        
    audio_contents = buffer.getvalue()
    segments, info = model.transcribe(BytesIO(audio_contents), beam_size=5)
    response = {'language': info.language, 'probability': info.language_probability,
                'texts': [{'start': segment.start , 'end': segment.end, 'text': segment.text} for segment in segments]}

    st.markdown(f"A transcrição occoreu usando a língua {response['language']}, com probabilidade de {response['probability']:0.2%}.")
    
    st.markdown("Segue a transcrição.")
    for text in response['texts']:
        st.write(f"{text['start']}s a {text['end']}s: {text['text']}")