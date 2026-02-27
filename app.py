import streamlit as st
import asyncio
import edge_tts
from gtts import gTTS
from deep_translator import GoogleTranslator
import os
import time
import base64

# Page Configuration
st.set_page_config(page_title="Text2Audio Ultra PRO", page_icon="🎤", layout="wide")

# --- Load Voices (Cached) ---
@st.cache_data
def load_edge_voices():
    try:
        async def fetch():
            v_mgr = await edge_tts.VoicesManager.create()
            return v_mgr.voices
        return asyncio.run(fetch())
    except: return []

all_edge_voices = load_edge_voices()

# --- Helper: Text Chunking for Translation ---
def chunk_text(text, batch_size=4500):
    """Breaks text into smaller parts to avoid the 5000-character API limit."""
    return [text[i:i + batch_size] for i in range(0, len(text), batch_size)]

# --- Premium CSS Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    * { font-family: 'Outfit', sans-serif; }
    .stApp { background: linear-gradient(135deg, #050505 0%, #1a1a2e 100%); color: #e0e0e0; }
    .main-title { font-size: 3rem; font-weight: 800; background: linear-gradient(90deg, #00d2ff, #3a7bd5, #00ff88); -webkit-background-clip: text; -webkit-text-fill-color: transparent; display: inline-block; }
    .pro-badge { background: #3a7bd5; color: white; padding: 5px 15px; border-radius: 8px; font-size: 0.5em; vertical-align: middle; margin-left: 10px; font-weight: 800; }
    .stButton>button { background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%); color: white; border-radius: 12px; height: 3.5em; font-weight: bold; border: none; }
    .translation-box { background: rgba(255, 255, 255, 0.05); padding: 15px; border-radius: 10px; border-left: 5px solid #00d2ff; margin: 10px 0; color: #fff; }
    .char-counter { font-size: 0.9em; padding: 5px; border-radius: 5px; text-align: right; margin-top: -10px; }
    .custom-footer { text-align: center; color: #888; padding: 20px; border-top: 1px solid #333; margin-top: 50px; }
    .custom-footer a { color: #00d2ff; text-decoration: none; font-weight: 600; }
    .custom-footer a:hover { color: #00ff88; text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.image("https://raw.githubusercontent.com/mohamed-salah-1/text2audio/main/artiphoria%20-%20logo.png", width=150)
    st.markdown("## ⚙️ Studio Settings")
    
    service = st.radio("TTS Engine", ["Microsoft Edge (Premium)", "Google TTS (Basic)"])
    
    st.markdown("---")
    st.markdown("### 🌐 Translation Settings")
    do_translate = st.checkbox("Enable Translation")
    
    target_lang = "en"
    if do_translate:
        target_lang_name = st.selectbox("Target Language", ["Arabic", "English"])
        target_lang = "ar" if target_lang_name == "Arabic" else "en"

    if service == "Microsoft Edge (Premium)":
        st.markdown("### 🎙️ Audio Filters")
        voice_lang_default = "Arabic" if (do_translate and target_lang == "ar") else "English"
        lang_map = {"English": "en", "Arabic": "ar", "French": "fr", "Spanish": "es"}
        sel_lang = st.selectbox("🌍 Voice Language", list(lang_map.keys()), 
                                index=list(lang_map.keys()).index(voice_lang_default))
        
        lang_filtered = [v for v in all_edge_voices if v['Locale'].startswith(lang_map[sel_lang])]
        v_selection = st.selectbox("Select Voice", options=[v['FriendlyName'] for v in lang_filtered])
        selected_voice_obj = next((v for v in lang_filtered if v['FriendlyName'] == v_selection), lang_filtered[0])
        
        st.markdown("#### 🎛️ Audio Tuning")
        speed = st.slider("⏩ Speed (%)", -50, 50, 0)
        pitch = st.slider("🎵 Pitch (Hz)", -50, 50, 0)
        volume = st.slider("🔊 Volume (%)", -50, 50, 0)
    else:
        g_lang = st.selectbox("🌍 Voice Language", ["en", "ar", "fr", "es"], 
                              index=1 if (do_translate and target_lang == "ar") else 0)

# --- Main App ---
st.markdown('<div style="margin-bottom: 25px;"><span class="main-title">TEXT2AUDIO ULTRA</span><span class="pro-badge">PRO</span></div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["✍️ Manual Text", "📂 Upload .txt File"])
input_text = ""

with tab1:
    manual_text = st.text_area("Enter your script:", height=250, placeholder="Paste your text here...")
    input_text = manual_text

with tab2:
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    if uploaded_file:
        input_text = uploaded_file.read().decode("utf-8")

# --- Character Counter Display ---
char_count = len(input_text)
char_color = "#00ff88" if char_count < 5000 else "#ffcc00" if char_count < 10000 else "#ff4b4b"
st.markdown(f'<div class="char-counter" style="color: {char_color};">Characters: <b>{char_count}</b></div>', unsafe_allow_html=True)

if st.button("🚀 GENERATE AUDIO"):
    if not input_text.strip():
        st.error("Please provide text first.")
    else:
        temp_filename = f"audio_{int(time.time())}.mp3"
        final_text = input_text
        
        with st.spinner("Processing..."):
            try:
                # 1. Translation with Chunking
                if do_translate:
                    translator = GoogleTranslator(source='auto', target=target_lang)
                    if char_count > 4500:
                        chunks = chunk_text(input_text)
                        translated_chunks = [translator.translate(c) for c in chunks]
                        final_text = " ".join(translated_chunks)
                    else:
                        final_text = translator.translate(input_text)
                    
                    st.markdown(f'<div class="translation-box"><b>Translated Result:</b><br>{final_text}</div>', unsafe_allow_html=True)

                # 2. Audio Synthesis
                if service == "Microsoft Edge (Premium)":
                    async def run_edge():
                        communicate = edge_tts.Communicate(final_text, selected_voice_obj['ShortName'], 
                                                           rate=f"{speed:+d}%", pitch=f"{pitch:+d}Hz", volume=f"{volume:+d}%")
                        await communicate.save(temp_filename)
                    asyncio.run(run_edge())
                else:
                    tts = gTTS(text=final_text, lang=target_lang if do_translate else g_lang)
                    tts.save(temp_filename)
                
                st.audio(temp_filename)
                with open(temp_filename, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                    st.markdown(f'<a href="data:audio/mp3;base64,{b64}" download="artiphoria_audio.mp3"><button style="width:100%; padding:12px; background:#00ff88; color:black; border:none; border-radius:10px; cursor:pointer; font-weight:bold;">📥 DOWNLOAD MP3</button></a>', unsafe_allow_html=True)
                
                os.remove(temp_filename)
                
            except Exception as e:
                st.error(f"Error: {e}")

# --- ENHANCED BRANDED FOOTER ---
st.markdown("""
<div class="custom-footer">
    Visit <a href="https://www.youtube.com/@artiphoria" target="_blank">Artiphoria-Hub</a> on YouTube | 
    Part of the <a href="https://www.facebook.com/groups/pythoncafe" target="_blank">PythonCafe Group</a> 
    and <a href="https://www.facebook.com/PythonKaliSecure" target="_blank">Code Secure Community</a>
</div>
""", unsafe_allow_html=True)