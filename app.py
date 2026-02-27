import streamlit as st
import asyncio
import edge_tts
from gtts import gTTS
import os
import time
import json
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
    except:
        return []

all_edge_voices = load_edge_voices()

# --- CSS Styling (Premium UI) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    * { font-family: 'Outfit', sans-serif; }
    .stApp { background: linear-gradient(135deg, #050505 0%, #1a1a2e 100%); color: #e0e0e0; }
    h1 { background: linear-gradient(90deg, #00d2ff, #3a7bd5, #00ff88); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800 !important; }
    .stButton>button { background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%); color: white; border-radius: 12px; height: 3em; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar: Service & Voice Settings ---
with st.sidebar:
    st.image("https://raw.githubusercontent.com/mohamed-salah-1/text2audio/main/artiphoria%20-%20logo.png", width=150)
    st.markdown("## ⚙️ Engine Settings")
    
    # SERVICE SELECTION (Edge vs Google)
    service = st.selectbox("Select TTS Service", ["Edge TTS (High Quality)", "Google TTS (Standard)"])
    
    st.markdown("---")
    
    if service == "Edge TTS (High Quality)":
        st.markdown("### 🎙️ Edge Voice Settings")
        
        # Language Filter
        lang_map = {"English": "en", "Arabic": "ar", "French": "fr", "Spanish": "es"}
        sel_lang = st.selectbox("🌍 Language", list(lang_map.keys()))
        lang_code = lang_map[sel_lang]
        
        lang_filtered = [v for v in all_edge_voices if v['Locale'].startswith(lang_code)]
        
        # Personality/Style Filter
        personalities = set()
        for v in lang_filtered:
            for t in v.get("VoiceTag", {}).get("VoicePersonalities", []): personalities.add(t)
        
        selected_style = st.selectbox("🎭 Voice Personality", ["All Styles"] + sorted(list(personalities)))
        
        # Filter by Style
        if selected_style != "All Styles":
            lang_filtered = [v for v in lang_filtered if selected_style in v.get("VoiceTag", {}).get("VoicePersonalities", [])]
        
        v_selection = st.selectbox("Select Voice", options=[v['FriendlyName'] for v in lang_filtered])
        selected_voice_obj = next((v for v in lang_filtered if v['FriendlyName'] == v_selection), lang_filtered[0])
        
        # Audio Controls
        speed = st.slider("⏩ Speed (%)", -50, 50, 0)
        pitch = st.slider("🎵 Pitch (Hz)", -50, 50, 0)

    else:
        st.markdown("### 🎙️ Google Voice Settings")
        g_lang = st.selectbox("🌍 Language", ["en", "ar", "fr", "es", "de"])
        st.info("Google TTS is stable but has fewer 'personality' options.")

# --- Main App Interface ---
st.markdown("<h1>TEXT2AUDIO ULTRA <span style='font-size:0.5em; vertical-align:middle; background:#3a7bd5; color:white; padding:2px 8px; border-radius:4px;'>PRO</span></h1>", unsafe_allow_html=True)

# Input Tabs (Manual vs File)
tab1, tab2 = st.tabs(["✍️ Manual Text", "📂 Upload .txt File"])
final_text = ""

with tab1:
    manual_text = st.text_area("Enter your script:", height=200, placeholder="Paste your text here...")
    if manual_text: final_text = manual_text

with tab2:
    uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
    if uploaded_file:
        final_text = uploaded_file.read().decode("utf-8")
        st.success("File content loaded!")

# --- Generation Logic ---
if st.button("🚀 GENERATE AUDIO"):
    if not final_text.strip():
        st.error("Please provide text first.")
    else:
        temp_filename = f"audio_{int(time.time())}.mp3"
        
        with st.spinner("🔊 Synthesizing..."):
            try:
                if service == "Edge TTS (High Quality)":
                    rate_str = f"{speed:+d}%"
                    pitch_str = f"{pitch:+d}Hz"
                    async def run_edge():
                        communicate = edge_tts.Communicate(final_text, selected_voice_obj['ShortName'], rate=rate_str, pitch=pitch_str)
                        await communicate.save(temp_filename)
                    asyncio.run(run_edge())
                else:
                    tts = gTTS(text=final_text, lang=g_lang)
                    tts.save(temp_filename)
                
                # Play and Download
                st.audio(temp_filename)
                
                with open(temp_filename, "rb") as f:
                    data = f.read()
                    b64 = base64.b64encode(data).decode()
                    st.markdown(f'<a href="data:audio/mp3;base64,{b64}" download="artiphoria_audio.mp3"><button style="width:100%; padding:10px; background:#00ff88; color:black; border:none; border-radius:8px; cursor:pointer;">📥 DOWNLOAD MP3</button></a>', unsafe_allow_html=True)
                
                # Immediate cleanup for VPS space
                os.remove(temp_filename)
                
            except Exception as e:
                st.error(f"Error: {e}")

st.markdown("---")
st.caption("Admin of PythonCafe? Check out 'Drawing with Code' on YouTube for more!")