import streamlit as st
import os
import asyncio
import edge_tts
from langdetect import detect, DetectorFactory
import base64
import json
import tempfile
from pathlib import Path
import re
import numpy as np
from scipy.io import wavfile

# Optional imports for Coqui TTS
try:
    from TTS.api import TTS
    COQUI_AVAILABLE = True
except ImportError:
    COQUI_AVAILABLE = False

# Optional import for gTTS
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False

# Optional import for Silero TTS
try:
    SILERO_AVAILABLE = True
except ImportError:
    SILERO_AVAILABLE = False

# Optional import for pyttsx3
try:
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

# Ensure consistent language detection
DetectorFactory.seed = 0

# ============================================
# FREE TTS ENGINE CONFIGURATIONS
# ============================================

# Coqui TTS Models - All FREE and LOCAL
COQUI_MODELS = {
    "Portuguese": "pt",
    "Polish": "pl",
    "Turkish": "tr",
    "Russian": "ru",
    "Dutch": "nl",
    "Czech": "cs",
    "Chinese": "zh-cn",
    "Japanese": "ja",
    "Korean": "ko",
    "Hindi": "hi"
}

# Page Configuration
st.set_page_config(
    page_title="Text2Audio Ultra PRO",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Display Artiforia-Hub logo at the top
st.markdown("<div style='text-align:center;margin-bottom:1.5rem;'>", unsafe_allow_html=True)
st.image("artiphoria - logo.png", width=180)
st.markdown("<span style='display:block;font-size:1.5rem;font-weight:800;color:#00d2ff;text-shadow:0 2px 8px #3a7bd5, 0 0 2px #00ff88;margin-top:0.5rem;'>Artiforia-Hub</span>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Premium CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    * { font-family: 'Outfit', sans-serif; }
    
    .stApp {
        background: linear-gradient(135deg, #050505 0%, #1a1a2e 100%);
        color: #e0e0e0;
    }
    
    .sidebar .sidebar-content {
        background: rgba(26, 26, 46, 0.95) !important;
        border-right: 1px solid rgba(255,255,255,0.1);
    }
    
    h1 {
        background: linear-gradient(90deg, #00d2ff, #3a7bd5, #00ff88);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 3rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    .stTextArea textarea, textarea[data-testid="stTextArea"] {
        background: #fff !important;
        color: #111 !important;
        font-size: 1.15rem !important;
        line-height: 1.7 !important;
        border-radius: 10px !important;
        padding: 18px !important;
        border: 1.5px solid #d0d0d0 !important;
        font-family: 'Segoe UI', 'Arial', sans-serif !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        text-transform: uppercase;
        letter-spacing: 2px;
        box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(0, 210, 255, 0.5);
    }
    
    .quote-box {
        background: rgba(255, 255, 255, 0.03);
        border-left: 4px solid #00ff88;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0;
        font-style: italic;
        color: #a0a0a0;
        backdrop-filter: blur(5px);
    }
    
    .free-badge {
        background: linear-gradient(90deg, #00d2ff, #3a7bd5);
        color: white;
        padding: 3px 10px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: bold;
        margin-left: 5px;
    }
    
    .engine-card {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid rgba(0,255,136,0.3);
    }
    
    .feature-box {
        background: linear-gradient(135deg, rgba(0,255,136,0.1), rgba(0,200,100,0.05));
        border: 1px solid rgba(0,255,136,0.3);
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Custom CSS for Text Area
st.markdown('''
    <style>
    textarea[data-testid="stTextArea"] {
        font-size: 1.15rem !important;
        line-height: 1.7 !important;
        background: #fff !important;
        color: #111 !important;
        border-radius: 10px !important;
        padding: 18px !important;
        border: 1.5px solid #d0d0d0 !important;
        font-family: 'Segoe UI', 'Arial', sans-serif !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.03);
    }
    </style>
''', unsafe_allow_html=True)

# ============================================
# TTS ENGINE IMPLEMENTATIONS (ALL FREE)
# ============================================

class CoquiTTSEngine:
    """Coqui TTS - Free, open-source neural TTS with voice cloning"""
    
    _tts_instances = {}
    
    @staticmethod
    def is_available():
        return COQUI_AVAILABLE
    
    @staticmethod
    def get_device():
        if COQUI_AVAILABLE:
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cpu"
    
    @classmethod
    def get_tts(cls, model_key):
        if model_key not in cls._tts_instances:
            model_name = COQUI_MODELS[model_key]["name"]
            device = cls.get_device()
            with st.spinner(f"Loading {COQUI_MODELS[model_key]['display']}... (first time may take a few minutes)"):
                cls._tts_instances[model_key] = TTS(model_name).to(device)
        return cls._tts_instances[model_key]
    
    @classmethod
    def get_speakers(cls, model_key):
        """Get available speakers for multi-speaker models"""
        try:
            tts = cls.get_tts(model_key)
            if hasattr(tts, 'speakers') and tts.speakers:
                return tts.speakers
        except:
            pass
        return None
    
    @classmethod
    def synthesize(cls, text, model_key, language="en", speaker=None, speaker_wav=None):
        tts = cls.get_tts(model_key)
        model_info = COQUI_MODELS.get(model_key, COQUI_MODELS["xtts_v2"])
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            if model_info.get("voice_clone") and "xtts" in model_key:
                # XTTS v2 - supports voice cloning and multilingual
                if speaker_wav:
                    tts.tts_to_file(
                        text=text,
                        file_path=tmp_path,
                        speaker_wav=speaker_wav,
                        language=language
                    )
                else:
                    # Use default speaker
                    speakers = tts.speakers if hasattr(tts, 'speakers') else None
                    if speakers:
                        tts.tts_to_file(
                            text=text,
                            file_path=tmp_path,
                            speaker=speakers[0],
                            language=language
                        )
                    else:
                        tts.tts_to_file(
                            text=text,
                            file_path=tmp_path,
                            language=language
                        )
            elif hasattr(tts, 'speakers') and tts.speakers and speaker:
                # Multi-speaker model
                tts.tts_to_file(text=text, file_path=tmp_path, speaker=speaker)
            else:
                # Single speaker model
                tts.tts_to_file(text=text, file_path=tmp_path)
            
            with open(tmp_path, "rb") as f:
                audio_data = f.read()
            
            return audio_data
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

class EdgeTTSEngine:
    """Edge TTS - Free Microsoft voices"""
    
    @staticmethod
    def is_available():
        return True
    
    @staticmethod
    async def _synthesize_async(text, voice_id, rate, pitch, volume):
        rate_str = f"{rate:+d}%"
        pitch_str = f"{pitch:+d}Hz"
        vol_str = f"{volume-100:+d}%"
        
        communicate = edge_tts.Communicate(text, voice_id, rate=rate_str, pitch=pitch_str, volume=vol_str)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        return audio_data
    
    @staticmethod
    def synthesize(text, voice_id, rate=0, pitch=0, volume=100):
        return asyncio.run(EdgeTTSEngine._synthesize_async(text, voice_id, rate, pitch, volume))


class GTTSEngine:
    """gTTS - Free Google Text-to-Speech"""
    
    LANGUAGES = {
        "English": "en", "Arabic": "ar", "French": "fr", "Spanish": "es",
        "German": "de", "Italian": "it", "Portuguese": "pt", "Russian": "ru",
        "Chinese": "zh-CN", "Japanese": "ja", "Korean": "ko", "Hindi": "hi",
        "Dutch": "nl", "Polish": "pl", "Turkish": "tr"
    }
    
    ACCENTS = {
        "English": {
            "American": "com", "British": "co.uk", "Australian": "com.au",
            "Indian": "co.in", "Canadian": "ca", "Irish": "ie"
        },
        "French": {"France": "fr", "Canadian": "ca"},
        "Spanish": {"Spain": "es", "Mexico": "com.mx", "Argentina": "com.ar"},
        "Portuguese": {"Brazil": "com.br", "Portugal": "pt"}
    }
    
    @staticmethod
    def is_available():
        return GTTS_AVAILABLE
    
    @staticmethod
    def synthesize(text, lang="en", tld="com", slow=False):
        """Generate speech using gTTS"""
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            tts = gTTS(text=text, lang=lang, tld=tld, slow=slow)
            tts.save(tmp_path)
            
            with open(tmp_path, "rb") as f:
                audio_data = f.read()
            return audio_data
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class SileroTTSEngine:
    """Silero TTS - Free, lightweight neural TTS with natural voices"""
    
    _model_cache = {}
    
    LANGUAGES = {
        "English": ("v3_en", ["en_0", "en_1", "en_2", "en_3", "en_4", "en_5", "en_6", "en_7", "en_8", "en_9",
                              "en_10", "en_11", "en_12", "en_13", "en_14", "en_15", "en_16", "en_17", "en_18", "en_19",
                              "en_20", "en_21", "en_22", "en_23", "en_24", "en_25", "en_26", "en_27", "en_28", "en_29",
                              "en_30", "en_31", "en_32", "en_33", "en_34", "en_35", "en_36", "en_37", "en_38", "en_39",
                              "en_40", "en_41", "en_42", "en_43", "en_44", "en_45", "en_46", "en_47", "en_48", "en_49",
                              "en_50", "en_51", "en_52", "en_53", "en_54", "en_55", "en_56", "en_57", "en_58", "en_59",
                              "en_60", "en_61", "en_62", "en_63", "en_64", "en_65", "en_66", "en_67", "en_68", "en_69",
                              "en_70", "en_71", "en_72", "en_73", "en_74", "en_75", "en_76", "en_77", "en_78", "en_79",
                              "en_80", "en_81", "en_82", "en_83", "en_84", "en_85", "en_86", "en_87", "en_88", "en_89",
                              "en_90", "en_91", "en_92", "en_93", "en_94", "en_95", "en_96", "en_97", "en_98", "en_99",
                              "en_100", "en_101", "en_102", "en_103", "en_104", "en_105", "en_106", "en_107", "en_108", "en_109",
                              "en_110", "en_111", "en_112", "en_113", "en_114", "en_115", "en_116", "en_117"]),
        "Russian": ("v4_ru", ["aidar", "baya", "kseniya", "xenia", "eugene", "random"]),
        "German": ("v3_de", ["bernd_ungerer", "eva_k", "friedrich", "hokuspokus", "karlsson", "random"]),
        "Spanish": ("v3_es", ["es_0", "es_1", "es_2", "random"]),
        "French": ("v3_fr", ["fr_0", "fr_1", "fr_2", "fr_3", "fr_4", "fr_5", "random"]),
        "Ukrainian": ("v4_ua", ["mykyta", "random"]),
        "Uzbek": ("v4_uz", ["dilnavoz", "random"]),
        "Indic": ("v4_indic", ["hindi_female", "hindi_male", "random"])
    }
    
    # Curated voice descriptions based on community testing
    SPEAKER_NAMES = {
        # English - Popular/Recommended voices with tested characteristics
        "en_0": "Voice 0 - Female, Clear", "en_1": "Voice 1 - Female, Warm", 
        "en_2": "Voice 2 - Female, Young", "en_3": "Voice 3 - Male, Deep", 
        "en_4": "Voice 4 - Male, Neutral", "en_5": "Voice 5 - Male, Young",
        "en_6": "Voice 6 - Female, Professional", "en_7": "Voice 7 - Female, Soft", 
        "en_8": "Voice 8 - Male, Strong", "en_9": "Voice 9 - Male, Casual",
        "en_10": "Voice 10 - Female, Energetic", "en_11": "Voice 11 - Male, Calm",
        "en_12": "Voice 12 - Female", "en_13": "Voice 13 - Male", "en_14": "Voice 14 - Female",
        "en_15": "Voice 15 - Male", "en_16": "Voice 16 - Female", "en_17": "Voice 17 - Male",
        "en_18": "Voice 18 - Female", "en_19": "Voice 19 - Male", "en_20": "Voice 20 - Female",
        "en_21": "Voice 21 - Male", "en_22": "Voice 22 - Female", "en_23": "Voice 23 - Male",
        "en_24": "Voice 24 - Female", "en_25": "Voice 25 - Male", "en_26": "Voice 26 - Female",
        "en_27": "Voice 27 - Male", "en_28": "Voice 28 - Female", "en_29": "Voice 29 - Male",
        "en_30": "Voice 30 - Female", "en_31": "Voice 31 - Male", "en_32": "Voice 32 - Female",
        # Russian named voices
        "aidar": "Aidar (Male)", "baya": "Baya (Female)", "kseniya": "Kseniya (Female)",
        "xenia": "Xenia (Female)", "eugene": "Eugene (Male)", "random": "Random Voice"
    }
    
    @staticmethod
    def is_available():
        return SILERO_AVAILABLE
    
    @classmethod
    def get_model(cls, language="English"):
        model_id, _ = cls.LANGUAGES.get(language, cls.LANGUAGES["English"])
        
        if model_id not in cls._model_cache:
            device = torch.device('cpu')
            model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language=model_id.split('_')[1],
                speaker=model_id
            )
            model.to(device)
            cls._model_cache[model_id] = model
        
        return cls._model_cache[model_id]
    
    @classmethod
    def synthesize(cls, text, language="English", speaker="en_0", sample_rate=48000):
        """Generate speech using Silero TTS"""
        model_id, speakers = cls.LANGUAGES.get(language, cls.LANGUAGES["English"])
        
        if speaker not in speakers:
            speaker = speakers[0]
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            device = torch.device('cpu')
            model, _ = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language=model_id.split('_')[1] if '_' in model_id else 'en',
                speaker=model_id
            )
            model.to(device)
            
            # Generate audio
            audio = model.apply_tts(
                text=text,
                speaker=speaker,
                sample_rate=sample_rate
            )
            
            # Save to file using scipy (no torchcodec needed)
            import scipy.io.wavfile as wavfile
            audio_np = audio.cpu().numpy()
            # Normalize to int16 range
            audio_int16 = (audio_np * 32767).astype('int16')
            wavfile.write(tmp_path, sample_rate, audio_int16)
            
            with open(tmp_path, "rb") as f:
                audio_data = f.read()
            return audio_data
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)




# ============================================
# VOICE LOADING
# ============================================

@st.cache_data
def load_all_voices():
    voices_file = "voices_full.json"
    if os.path.exists(voices_file):
        with open(voices_file, "r", encoding="utf-8") as f:
            return json.load(f)
    
    try:
        async def fetch():
            v_mgr = await edge_tts.VoicesManager.create()
            return v_mgr.voices
        voices = asyncio.run(fetch())
        with open(voices_file, "w", encoding="utf-8") as f:
            json.dump(voices, f, ensure_ascii=False, indent=2)
        return voices
    except:
        return []

all_voices = load_all_voices()

# ============================================
# SIDEBAR - ENGINE SELECTION
# ============================================

with st.sidebar:
    st.markdown("## 🎙️ TTS ENGINE")
    st.markdown("---")
    
    # Engine selection - show available engines first
    engine_options = []
    if GTTS_AVAILABLE:
        engine_options.append("🌐 Google TTS (Online)")
    engine_options.append("💬 Edge TTS (Microsoft)")
    if COQUI_AVAILABLE:
        engine_options.extend([
            "🌟 XTTS v2 (Best Quality)",
            "🎭 VITS Multi-Speaker",
            "📚 Tacotron2 (Audiobook)",
            "⚡ Speedy Speech (Fast)"
        ])
    selected_engine = st.selectbox("Select Engine", engine_options)
    
    # Engine-specific settings
    st.markdown("---")
    
    # Map selection to model key
    engine_model_map = {
        "🌟 XTTS v2 (Best Quality)": "xtts_v2",
        "🎭 VITS Multi-Speaker": "vits_english",
        "📚 Tacotron2 (Audiobook)": "tacotron2",
        "⚡ Speedy Speech (Fast)": "speedy_speech"
    }
    
    if selected_engine in engine_model_map and COQUI_AVAILABLE:
        model_key = engine_model_map.get(selected_engine, "xtts_v2")
        model_info = COQUI_MODELS.get(model_key, COQUI_MODELS["xtts_v2"])
        
        st.markdown("### ⚙️ Voice Settings")
        st.caption(model_info['description'])
        
        # Language selection for XTTS
        if model_info.get("multilingual"):
            coqui_language = st.selectbox("🌍 Language", list(XTTS_LANGUAGES.keys()))
        else:
            coqui_language = "English"
        
        # Voice cloning for XTTS
        clone_audio = None
        if model_info.get("voice_clone"):
            st.markdown("#### 🎤 Voice Cloning")
            st.caption("Upload a clear voice sample (5-15 seconds)")
            clone_audio = st.file_uploader("Reference Audio", type=["wav", "mp3", "ogg", "flac"])
            if clone_audio:
                st.success("✅ Voice sample loaded!")
        
        # Speaker selection for multi-speaker models
        selected_speaker = None
        if model_key == "vits_english":
            try:
                speakers = CoquiTTSEngine.get_speakers(model_key)
                if speakers:
                    selected_speaker = st.selectbox("🎭 Speaker Style", speakers[:50])
            except:
                pass
        
        # ElevenLabs-style Voice Controls
        st.markdown("---")
        st.markdown("#### 🎛️ Voice Controls")
        
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            stability = st.slider("⚖️ Stability", 0, 100, 50, 
                help="Higher = more consistent, Lower = more expressive variation")
            speed_factor = st.slider("⏩ Speed", 0.5, 2.0, 1.0, 0.1,
                help="Adjust speaking rate")
        with col_ctrl2:
            similarity_boost = st.slider("🧬 Clarity", 0, 100, 75,
                help="Higher = clearer pronunciation")
            style_exag = st.slider("✨ Style", 0, 100, 0,
                help="Amplify speaker's emotional style")
    
    
    
    elif "Google" in selected_engine:
        # gTTS Settings
        st.markdown("### ⚙️ Google TTS Settings")
        st.caption("✅ Natural-sounding voices powered by Google")
        
        gtts_language = st.selectbox("🌍 Language", list(GTTSEngine.LANGUAGES.keys()))
        
        # Show accents for supported languages
        gtts_accent = None
        if gtts_language in GTTSEngine.ACCENTS:
            accent_options = list(GTTSEngine.ACCENTS[gtts_language].keys())
            gtts_accent = st.selectbox("🎯 Accent", accent_options)
        
        gtts_slow = st.checkbox("🐢 Slow Mode", value=False, help="Speak slower for learning")
        
        st.markdown("---")
        st.info("💡 Google TTS provides natural-sounding speech with multiple accents")
        st.caption("⚠️ Gender selection not available - Google TTS uses a single voice per language/accent")
    
    elif "Edge" in selected_engine:
        # Edge TTS Settings (no sidebar filters)
        st.markdown("### ⚙️ Edge TTS Settings")
        st.caption("💬 Microsoft voices with full control")
        st.markdown("---")
        st.markdown("#### 🎛️ Voice Controls")
        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            speed_pro = st.slider("⏩ Speed", -50, 50, 0, key="edge_speed",
                help="Adjust speaking rate")
            pitch_pro = st.slider("🎵 Pitch", -50, 50, 0, key="edge_pitch",
                help="Adjust voice pitch")
        with col_ctrl2:
            volume_pro = st.slider("🔊 Volume", 0, 150, 100, key="edge_vol",
                help="Adjust loudness")
        # Move voice and style selection here
        language_options = [
            ("English", "en"), ("Arabic", "ar"), ("French", "fr"), ("Spanish", "es"), ("German", "de"), ("Chinese", "zh"), ("All", None)
        ]
        lang_display, lang_codes = zip(*language_options)
        selected_lang = st.selectbox("🌍 Language", lang_display, key="edge_lang_main")
        lang_code = dict(language_options)[selected_lang]
        if lang_code:
            lang_filtered = [v for v in all_voices if v['Locale'].startswith(lang_code)]
        else:
            lang_filtered = all_voices
        personalities = set()
        for v in lang_filtered:
            tags = v.get("VoiceTag", {}).get("VoicePersonalities", [])
            for t in tags: personalities.add(t)
        style_options = ["All"] + sorted(list(personalities))
        selected_style = st.selectbox("🎭 Style (Filter by personality)", style_options, key="edge_style_main")
        if selected_style == "All":
            filtered = lang_filtered
        else:
            filtered = [v for v in lang_filtered if selected_style in v.get("VoiceTag", {}).get("VoicePersonalities", [])]
        if not filtered:
            st.info(f"No voices found for {selected_lang} + style '{selected_style}'. Showing all {selected_lang} voices.")
            filtered = lang_filtered
        filter_key = f"voice_edge_{selected_lang}_{selected_style}"
        v_selection = st.selectbox("Select Voice", options=[v['FriendlyName'] for v in filtered], key=filter_key)
        selected_voice = next((v for v in filtered if v['FriendlyName'] == v_selection), filtered[0] if filtered else None)
        if selected_voice:
            st.session_state['edge_selected_voice'] = selected_voice
        st.info("Select your preferred voice and style below, then click Generate Audio.")

# ============================================
# MAIN APP
# ============================================

st.markdown("<h1>TEXT2AUDIO ULTRA <span class='free-badge'>PRO</span></h1>", unsafe_allow_html=True)
st.markdown("### Professional Neural Text-to-Speech Studio")

# Main Layout
col_left, col_right = st.columns([2, 1])

with col_left:
    clone_audio = None
    text_input = st.text_area(
        "✍️ ENTER YOUR TEXT", 
        height=300, 
        placeholder="Type or paste your text here...\n\nFor best results with XTTS v2:\n• Use clear, well-punctuated sentences\n• Avoid very long paragraphs (split them)\n• Include commas and periods for natural pauses"
    )
    # Character count
    if text_input:
        char_count = len(text_input)
        word_count = len(text_input.split())
        st.caption(f"📊 {char_count} characters • {word_count} words")
        # Language detection for Edge TTS
    generate_btn = st.button("🚀 GENERATE AUDIO")
    audio_data = None
    audio_format = None
    file_ext = None
    if generate_btn:
        if not text_input:
            st.error("Please enter some text to synthesize.")
        else:
            with st.spinner("🔊 Generating audio, please wait..."):
                try:
                    audio_data = None
                    audio_format = None
                    file_ext = None
                    if selected_engine in engine_model_map and COQUI_AVAILABLE:
                        model_key = engine_model_map.get(selected_engine, "xtts_v2")
                        model_info = COQUI_MODELS.get(model_key, COQUI_MODELS["xtts_v2"])
                        # Handle voice clone file
                        speaker_wav_path = None
                        if clone_audio and model_info.get("voice_clone"):
                            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                                tmp.write(clone_audio.read())
                                speaker_wav_path = tmp.name
                        try:
                            lang_code = XTTS_LANGUAGES.get(coqui_language, "en") if model_info.get("multilingual") else "en"
                            audio_data = CoquiTTSEngine.synthesize(
                                text=text_input,
                                model_key=model_key,
                                language=lang_code,
                                speaker=selected_speaker if 'selected_speaker' in dir() else None,
                                speaker_wav=speaker_wav_path
                            )
                        finally:
                            if speaker_wav_path and os.path.exists(speaker_wav_path):
                                os.unlink(speaker_wav_path)
                        audio_format = "audio/wav"
                        file_ext = "wav"
                    elif "Google" in selected_engine and GTTS_AVAILABLE:
                        lang_code = GTTSEngine.LANGUAGES.get(gtts_language, "en")
                        tld = None
                        if gtts_accent:
                            tld = GTTSEngine.ACCENTS[gtts_language][gtts_accent]
                        audio_data = GTTSEngine.synthesize(
                            text=text_input,
                            lang=lang_code,
                            tld=tld if tld else "com",
                            slow=gtts_slow
                        )
                        audio_format = "audio/mp3"
                        file_ext = "mp3"
                    elif "Edge" in selected_engine:
                        # Use selected_voice from session state
                        edge_voice = st.session_state.get('edge_selected_voice')
                        if not edge_voice:
                            st.error("Please select a voice for Edge TTS.")
                        else:
                            audio_data = EdgeTTSEngine.synthesize(
                                text=text_input,
                                voice_id=edge_voice['ShortName'],
                                rate=speed_pro,
                                pitch=pitch_pro,
                                volume=volume_pro
                            )
                            audio_format = "audio/mp3"
                            file_ext = "mp3"
                except Exception as e:
                    st.error(f"⚠️ Synthesis failed: {e}")
    if audio_data:
        st.success("✨ Audio generated successfully!")
        st.audio(audio_data, format=audio_format)
        b64 = base64.b64encode(audio_data).decode()
        engine_name = "neural" if selected_engine in engine_model_map else "edge"
        btn_css = "background: linear-gradient(90deg, #00d2ff, #3a7bd5); color: white; border: none; padding: 12px 24px; border-radius: 8px; cursor: pointer; font-weight: 600;"
        st.markdown(
            f'<a href="data:{audio_format};base64,{b64}" download="audio_{engine_name}.{file_ext}" style="text-decoration:none;">'
            f'<button style="{btn_css}">📥 DOWNLOAD (.{file_ext.upper()})</button></a>',
            unsafe_allow_html=True
        )
                
# Footer
st.markdown("---")
st.markdown("""
<p style='text-align: center; color: #00d2ff; font-size: 1rem; font-weight:700; letter-spacing:1px;'>
Powered by Artiphoria-Hub Technology
</p>
""", unsafe_allow_html=True)
