import time
import streamlit as st
import os
import sounddevice as sd
import soundfile as sf
import numpy as np
import tempfile
import librosa
from tensorflow.keras.models import load_model
import joblib
import wave
import io
from pydub import AudioSegment
AudioSegment.ffmpeg = "/usr/local/bin/ffmpeg"
from pydub.playback import play
import threading
from sklearn.preprocessing import LabelEncoder

# Set page config with enhanced styling
st.set_page_config(
    page_title="🎙️ Sound Emotion Detector",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS with header-matching tab buttons
st.markdown("""
<style>
    .stApp {
        background: #5a34d9;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .header {
        text-align: center;
        margin-bottom: 2rem;
        padding: 1.5rem;
        background: #610b23;
        border-radius: 15px;
    }
    .header h1 {
        color: #f7f2f4;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    .header p {
        color: #f7f2f4;
        font-size: 1.1rem;
    }
    .upload-box {
        border: 2px dashed #4a4a4a;
        border-radius: 15px;
        padding: 3rem;
        text-align: center;
        background-color: #18166b;
        margin-bottom: 2rem;
        transition: all 0.3s;
    }
    .upload-box:hover {
        background-color: rgba(255, 255, 255, 0.1);
        border-color: #06D6A0;
    }
    .stButton>button {
        background: #750d14;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
    }
    .info-card {
        background-color: #167d45;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    .info-card h3 {
        color: #f7f2f4;
        border-bottom: 2px solid #06D6A0;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    .positive { 
        color: #06D6A0; 
        font-weight: bold; 
    }
    .negative { 
        color: #EF476F; 
        font-weight: bold; 
    }
    .neutral { 
        color: #FFD166; 
        font-weight: bold; 
    }
    .result-container {
        background: rgba(255,255,255,0.9);
        padding: 2rem;
        border-radius: 12px;
        margin-top: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .recording-status {
        background-color: #118AB2;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
    }
    .recording-active {
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    .progress-container {
        margin: 1.5rem 0;
    }
    .progress-label {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
    }
    .progress-bar {
        height: 20px;
        background: #e9ecef;
        border-radius: 10px;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: flex-end;
        padding-right: 10px;
        color: white;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .tab-container {
        background: rgba(255,255,255,0.8);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #610b23;  /* Matches header color */
        color: white;
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 20px !important;
        transition: all 0.3s;
        margin-right: 5px !important;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background: #4a091b !important;  /* Slightly darker than header */
        color: white !important;
        box-shadow: inset 0 -2px 0 #06D6A0;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: #520a1f;
    }
    .audio-visualization {
        margin: 1.5rem 0;
        border-radius: 12px;
        overflow: hidden;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #6c757d;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

class AudioRecorder:
    def __init__(self):
        self.recording = False
        self.audio_data = None
        self.sample_rate = 44100
        self.thread = None

    def start_recording(self, duration):
        self.recording = True
        self.audio_data = None

        def record():
            with sd.InputStream(samplerate=self.sample_rate, channels=1, dtype='float32') as stream:
                frames = []
                for _ in range(int(duration * self.sample_rate / 1024)):
                    if not self.recording:
                        break
                    data, _ = stream.read(1024)
                    frames.append(data)
                self.audio_data = np.concatenate(frames)

        self.thread = threading.Thread(target=record)
        self.thread.start()

    def stop_recording(self):
        self.recording = False
        if self.thread:
            self.thread.join()
        return self.audio_data

def save_audio_file(audio_data, sample_rate, filename):
    """Save audio data to a WAV file"""
    try:
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes((audio_data * 32767).astype(np.int16))
        return True
    except Exception as e:
        st.error(f"Error saving audio file: {str(e)}")
        return False

def play_audio(audio_data, sample_rate):
    """Play audio directly from numpy array"""
    try:
        audio_segment = AudioSegment(
            audio_data.tobytes(),
            frame_rate=sample_rate,
            sample_width=audio_data.dtype.itemsize,
            channels=1
        )
        play(audio_segment)
    except Exception as e:
        st.error(f"Error playing audio: {str(e)}")

def record_audio_interface(duration):
    """UI for recording audio with real-time feedback"""
    recorder = AudioRecorder()

    # Create containers for better layout
    status_container = st.empty()
    progress_container = st.empty()
    time_container = st.empty()

    # Start recording
    recorder.start_recording(duration)

    # Update UI during recording
    start_time = time.time()
    while time.time() - start_time < duration and recorder.recording:
        elapsed = time.time() - start_time
        progress = min(elapsed / duration, 1.0)

        # Update status
        status_container.markdown(
            f'<div class="recording-status recording-active">'
            f'🎤 Recording... ({int(elapsed)}s / {duration}s)'
            f'</div>',
            unsafe_allow_html=True
        )

        # Update progress bar
        progress_container.markdown(
            f'<div class="progress-container">'
            f'<div class="progress-label">'
            f'<span>Recording Progress</span>'
            f'<span>{int(progress*100)}%</span>'
            f'</div>'
            f'<div class="progress-bar">'
            f'<div class="progress-fill" style="width: {progress*100}%; background: #118AB2;">'
            f'{int(progress*100)}%'
            f'</div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        time.sleep(0.1)

    # Stop recording
    audio_data = recorder.stop_recording()
    status_container.empty()
    progress_container.empty()
    time_container.empty()

    if audio_data is not None and len(audio_data) > 0:
        st.success("✅ Recording complete!")
        return audio_data, recorder.sample_rate
    else:
        st.error("No audio data was recorded. Please try again.")
        return None, None

def save_and_display_audio(audio_data, sample_rate):
    """Save audio to temp file and create player"""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
            tmp_filename = tmp_file.name
            save_audio_file(audio_data, sample_rate, tmp_filename)

        # Display audio player with enhanced visualization
        st.markdown("### 🎧 Recorded Audio Preview")
        with open(tmp_filename, "rb") as f:
            audio_bytes = f.read()

        # Create a container for better layout
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.audio(audio_bytes, format='audio/wav')
            with col2:
                if st.button("▶️ Play Audio", key="play_button",
                             help="Play the recorded audio through your speakers"):
                    play_audio(audio_data, sample_rate)

        return tmp_filename
    except Exception as e:
        st.error(f"Error handling audio: {str(e)}")
        return None

def load_emotion_model():
    """Load the emotion detection model and label encoder"""
    try:
        model = load_model('/Users/bhagat09/IdeaProjects/PROJECT/.venv/emotion_sensor.h5')
        le = joblib.load('/Users/bhagat09/IdeaProjects/PROJECT/.venv/label_encoder.joblib')
        return model, le
    except Exception as e:
        st.error(f"Model loading error: {str(e)}")
        return None, None

def extract_mfcc(filename):
    """Extract MFCC features from audio file"""
    try:
        y, sr = librosa.load(filename, duration=3, offset=0.5)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return mfcc
    except Exception as e:
        st.error(f"Feature extraction error: {str(e)}")
        return None

def preprocess_audio(audio_path, model):
    """Preprocess audio to match model input requirements"""
    try:
        mfcc = extract_mfcc(audio_path)
        if mfcc is None:
            return None

        mfcc_processed = np.expand_dims(mfcc, axis=0)
        mfcc_processed = np.expand_dims(mfcc_processed, -1)
        return mfcc_processed
    except Exception as e:
        st.error(f"Preprocessing error: {str(e)}")
        return None

def predict_emotion(audio_path):
    """Predict emotion from audio file"""
    model, le = load_emotion_model()
    if model is None or le is None:
        return None, None

    processed_audio = preprocess_audio(audio_path, model)
    if processed_audio is None:
        return None, None

    try:
        prediction = model.predict(processed_audio)
        predicted_class = np.argmax(prediction, axis=1)
        emotion = le.inverse_transform(predicted_class)[0]
        probability = np.max(prediction)
        return emotion, probability
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None

def display_results(emotion, probability):
    """Display emotion analysis results with enhanced visualization"""
    if emotion is None or probability is None:
        st.error("Could not analyze the audio. Please try again.")
        return

    # Determine emotion category and styling
    emotion = emotion.lower()
    if emotion in ['joyfully', 'euphoric', 'happy']:
        emoji = "😊"
        category = "positive"
        color = "#06D6A0"
        gradient = "linear-gradient(135deg, #06D6A0 0%, #118AB2 100%)"
    elif emotion in ['sad', 'angry', 'fear', 'disgust']:
        emoji = "🤢" if emotion == 'disgust' else "😞"
        category = "negative"
        color = "#EF476F"
        gradient = "linear-gradient(135deg, #EF476F 0%, #FF6B6B 100%)"
    else:
        emoji = "😐"
        category = "neutral"
        color = "#FFD166"
        gradient = "linear-gradient(135deg, #FFD166 0%, #F8961E 100%)"

    confidence_level = int(probability * 100)

    # Create columns for better layout
    col1, col2 = st.columns([1, 1])

    with col1:
        # Emotion result box
        st.markdown(
            f"""
            <div style="background: {gradient}; padding: 1.5rem; border-radius: 10px; color: white; margin-bottom: 1.5rem;">
                <h2 style="margin: 0; color: white;">Detected Emotion: {emotion.capitalize()} {emoji}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Confidence level
        st.markdown("### Confidence Level")
        st.progress(confidence_level/100)
        st.markdown(f"**{confidence_level}% certainty**")

    with col2:
        # Visualization placeholder (you can add a relevant visualization here)
        if category == "positive":
            st.image("https://cdn-icons-png.flaticon.com/512/2583/2583344.png", width=150)
        elif category == "negative":
            st.image("https://cdn-icons-png.flaticon.com/512/2583/2583437.png", width=150)
        else:
            st.image("https://cdn-icons-png.flaticon.com/512/2583/2583277.png", width=150)

    # Add appropriate feedback based on emotion
    if category == "positive":
        st.balloons()
        st.success("This sounds like a positive emotional state! 😊 Keep it up!")
    elif category == "negative":
        st.warning("This sounds like a challenging emotional state. 💙 Remember, it's okay to feel this way.")
    else:
        st.info("This sounds like a neutral emotional state. 😐")

def main():
    """Main application function"""
    # App header with enhanced styling
    st.markdown("""
    <div class='header'>
        <h1>🎙️ Sound Emotion Detector</h1>
        <p>Discover the emotional tone in speech using advanced AI analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'recording_data' not in st.session_state:
        st.session_state.recording_data = None
    if 'sample_rate' not in st.session_state:
        st.session_state.sample_rate = None
    if 'audio_file' not in st.session_state:
        st.session_state.audio_file = None
    if 'show_result' not in st.session_state:
        st.session_state.show_result = False
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None
    if 'show_upload_result' not in st.session_state:
        st.session_state.show_upload_result = False

    # Use tabs to separate functionalities
    tab1, tab2 = st.tabs(["🎤 Live Recording", "📁 Upload Audio"])

    with tab1:
        st.markdown("""
        <div class='upload-box'>
            <h3>Real-time Microphone Input</h3>
            <p>Record directly from your microphone for instant analysis</p>
        </div>
        """, unsafe_allow_html=True)

        duration = st.slider("Recording duration (seconds)", 3, 10, 5, key="recording_duration")

        if st.button("🎤 Start Recording", key="record_button"):
            st.session_state.recording_data, st.session_state.sample_rate = record_audio_interface(duration)
            if st.session_state.recording_data is not None:
                st.session_state.audio_file = save_and_display_audio(
                    st.session_state.recording_data,
                    st.session_state.sample_rate
                )
                st.session_state.show_result = False

        if st.session_state.audio_file and os.path.exists(st.session_state.audio_file):
            if st.button("🔍 Analyze Emotion", key="analyze_button"):
                with st.spinner("Analyzing emotional tone..."):
                    emotion, probability = predict_emotion(st.session_state.audio_file)
                    st.session_state.show_result = True
                    st.session_state.result = (emotion, probability)

                try:
                    os.unlink(st.session_state.audio_file)
                except:
                    pass

        if st.session_state.show_result and 'result' in st.session_state:
            display_results(*st.session_state.result)

    with tab2:
        st.markdown("""
        <div class='upload-box'>
            <h3>Upload Audio File</h3>
            <p>Select an existing audio file for emotion analysis</p>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose an audio file (WAV, MP3, OGG, FLAC)",
            type=["wav", "mp3", "ogg", "flac"],
            key="file_uploader",
            label_visibility="collapsed"
        )

        if uploaded_file:
            st.session_state.uploaded_file = uploaded_file
            st.audio(uploaded_file)
            st.session_state.show_upload_result = False

        if st.session_state.uploaded_file:
            if st.button("🔍 Analyze Uploaded File", key="analyze_upload"):
                with st.spinner("Analyzing emotional tone..."):
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                            tmp_filename = tmp_file.name
                            tmp_file.write(st.session_state.uploaded_file.getbuffer())

                        emotion, prob = predict_emotion(tmp_filename)
                        st.session_state.show_upload_result = True
                        st.session_state.upload_result = (emotion, prob)

                        try:
                            os.unlink(tmp_filename)
                        except:
                            pass
                    except Exception as e:
                        st.error(f"Upload analysis error: {str(e)}")

        if st.session_state.show_upload_result and 'upload_result' in st.session_state:
            display_results(*st.session_state.upload_result)

    # Information sidebar with enhanced content
    with st.sidebar:
        st.markdown("""
        <div class='info-card'>
            <h3>ℹ️ How It Works</h3>
            <p style="margin-bottom: 1rem;">This AI-powered tool analyzes the acoustic properties of your voice to detect emotional states.</p>
            <ol style="padding-left: 1.5rem;">
                <li style="margin-bottom: 0.5rem;">Record or upload audio containing speech</li>
                <li style="margin-bottom: 0.5rem;">Our system extracts acoustic features</li>
                <li style="margin-bottom: 0.5rem;">AI model analyzes emotional patterns</li>
                <li>You receive detailed emotional analysis</li>
            </ol>
        </div>
        
        <div class='info-card'>
            <h3>🎭 Emotion Spectrum</h3>
            <div style="margin-bottom: 1rem;">
                <p style="margin-bottom: 0.25rem;"><span class='positive'>Positive 😊</span></p>
                <p style="margin: 0 0 0.5rem 1rem;">• Joyfully</p>
                <p style="margin: 0 0 0.5rem 1rem;">• Happy</p>
                <p style="margin: 0 0 0.5rem 1rem;">• Euphoric</p>
            </div>
            <div style="margin-bottom: 1rem;">
                <p style="margin-bottom: 0.25rem;"><span class='neutral'>Neutral 😐</span></p>
                <p style="margin: 0 0 0.5rem 1rem;">• Surprised</p>
            </div>
            <div>
                <p style="margin-bottom: 0.25rem;"><span class='negative'>Negative 😞</span></p>
                <p style="margin: 0 0 0.5rem 1rem;">• Sad</p>
                <p style="margin: 0 0 0.5rem 1rem;">• Angry</p>
                <p style="margin: 0 0 0.5rem 1rem;">• Fear</p>
            </div>
        </div>
        
        <div class='info-card'>
            <h3>💡 Tips for Best Results</h3>
            <p style="margin-bottom: 0.5rem;">• Speak clearly in a quiet environment</p>
            <p style="margin-bottom: 0.5rem;">• Record for 3-5 seconds minimum</p>
            <p style="margin-bottom: 0.5rem;">• Use WAV format for best accuracy</p>
            <p style="margin-bottom: 0.5rem;">• Position microphone 6-12 inches away</p>
            <p>• Avoid background noise</p>
        </div>
        
        <div class='info-card'>
            <h3>⚙️ Technical Details</h3>
            <p style="margin-bottom: 0.5rem;">• Deep Learning LSTM model</p>
            <p style="margin-bottom: 0.5rem;">• 40 MFCC acoustic features</p>
            <p style="margin-bottom: 0.5rem;">• 44.1kHz sample rate</p>
            <p style="margin-bottom: 0.5rem;">• Trained on diverse voice dataset</p>
            <p>• Real-time processing</p>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("""
    <div class='footer'>
        <hr style="border: 0.5px solid #e9ecef; margin: 2rem 0;">
        <p>Sound Emotion Detector v1.0 • Powered by Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()