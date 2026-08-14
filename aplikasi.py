import cv2
import numpy as np
import streamlit as st

from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase
)

from voice_component import voice_recognition

from utils import (
    load_model,
    predict_image,
    format_nominal
)

st.set_page_config(
    page_title="Deteksi Nominal Uang Rupiah",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CSS

with open(
    "ui.css",
    encoding="utf-8"
) as css:

    st.markdown(
        f"<style>{css.read()}</style>",
        unsafe_allow_html=True
    )


# MODEL

@st.cache_resource
def load_cnn():

    return load_model()


model, class_names = load_cnn()

default_state = {
    "image": None,
    "prediction": None,
    "confidence": 0,
    "probability": None,
    "inference": 0,
    "history": [],
    "last_command_id": None,
    "tts_text": None,
    "tts_id": None
}


for key, value in default_state.items():
    if key not in st.session_state:
        st.session_state[key] = value


# TTS

def browser_speak(text):
    """
    TTS menggunakan Web Speech API pada browser.

    Berbeda dengan pyttsx3:
    - pyttsx3 berjalan di server
    - Web Speech API berjalan di browser pengguna
    """
    if not text:

        return
        
    escaped_text = (
        text
        .replace("\\", "\\\\")
        .replace("'", "\\'")
        .replace("\n", " ")
    )

    st.components.v1.html(
        f"""
        <script>
        const text = '{escaped_text}';
        
        if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            
            const speech =
                new SpeechSynthesisUtterance(text);
            speech.lang = 'id-ID';
            speech.rate = 0.9;
            speech.pitch = 1.0;
            speech.volume = 1.0;
            window.speechSynthesis.speak(speech);
        }}
        
        </script>
        """,
        height=0
    )


# VOICE

def nominal_to_speech(label):
    
    nominal_map = {
        "1rb": "seribu rupiah",
        "2rb": "dua ribu rupiah",
        "5rb": "lima ribu rupiah",
        "10rb": "sepuluh ribu rupiah",
        "20rb": "dua puluh ribu rupiah",
        "50rb": "lima puluh ribu rupiah",
        "100rb": "seratus ribu rupiah"
    }

    return nominal_map.get(
        label,
        label
    )


# SIDE-BAR

with st.sidebar:
    st.markdown("# 💵 Deteksi Uang")
    st.caption("CNN Classification")
    st.divider()

    mode = st.radio(
        "Mode",
        [
            "📷 Webcam",
            "📁 Upload Gambar"
        ]
    )

    st.divider()
    st.subheader("💰 Kelas")

    for cls in class_names:
        st.write("•", format_nominal(cls))

if mode == "📁 Upload Gambar":
    st.session_state.last_command_id = None


# HEADER

st.markdown(
    """
    <div class="hero">
        <h1>Deteksi Nominal Uang Rupiah</h1>
        <p>
            Aplikasi klasifikasi nominal uang Rupiah
            menggunakan Convolutional Neural Network (CNN)
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# LAYOUT

top_left, top_right = st.columns(
    [1.5, 1],
    gap="large"
)


# KIRI

with top_left:
    st.markdown("## 📷 Input Gambar")


    # MODE WEBCAM

    if mode == "📷 Webcam":
        st.info(
            'Aktifkan kamera. Setelah kamera aktif, '
            'katakan **"Ambil foto"** untuk mengambil gambar.'
        )

        class Camera(VideoProcessorBase):
            def __init__(self):
                self.frame = None

            def recv(self, frame):
                self.frame = frame.to_ndarray(
                    format="bgr24"
                )
                
                return frame

        ctx = webrtc_streamer(
            key="camera",
            media_stream_constraints={
                "video": True,
                "audio": False
            },

            video_processor_factory=Camera,
            async_processing=True,
            desired_playing_state=True
        )

        if ctx.state.playing:
            st.success("📷 Kamera aktif.")


            # VOICE-RECOG
            
            voice_command = voice_recognition()

            if voice_command is not None:

                if isinstance(
                    voice_command,
                    dict
                ):

                    command = voice_command.get(
                        "command"
                    )

                    command_id = voice_command.get(
                        "id"
                    )

                    if command == "ambil_foto":
                        if (
                            command_id
                            !=
                            st.session_state.last_command_id
                        ):
                            st.session_state.last_command_id = (
                                command_id
                            )


                            # FRAME WEBCAM

                            if (
                                ctx.video_processor
                                is not None
                                and
                                ctx.video_processor.frame
                                is not None
                            ):

                                st.session_state.image = (
                                    ctx.video_processor.frame.copy()
                                )

                                st.success("📸 Foto berhasil diambil.")

                                
                                # PREDIKSI

                                with st.spinner("Sedang melakukan prediksi..."):
                                    (
                                        prediction,
                                        confidence,
                                        probability,
                                        inference
                                    ) = predict_image(
                                        model,
                                        st.session_state.image,
                                        class_names
                                    )


                                # HASIL
                                
                                st.session_state.prediction = (prediction)
                                st.session_state.confidence = (confidence)
                                st.session_state.probability = (probability)
                                st.session_state.inference = (inference)

                                
                                # TTS BROWSER

                                speech_text = (
                                    nominal_to_speech(
                                        prediction
                                    )
                                )

                                st.session_state.tts_text = (speech_text)
                                st.session_state.tts_id = (command_id)

                            else:
                                st.warning(
                                    "Frame kamera belum tersedia. "
                                    "Silakan tunggu sebentar lalu "
                                    "katakan ambil foto kembali."
                                )

                else:
                    if voice_command == "ambil_foto":
                        if (
                            st.session_state.last_command_id
                            !=
                            "legacy_ambil_foto"
                        ):
                            st.session_state.last_command_id = (
                                "legacy_ambil_foto"
                            )

                            if (
                                ctx.video_processor
                                is not None
                                and
                                ctx.video_processor.frame
                                is not None
                            ):
                                st.session_state.image = (
                                    ctx.video_processor.frame.copy()
                                )
                                
                                st.success("📸 Foto berhasil diambil.")


                                # PREDIKSI

                                with st.spinner("Sedang melakukan prediksi..."):
                                    (
                                        prediction,
                                        confidence,
                                        probability,
                                        inference
                                    ) = predict_image(
                                        model,
                                        st.session_state.image,
                                        class_names
                                    )


                                st.session_state.prediction = (prediction)
                                st.session_state.confidence = (confidence)
                                st.session_state.probability = (probability)
                                st.session_state.inference = (inference)

                                # ==============================
                                # TTS
                                # ==============================

                                speech_text = (
                                    nominal_to_speech(
                                        prediction
                                    )
                                )

                                st.session_state.tts_text = (speech_text)
                                st.session_state.tts_id = ("legacy_ambil_foto")


    # MODE UPLOAD

    else:
        uploaded_file = st.file_uploader(
            "Upload gambar uang Rupiah",
            type=["jpg", "jpeg", "png", "bmp", "webp"]
        )

        if uploaded_file is not None:
            file_bytes = np.asarray(
                bytearray(
                    uploaded_file.read()
                ),
                dtype=np.uint8
            )

            image = cv2.imdecode(
                file_bytes,
                cv2.IMREAD_COLOR
            )

            st.session_state.image = image


    # PREVIEW

    if st.session_state.image is not None:
        st.markdown("---")
        st.markdown("## 🖼️ Preview")

        preview = cv2.cvtColor(
            st.session_state.image,
            cv2.COLOR_BGR2RGB
        )

        st.image(
            preview,
            use_container_width=True
        )

        
        # PREDIKSI UPLOAD

        if mode == "📁 Upload Gambar":
            
            if st.button(
                "🔍 Prediksi Nominal",
                type="primary",
                use_container_width=True
            ):
                with st.spinner(
                    "Sedang melakukan prediksi..."
                ):
                    (
                        prediction,
                        confidence,
                        probability,
                        inference

                    ) = predict_image(
                        model,
                        st.session_state.image,
                        class_names
                    )


                # HASIL

                st.session_state.prediction = (prediction)
                st.session_state.confidence = (confidence)
                st.session_state.probability = (probability)
                st.session_state.inference = (inference)

                
                # TTS BROWSER

                speech_text = (
                    nominal_to_speech(
                        prediction
                    )
                )

                st.session_state.tts_text = (speech_text)
                st.session_state.tts_id = (f"upload_{prediction}")
                
                if st.session_state.tts_text is not None:
                    browser_speak(st.session_state.tts_text)


# ======================================================
# BROWSER TTS
# ======================================================

if st.session_state.tts_text is not None:

    browser_speak(
        st.session_state.tts_text
    )


# KANAN

with top_right:
    st.markdown("## 🤖 Hasil Prediksi")

    if st.session_state.prediction is None:
        st.info("Belum ada hasil prediksi.")

    else:
        nominal = format_nominal(st.session_state.prediction)
        confidence = (st.session_state.confidence)
        inference = (st.session_state.inference)
        
        st.metric(
            label="💵 Nominal Terdeteksi", 
            value=nominal
        )


        # CONFIDENCE & INFERENCE

        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="Confidence",
                value=f"{confidence:.2f}%"
            )

        with col2:
            st.metric(
                label="Inference",
                value=f"{inference:.2f} ms"
            )


        # CONFIDENCE 

        st.markdown("### 🎯 Tingkat Keyakinan")
        st.progress(
            min(
                max(
                    confidence / 100, 
                    0
                ),
                1
            )
        )

        st.caption(f"{confidence:.2f}%")
