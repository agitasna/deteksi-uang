import os
import time
import threading

import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import gdown


# ======================================================
# PATH MODEL
# ======================================================

MODEL_PATH = "model/model_uang.keras"
LABEL_PATH = "model/labels.txt"

# ======================================================
# GOOGLE DRIVE
# ======================================================

GOOGLE_DRIVE_FILE_ID = (
    "1cuDYSRLrpqw4s_xIa5D1Sk9wygrx4bWR"
)


# ======================================================
# DOWNLOAD MODEL
# ======================================================

def download_model():

    if os.path.exists(MODEL_PATH):

        print(
            "Model sudah tersedia secara lokal."
        )

        return


    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True
    )


    print(
        "Model tidak ditemukan."
    )

    print(
        "Mengunduh model dari Google Drive..."
    )


    url = (
        "https://drive.google.com/uc?id="
        + GOOGLE_DRIVE_FILE_ID
    )


    gdown.download(
        url,
        MODEL_PATH,
        quiet=False
    )


    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            "Model gagal diunduh dari Google Drive."
        )


    print(
        "Model berhasil diunduh."
    )


# ======================================================
# LOAD MODEL
# ======================================================

def load_model():

    # ----------------------------------------------
    # DOWNLOAD MODEL JIKA BELUM ADA
    # ----------------------------------------------

    download_model()


    # ----------------------------------------------
    # LOAD CNN
    # ----------------------------------------------

    print(
        "Memuat model CNN..."
    )


    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )


    # ----------------------------------------------
    # LOAD LABEL
    # ----------------------------------------------

    with open(
        LABEL_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        class_names = [
            line.strip()
            for line in file.readlines()
            if line.strip()
        ]


    print(
        "Model berhasil dimuat."
    )


    return model, class_names


# ======================================================
# TEXT TO SPEECH
# ======================================================

def _speak_text(text):

    try:

        engine = pyttsx3.init()

        engine.setProperty(
            "rate",
            150
        )

        engine.setProperty(
            "volume",
            1.0
        )

        engine.say(text)

        engine.runAndWait()

        engine.stop()

    except Exception as e:

        print(
            "TTS Error:",
            e
        )


# ======================================================
# SPEAK MESSAGE
# ======================================================

def speak_message(message):

    thread = threading.Thread(
        target=_speak_text,
        args=(message,),
        daemon=True
    )

    thread.start()


# ======================================================
# FORMAT NOMINAL
# ======================================================

def format_nominal(label):

    nominal_map = {

        "1rb": "Rp1.000",

        "2rb": "Rp2.000",

        "5rb": "Rp5.000",

        "10rb": "Rp10.000",

        "20rb": "Rp20.000",

        "50rb": "Rp50.000",

        "100rb": "Rp100.000"

    }

    return nominal_map.get(
        label,
        label
    )


# ======================================================
# FORMAT NOMINAL UNTUK SUARA
# ======================================================

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


# ======================================================
# SPEAK HASIL PREDIKSI
# ======================================================

def speak(label):

    text = nominal_to_speech(label)

    speak_message(text)


# ======================================================
# PREDICT IMAGE
# ======================================================

def predict_image(
    model,
    image,
    class_names
):

    start_time = time.perf_counter()


    # ----------------------------------------------
    # VALIDASI IMAGE
    # ----------------------------------------------

    if image is None:

        raise ValueError(
            "Gambar tidak tersedia."
        )


    # ----------------------------------------------
    # BGR → RGB
    # ----------------------------------------------

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # ----------------------------------------------
    # RESIZE
    # ----------------------------------------------

    image = cv2.resize(
        image,
        (224, 224)
    )


    # ----------------------------------------------
    # FLOAT32
    # ----------------------------------------------

    image = image.astype(
        np.float32
    )


    # ----------------------------------------------
    # BATCH
    # ----------------------------------------------

    image = np.expand_dims(
        image,
        axis=0
    )


    # ----------------------------------------------
    # PREDICTION
    # ----------------------------------------------

    probability = model.predict(
        image,
        verbose=0
    )[0]


    # ----------------------------------------------
    # CLASS INDEX
    # ----------------------------------------------

    index = int(
        np.argmax(
            probability
        )
    )


    # ----------------------------------------------
    # LABEL
    # ----------------------------------------------

    prediction = class_names[index]


    # ----------------------------------------------
    # CONFIDENCE
    # ----------------------------------------------

    confidence = float(
        probability[index] * 100
    )


    # ----------------------------------------------
    # INFERENCE TIME
    # ----------------------------------------------

    inference = (
        time.perf_counter()
        - start_time
    ) * 1000


    return (
        prediction,
        confidence,
        probability,
        inference
    )


# ======================================================
# ADD HISTORY
# ======================================================

def add_history(
    history,
    prediction,
    confidence
):

    history = list(history)

    history.append({

        "prediction": prediction,

        "confidence": confidence,

        "timestamp": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        )

    })

    return history