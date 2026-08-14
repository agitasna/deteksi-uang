import os
import time
import cv2
import numpy as np
import tensorflow as tf
import gdown


# GOOGLE DRIVE

MODEL_URL = (
    "https://drive.google.com/uc?id="
    "1cuDYSRLrpqw4s_xIa5D1Sk9wygrx4bWR"
)

def download_model():
    if os.path.exists(MODEL_PATH):
        return
        
    os.makedirs(
        "model",
        exist_ok=True
    )

    print(
        "Model belum tersedia. "
        "Mengunduh model dari Google Drive..."
    )

    gdown.download(
        MODEL_URL,
        MODEL_PATH,
        quiet=False
    )

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model gagal diunduh dari Google Drive."
        )

def load_model():
    download_model()

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

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

    return model, class_names


# FORMAT NOMINAL

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


# FORMAT NOMINAL UNTUK SUARA

def nominal_to_speech(label):
    nominal_map = {
        "1rb": "seribu rupiah",
        "2rb": "dua ribu rupiah",
        "5rb": "lima ribu rupiah",
        "10rb": "sepuluh ribu rupiah",
        "20rb": "dua puluh ribu rupiah",
        "50rb": "lima puluh rupiah",
        "100rb": "seratus ribu rupiah"
    }

    return nominal_map.get(
        label,
        label
    )


# PREDIKSI

def predict_image(
    model,
    image,
    class_names
):

    start_time = time.perf_counter()

    if image is None:
        raise ValueError(
            "Gambar tidak tersedia."
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    
    # RESIZE

    image = cv2.resize(
        image,
        (224, 224)
    )

    image = image.astype(np.float32)

    image = np.expand_dims(
        image,
        axis=0
    )


    # PREDICTION

    probability = model.predict(
        image,
        verbose=0
    )[0]

    index = int(
        np.argmax(
            probability
        )
    )

    prediction = class_names[index]


    # CONFIDENCE
    
    confidence = float(
        probability[index] * 100
    )


    # INFERENCE TIME

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
