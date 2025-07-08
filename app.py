import streamlit as st
import cv2
import numpy as np
from PIL import Image
import dlib
import os
import urllib.request
import bz2

# ========== CONFIGURATION ==========
SHAPE_PREDICTOR_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
FACE_ENCODER_URL = "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
SHAPE_PREDICTOR_FILE = "shape_predictor_68_face_landmarks.dat"
FACE_ENCODER_FILE = "dlib_face_recognition_resnet_model_v1.dat"


# ========== TÉLÉCHARGEMENT DES MODÈLES ==========
def download_and_extract_model(url, filename):
    if not os.path.exists(filename):
        st.info(f"Téléchargement de {filename} en cours...")
        compressed_file = filename + ".bz2"
        urllib.request.urlretrieve(url, compressed_file)
        with bz2.BZ2File(compressed_file) as fr, open(filename, "wb") as fw:
            fw.write(fr.read())
        os.remove(compressed_file)
        st.success(f"{filename} téléchargé et prêt ✅")

download_and_extract_model(SHAPE_PREDICTOR_URL, SHAPE_PREDICTOR_FILE)
download_and_extract_model(FACE_ENCODER_URL, FACE_ENCODER_FILE)


# ========== CHARGEMENT DES MODÈLES DLIB ==========
try:
    face_detector = dlib.get_frontal_face_detector()
    shape_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_FILE)
    face_encoder = dlib.face_recognition_model_v1(FACE_ENCODER_FILE)
except Exception as e:
    st.error(f"Erreur lors du chargement des modèles Dlib : {e}")
    st.stop()


# ========== DÉTECTION ==========
def get_face_encodings(image):
    if image is None or image.size == 0:
        raise ValueError("L'image transmise est vide ou invalide.")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Image invalide : doit être RGB avec 3 canaux.")
    if image.dtype != np.uint8:
        raise ValueError("Image invalide : doit être en format 8-bit uint8.")

    dets = face_detector(image, 1)
    encodings = []
    for det in dets:
        shape = shape_predictor(image, det)
        face_descriptor = face_encoder.compute_face_descriptor(image, shape)
        encodings.append(np.array(face_descriptor))
    return encodings, dets


# ========== INTERFACE UTILISATEUR ==========
st.title("🎥 Détection Faciale Simple")
st.markdown("Capturez une photo avec votre webcam pour détecter les visages.")

img_file_buffer = st.camera_input("📸 Prenez une photo")

if img_file_buffer is not None:
    try:
        image = Image.open(img_file_buffer).convert('RGB')
        img_np = np.array(image)

        if img_np is None or img_np.size == 0:
            st.warning("⚠️ L'image capturée est vide.")
            st.stop()

        # ✅ Conversion en uint8 si besoin
        if img_np.dtype != np.uint8:
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

        rgb_image = img_np  # reste en RGB (pas de conversion en BGR ici)

        st.write("✅ Image bien chargée. Dimensions :", rgb_image.shape)

        # Détection des visages
        encodings, dets = get_face_encodings(rgb_image)

        # Annotation
        for det in dets:
            x1, y1, x2, y2 = det.left(), det.top(), det.right(), det.bottom()
            cv2.rectangle(rgb_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        st.image(rgb_image, caption=f"{len(dets)} visage(s) détecté(s)")

    except Exception as e:
        st.error(f"❌ Une erreur est survenue : {e}")
