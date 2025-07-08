import streamlit as st
import dlib
import numpy as np
import os
import cv2
from PIL import Image
import urllib.request
import bz2

# ========== CONFIGURATION ==========
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
ENCODER_PATH = "dlib_face_recognition_resnet_model_v1.dat"

# ========== TÉLÉCHARGEMENT DES FICHIERS LOURDS ==========
def download_and_extract(url, output_path):
    compressed_path = output_path + ".bz2"
    if not os.path.exists(output_path):
        st.info(f"Téléchargement de {output_path}...")
        urllib.request.urlretrieve(url, compressed_path)
        with bz2.open(compressed_path, 'rb') as f_in, open(output_path, 'wb') as f_out:
            f_out.write(f_in.read())
        os.remove(compressed_path)
        st.success(f"{output_path} prêt.")
    else:
        st.info(f"{output_path} déjà présent.")

download_and_extract(
    "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
    PREDICTOR_PATH
)
download_and_extract(
    "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2",
    ENCODER_PATH
)

# ========== INITIALISATION DLIB ==========
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(PREDICTOR_PATH)
face_encoder = dlib.face_recognition_model_v1(ENCODER_PATH)

# ========== FONCTION POUR ENCODER UN VISAGE ==========
def get_face_encoding(image):
    dets = face_detector(image, 1)
    if len(dets) == 0:
        return None
    shape = shape_predictor(image, dets[0])
    return np.array(face_encoder.compute_face_descriptor(image, shape))

# ========== CHARGEMENT DES VISAGES CONNUS ==========
KNOWN_FACES_DIR = "Known_faces"
known_encodings = []
known_names = []

for name in os.listdir(KNOWN_FACES_DIR):
    person_dir = os.path.join(KNOWN_FACES_DIR, name)
    for filename in os.listdir(person_dir):
        img_path = os.path.join(person_dir, filename)
        img = cv2.imread(img_path)
        if img is None:
            continue
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encoding = get_face_encoding(rgb)
        if encoding is not None:
            known_encodings.append(encoding)
            known_names.append(name)

# ========== RECONNAISSANCE ==========
def recognize_face(encoding):
    distances = [np.linalg.norm(encoding - known) for known in known_encodings]
    if len(distances) == 0 or min(distances) > 0.6:
        return "Inconnu"
    return known_names[np.argmin(distances)]

# ========== INTERFACE STREAMLIT ==========
st.title("🧠 Application de Reconnaissance Faciale")
st.write("Chargez une image pour identifier un visage connu.")

uploaded_file = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)
    rgb = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    faces = face_detector(rgb)

    if len(faces) == 0:
        st.warning("Aucun visage détecté.")
    else:
        for face in faces:
            shape = shape_predictor(rgb, face)
            encoding = np.array(face_encoder.compute_face_descriptor(rgb, shape))
            name = recognize_face(encoding)

            x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_np, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        st.image(img_np, caption="Résultat de la reconnaissance", channels="RGB")
