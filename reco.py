import streamlit as st
import cv2
import dlib
import os
import numpy as np
from PIL import Image
from datetime import datetime
import pandas as pd
import time

# Titre
st.title("🎥 Reconnaissance Faciale en Temps Réel")
st.text("Application Streamlit + OpenCV + dlib")

# Chargement des modèles Dlib
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

# Fonction de log
def log_presence(name, log_path="log.csv"):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M")
    if not os.path.exists(log_path):
        df = pd.DataFrame(columns=["Nom", "Date", "Heure"])
        df.to_csv(log_path, index=False)
    df = pd.read_csv(log_path)
    already_logged = ((df["Nom"] == name) &
                      (df["Date"] == date_str) &
                      (df["Heure"] == time_str)).any()
    if not already_logged:
        df.loc[len(df)] = [name, date_str, time_str]
        df.to_csv(log_path, index=False)

# Chargement des visages connus
def load_known_faces(path="known_faces"):
    encodings = []
    names = []
    for name in os.listdir(path):
        person_path = os.path.join(path, name)
        for file in os.listdir(person_path):
            img_path = os.path.join(person_path, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            dets = face_detector(rgb)
            if dets:
                shape = shape_predictor(rgb, dets[0])
                encoding = np.array(face_encoder.compute_face_descriptor(rgb, shape))
                encodings.append(encoding)
                names.append(name)
    return encodings, names

# Initialisation de session
if "known_encodings" not in st.session_state:
    st.session_state.known_encodings, st.session_state.known_names = load_known_faces()

# Sidebar - Ajouter un visage
st.sidebar.markdown("## ➕ Ajouter un nouveau visage")
new_name = st.sidebar.text_input("Nom à enregistrer :", placeholder="ex : Fatou")
if st.sidebar.button("📸 Activer la capture"):
    if new_name.strip():
        st.session_state.capture_mode = True
        st.session_state.capture_name = new_name.strip()
        st.sidebar.success(f"Capture activée pour {new_name.strip()}")
    else:
        st.sidebar.warning("Veuillez saisir un nom valide.")

# Sidebar - Recharger les visages connus
if st.sidebar.button("🔄 Recharger les visages connus"):
    st.session_state.known_encodings, st.session_state.known_names = load_known_faces()
    st.sidebar.success("✅ Visages rechargés")

# Flux vidéo
FRAME_WINDOW = st.image([])
camera = cv2.VideoCapture(0)

while True:
    ret, frame = camera.read()
    if not ret:
        st.error("Erreur de lecture webcam.")
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_detector(rgb)

    for face in faces:
        shape = shape_predictor(rgb, face)
        encoding = np.array(face_encoder.compute_face_descriptor(rgb, shape))

        name = "Inconnu"
        confidence_display = ""

        distances = [np.linalg.norm(encoding - e) for e in st.session_state.known_encodings]
        if distances:
            min_distance = min(distances)
            if min_distance < 0.6:
                index = np.argmin(distances)
                name = st.session_state.known_names[index]
                confidence = max(0, 1.0 - min_distance)
                confidence_display = f" ({int(confidence * 100)}%)"
                log_presence(name)

        # Capture automatique si demandé
        if st.session_state.get("capture_mode", False):
            person_name = st.session_state.get("capture_name", "")
            if person_name:
                save_dir = os.path.join("known_faces", person_name)
                os.makedirs(save_dir, exist_ok=True)
                filename = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jpg"
                path = os.path.join(save_dir, filename)
                cv2.imwrite(path, frame)
                st.sidebar.success(f"✅ Visage capturé dans {save_dir}")
                st.session_state.capture_mode = False

        # Annotation sur image
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"{name}{confidence_display}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    time.sleep(0.05)

camera.release()
