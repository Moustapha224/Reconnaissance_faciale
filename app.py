import streamlit as st
import cv2
import numpy as np
from PIL import Image
import dlib

# Chargement des modèles (met bien les fichiers dans le même dossier)
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

st.title("🎥 Reconnaissance faciale simplifiée avec Streamlit")

def get_face_encodings(image):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    dets = face_detector(rgb, 1)
    encodings = []
    for det in dets:
        shape = shape_predictor(rgb, det)
        face_descriptor = face_encoder.compute_face_descriptor(rgb, shape)
        encodings.append(np.array(face_descriptor))
    return encodings, dets

img_file_buffer = st.camera_input("📸 Prenez une photo")

if img_file_buffer is not None:
    image = Image.open(img_file_buffer).convert('RGB')
    img_np = np.array(image)[:, :, ::-1]  # RGB to BGR pour OpenCV/dlib
    
    encodings, dets = get_face_encodings(img_np)

    # Annoter les visages détectés
    for det in dets:
        x1, y1, x2, y2 = det.left(), det.top(), det.right(), det.bottom()
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

    st.image(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB), caption=f"{len(dets)} visage(s) détecté(s)")

