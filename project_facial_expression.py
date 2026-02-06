# app.py
import streamlit as st
from pathlib import Path
from datetime import datetime
import uuid
import cv2

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Upload Photo + Predict", page_icon="📷")

st.title("🧠Facial Expression Prediction")
st.write("Take a photo or upload one, then run it through the CNN model and optionally save it")

# ---------- Settings ----------
MODEL_PATH = "karin_final_best.keras"
IMG_SIZE = (48, 48)
LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Folder where images will be saved (on the computer/server)
SAVE_DIR = Path("saved_images")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

model = get_model()

def save_uploaded_image(file_obj, prefix: str = "img") -> Path:
    original_name = getattr(file_obj, "name", "") or ""
    ext = Path(original_name).suffix.lower() if Path(original_name).suffix else ".jpg"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique = uuid.uuid4().hex[:8]
    filename = f"{prefix}_{timestamp}_{unique}{ext}"
    out_path = SAVE_DIR / filename

    data = file_obj.getvalue()
    out_path.write_bytes(data)
    return out_path


def preprocess_for_model(uploaded_file):
    img = Image.open(uploaded_file).convert("L")
    img_array = np.array(img)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img_array, 1.3, 5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        img_array = img_array[y:y + h, x:x + w]

    final_img = Image.fromarray(img_array).resize((48, 48))
    arr = np.array(final_img, dtype=np.float32) / 255.0
    return np.expand_dims(np.expand_dims(arr, axis=0), axis=-1)
# ---------- UI ----------
st.subheader("1) Take a picture (phone camera)")
camera_photo = st.camera_input("Open camera and take a photo")

st.subheader("2) Or upload from gallery")
gallery_photo = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=False
)

chosen = camera_photo if camera_photo is not None else gallery_photo

if chosen is not None:
    st.image(chosen, caption="Preview", use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🧠 Predict Emotion"):
            try:
                x = preprocess_for_model(chosen)
                predictions = model.predict(x, verbose=0)[0]

                predicted_class_index = np.argmax(predictions)
                confidence = predictions[predicted_class_index]

                if confidence < 0.60:
                    st.warning("Prediction: **Unknown**")
                    st.info(f"Confidence is too low: {confidence:.2%}")
                else:
                    label = LABELS[predicted_class_index]
                    st.success(f"Prediction: **{label}**")
                    st.info(f"Confidence: **{confidence:.2%}**")
                    if label == "Happy":
                        st.balloons()
                # --------------------------------------

                st.bar_chart(dict(zip(LABELS, predictions)))

            except Exception as e:
                st.error(f"Prediction failed: {e}")

    with col2:
        if st.button("💾 Save image to computer"):
            try:
                saved_path = save_uploaded_image(chosen, prefix="phone")
                st.success(f"Saved ✅  {saved_path.resolve()}")
                st.info("The image was saved on the computer running Streamlit (server)")
            except Exception as e:
                st.error(f"Failed to save: {e}")

st.caption(f"Images will be saved to: {SAVE_DIR.resolve()}")
st.caption(f"Model file: {Path(MODEL_PATH).resolve()}")