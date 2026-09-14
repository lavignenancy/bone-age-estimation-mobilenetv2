import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

import model_utils as mu

st.set_page_config(page_title="Bone Age Estimator", page_icon="🦴", layout="centered")

st.title("🦴 Pediatric Bone Age Estimator")
st.caption(
    "Coursework prototype, not a medical device. Estimates skeletal maturity "
    "from a left-hand X-ray. Do not use for real medical decisions."
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("bone_age_model.h5", compile=False)

model = load_model()


def looks_like_hand_xray(img):
    gray = np.array(ImageOps.grayscale(img))
    dark_frac = (gray < 40).mean()
    return dark_frac > 0.01


uploaded = st.file_uploader("Upload a left-hand X-ray image", type=["png", "jpg", "jpeg", "bmp"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded X-ray", use_container_width=True)

    if not looks_like_hand_xray(img):
        st.warning(
            "This image may not look like a standard hand X-ray. "
            "The prediction may be unreliable."
        )
    else:
        x = mu.load_and_preprocess_image(img, mu.IMG_SIZE)
        x = np.expand_dims(x, axis=0)

        scaled_pred = model.predict(x, verbose=0)[0][0]
        months = mu.scaled_to_months(scaled_pred)
        years = months / 12

        st.subheader("Prediction")
        st.metric(
            label="Estimated bone age",
            value=f"{months:.1f} months",
            delta=f"≈ {int(years)} years, {int(months % 12)} months"
        )

        st.info(
            "This is an estimate of **skeletal maturity**, not a diagnosis. "
            "Doctors compare bone age with the child's real age to look for growth issues."
        )
