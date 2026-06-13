import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json

# පිටුවේ සැකසුම් (Page Configuration)
st.set_page_config(page_title="Smart Fruit Classifier", layout="centered", page_icon="🍎")

st.title("🍎 Smart Fruit Classification System")
st.write("Upload a fruit image, and the AI will classify it into one of the 10 categories!")
st.write("---")

# 1. Model එක සහ Class Names පූරණය කිරීම
@st.cache_resource
def load_my_model():
    # Colab එකෙන් හදපු model එක load කිරීම
    model = tf.keras.models.load_model('fruit_classifier_model.keras')
    return model

try:
    model = load_my_model()
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
except Exception as e:
    st.error(f"Error loading model or class names: {e}")

# 2. Image Upload කරන්න බටන් එකක් සෑදීම
uploaded_file = st.file_uploader("Choose a fruit image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # පින්තූරය තිරය මත පෙන්වීම
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    st.write("🔄 **AI is analyzing the image...**")
    
    # 3. Preprocessing (පින්තූරය 224x224 සයිස් එකට හැරවීම)
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Batch dimension එකක් එකතු කිරීම
    
    # 4. AI Prediction එක සිදු කිරීම
    predictions = model.predict(img_array)
    
    # වැඩිම සම්භාවිතාවක් (Probability) තියෙන පලතුර තෝරාගැනීම
    predicted_class_idx = np.argmax(predictions[0])
    predicted_class = class_names[predicted_class_idx]
    
    # Confidence Score එක (විශ්වාසනීයත්වය) ප්‍රතිශතයක් ලෙස ගැනීම
    confidence = predictions[0][predicted_class_idx] * 100

    st.write("---")
    # 5. ප්‍රතිඵලය ලස්සනට පෙන්වීම
    st.success(f"🎯 AI Prediction: **{predicted_class}**")
    st.info(f"📊 Confidence Score: **{confidence:.2f}%**")

    # හැම පලතුරු වර්ගයකටම තියෙන ප්‍රතිශතයන් Graph එකක් මගින් පෙන්වීම
    st.subheader("📈 Class Probabilities")
    for i, class_name in enumerate(class_names):
        prob = predictions[0][i] * 100
        st.write(f"{class_name}: {prob:.2f}%")
        st.progress(int(prob))
