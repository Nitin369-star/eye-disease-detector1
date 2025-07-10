# app.py
import streamlit as st
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
from deep_translator import GoogleTranslator
import requests
import pyttsx3
import whisper
from streamlit_mic_recorder import mic_recorder
import tempfile
import base64
import os
from fpdf import FPDF
import qrcode
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import datetime
import openai

# ----------------------------
# 🎯 App Config & Title
# ----------------------------
st.set_page_config(page_title="Eye Disease Detector", layout="centered")
st.title("🧠 Eye Disease Detector")
st.caption("📸 Upload retina image → 🧠 Detect disease → 💊 Get treatment advice")

# ----------------------------
# 🌐 Language Selector
# ----------------------------
language = st.sidebar.radio("🌍 Choose Language", ["English", "Hindi"])

# ----------------------------
# 📤 Sidebar File Uploader
# ----------------------------
uploaded_file = st.sidebar.file_uploader("📥 Upload Retina Image", type=["jpg", "jpeg", "png"])

# ----------------------------
# 📷 Webcam Capture Tab
# ----------------------------
def capture_webcam_image():
    class VideoProcessor(VideoTransformerBase):
        def __init__(self):
            self.frame = None

        def transform(self, frame):
            self.frame = frame.to_ndarray(format="bgr24")
            return self.frame

    ctx = webrtc_streamer(key="example", video_processor_factory=VideoProcessor)
    if ctx.video_processor:
        if st.button("📸 Capture Frame"):
            img = ctx.video_processor.frame
            if img is not None:
                image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                st.image(image, caption="🖼️ Captured Image", use_container_width=True)
                return image
    return None

# ----------------------------
# 📦 Load Model and Labels
# ----------------------------
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

# ----------------------------
# 📚 Disease Info
# ----------------------------
disease_info = {
    "Normal": {
        "desc": "The retina appears healthy with no visible signs of disease.",
        "treat": "No treatment needed. Maintain regular checkups and a healthy lifestyle."
    },
    "Glaucoma": {
        "desc": "A condition that damages the optic nerve, often due to high eye pressure.",
        "treat": "Use of eye drops, oral medications, or surgery to reduce eye pressure."
    },
    "Diabetic Retinopathy": {
        "desc": "Damage to the retina caused by complications of diabetes.",
        "treat": "Control blood sugar levels, laser therapy, or injections."
    },
    "Cataract": {
        "desc": "Clouding of the eye’s natural lens that leads to blurry vision.",
        "treat": "Surgical replacement of the cloudy lens with an artificial one."
    },
    "AMD": {
        "desc": "Age-related macular degeneration affects central vision due to retinal damage.",
        "treat": "Anti-VEGF injections, laser therapy, or dietary supplements."
    },
    "Papilledema": {
        "desc": "Swelling of the optic disc caused by increased intracranial pressure.",
        "treat": "Treat underlying cause: brain imaging, surgery, or medication."
    },
    "Pseudopapilledema": {
        "desc": "Optic disc elevation that mimics papilledema but is usually benign.",
        "treat": "Monitoring; no treatment unless associated with another condition."
    }
}

# ----------------------------
# 📄 PDF Generator
# ----------------------------
def generate_pdf(patient_name, patient_age, image, disease, confidence, desc, treat, qr_type="info"):
    pdf = FPDF()
    pdf.add_page()
    
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Eye Disease Detection Report", ln=True, align="C")

    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Name: {patient_name}", ln=True)
    pdf.cell(0, 10, f"Age: {patient_age}", ln=True)
    pdf.cell(0, 10, f"Date: {st.session_state.get('timestamp', '')}", ln=True)

    # Save retina image temporarily
    img_path = "temp_retina.jpg"
    image.save(img_path)
    pdf.image(img_path, x=60, y=None, w=90)
    os.remove(img_path)

    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Predicted Disease: {disease}", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Confidence: {confidence:.2%}", ln=True)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Description:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, desc)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Recommended Treatment:", ln=True)
    pdf.set_font("Arial", "", 12)
    pdf.multi_cell(0, 8, treat)

    # QR Code
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Scan for More Info:", ln=True)
    qr_url = f"https://www.google.com/search?q={'+'.join(disease.split())}+eye+disease" if qr_type == "info" else "https://www.google.com/search?q=eye+hospital+near+me"
    qr = qrcode.make(qr_url)
    qr_path = "qr_temp.png"
    qr.save(qr_path)
    pdf.image(qr_path, x=70, y=None, w=70)
    os.remove(qr_path)

    pdf_path = "Eye_Report.pdf"
    pdf.output(pdf_path)
    return pdf_path

# ----------------------------
# 🚀 Main App Logic
# ----------------------------
# ----------------------------
# 🚀 Image Input Mode Selector
# ----------------------------
st.markdown("---")
st.header("🖼️ Retina Image Input")
input_mode = st.radio("Select Input Mode:", ["Single Image", "Multiple Images (Batch)"], horizontal=True)

# Load patient details
st.markdown("---")
st.header("🧑 Patient Information")
col1, col2 = st.columns(2)
with col1:
    patient_name = st.text_input("👤 Name", "")
with col2:
    patient_age = st.text_input("🎂 Age", "")

# ---------------
# 🔁 Prediction Function
# ---------------
def predict_image(image):
    image = image.resize((224, 224))
    img_array = np.asarray(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence = prediction[0][index]
    raw_class = class_name
    clean_class = ' '.join(raw_class.split(' ')[1:]).replace('_', ' ')
    selected = clean_class.strip().title()
    info = disease_info.get(selected, {"desc": "No info available", "treat": "N/A"})
    return selected, confidence, info

# ----------------------------
# 📸 SINGLE IMAGE MODE
# ----------------------------
if input_mode == "Single Image":
    input_method = st.radio("📷 Select Image Input Method", ["Upload from device", "Capture with webcam (demo)"])

    image = None
    if input_method == "Upload from device":
        uploaded_file = st.file_uploader("📥 Upload Retina Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
    elif input_method == "Capture with webcam (demo)":
        image = capture_webcam_image()

    if image:
        st.image(image, caption="🖼️ Input Image", use_container_width=True)
        selected, confidence, info = predict_image(image)

        if language == "Hindi":
            translator = GoogleTranslator(source='auto', target='hi')
            translated_desc = translator.translate(info['desc'])
            translated_treat = translator.translate(info['treat'])

        tab1, tab2 = st.tabs(["🔍 Prediction", "📌 Explanation"])
        with tab1:
            st.subheader("🔍 Prediction Result")
            st.write(f"🩺 **Detected Disease:** {selected}")
            st.write(f"📊 **Confidence:** {confidence:.2%}")
        with tab2:
            st.subheader("🧠 Disease Explanation")
            if language == "English":
                st.write(f"📌 {info['desc']}")
                st.write(f"💊 {info['treat']}")
            else:
                st.write(f"📌 {translated_desc}")
                st.write(f"💊 {translated_treat}")

        if st.button("📄 Generate PDF Report"):
            st.session_state["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pdf_path = generate_pdf(patient_name, patient_age, image, selected, confidence, info['desc'], info['treat'])
            with open(pdf_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                href = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="Eye_Report.pdf">📥 Download Report</a>'
                st.markdown(href, unsafe_allow_html=True)
            os.remove(pdf_path)

# ----------------------------
# 📦 BATCH MODE
# ----------------------------
elif input_mode == "Multiple Images (Batch)":
    uploaded_files = st.file_uploader("📥 Upload Retina Images (Multiple)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        st.markdown("### 🔍 Batch Prediction Results")
        records = []
        for i, file in enumerate(uploaded_files):
            image = Image.open(file).convert("RGB")
            selected, confidence, info = predict_image(image)
            records.append({
                "Image": file.name,
                "Prediction": selected,
                "Confidence": f"{confidence:.2%}"
            })
            st.image(image, caption=f"🖼️ {file.name}", use_container_width=True)
            st.write(f"🩺 **Prediction:** {selected}")
            st.write(f"📊 **Confidence:** {confidence:.2%}")
            st.write(f"📌 **Description:** {info['desc']}")
            st.write(f"💊 **Treatment:** {info['treat']}")

            if st.button(f"📄 Generate PDF for {file.name}", key=f"pdf_{i}"):
                st.session_state["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                pdf_path = generate_pdf(patient_name, patient_age, image, selected, confidence, info['desc'], info['treat'])
                with open(pdf_path, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                    href = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="Report_{file.name}.pdf">📥 Download Report</a>'
                    st.markdown(href, unsafe_allow_html=True)
                os.remove(pdf_path)

        # Show Summary Table
        st.markdown("### 📊 Summary Table")
        st.dataframe(records)

# ----------------------------
# 💬 Voice Assistant
# 💬 Voice Assistant
st.markdown("---")
st.header("💬 Ask About Eye Diseases (Voice Assistant)")
audio = mic_recorder(start_prompt="🎙️ Click to Record", stop_prompt="⏹️ Stop Recording", key="voice")

if audio:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        tmp_audio.write(audio["bytes"])
        tmp_audio_path = tmp_audio.name
    st.audio(audio["bytes"], format="audio/wav")

    with st.spinner("🧠 Transcribing..."):
        model_whisper = whisper.load_model("base")
        result = model_whisper.transcribe(tmp_audio_path)
        question = result["text"]
    st.success(f"🔊 You said: {question}")

    # ✅ Secure API key use
    openai.api_key = st.secrets["openai"]["openai_api_key"]

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant specialized in eye diseases."},
            {"role": "user", "content": question}
        ],
        temperature=0.7,
        max_tokens=200
    )

    answer = response['choices'][0]['message']['content']
    st.markdown("### 🤖 Assistant's Answer:")
    st.write(answer)

    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(answer)
    engine.runAndWait()
    os.remove(tmp_audio_path)
else:
    st.info("Click the mic and ask your eye disease question.")
