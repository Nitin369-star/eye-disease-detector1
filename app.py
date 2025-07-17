import streamlit as st
st.set_page_config(page_title="Eye Disease Detector", layout="centered")
from tensorflow.keras.models import load_model
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
from openai import OpenAI
import tensorflow as tf
import matplotlib.cm as cm 
import matplotlib.pyplot as plt
import pandas as pd

RECORDS_FILE = "patient_records.csv"

# ✅ Initialize CSV if missing or empty
if not os.path.exists(RECORDS_FILE) or os.stat(RECORDS_FILE).st_size == 0:
    df = pd.DataFrame(columns=["Timestamp", "Name", "Age", "Phone", "Email", "Disease", "Confidence"])
    df.to_csv(RECORDS_FILE, index=False)


# 🌍 Get User Location
def get_location():
    try:
        ip_info = requests.get("https://ipinfo.io").json()
        loc = ip_info['loc'].split(",")
        latitude = loc[0]
        longitude = loc[1]
        return latitude, longitude
    except Exception as e:
        st.error("❌ Failed to get location.")
        return None, None


# ----------------------------
# 🌐 Language Selector
# ----------------------------
language = st.sidebar.radio("🌍 Choose Language / भाषा चुनें", ["English", "Hindi"])

# ----------------------------
# ----------------------------# ----------------------------
# 🎯 App Title & Caption
# ----------------------------
if language == "Hindi":
    st.title("🧠 आंख की बीमारी डिटेक्टर")
    st.caption("📸 रेटिना इमेज अपलोड करें → 🧠 बीमारी पहचानें → 💊 उपचार सलाह प्राप्त करें")
else:
    st.title("🧠 Eye Disease Detector")
    st.caption("📸 Upload retina image → 🧠 Detect disease → 💊 Get treatment advice")


# ----------------------------
# 📤 Sidebar File Uploader
# ----------------------------
if language == "Hindi":
    uploaded_file = st.sidebar.file_uploader("📥 रेटिना इमेज अपलोड करें", type=["jpg", "jpeg", "png"])
else:
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
        btn_label = "📸 फ्रेम कैप्चर करें" if language == "Hindi" else "📸 Capture Frame"
        if st.button(btn_label):
            img = ctx.video_processor.frame
            if img is not None:
                image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                st.image(image, caption="🖼️ कैप्चर की गई इमेज" if language == "Hindi" else "🖼️ Captured Image", use_container_width=True)
                return image
    return None

# ----------------------------
# ----------------------------
# 📦 Load Model and Labels
# ----------------------------

# Load the full model
model = load_model("keras_model.h5")

# Extract MobileNetV2 feature extractor
feature_model = model.get_layer("sequential_1").get_layer("model1")

# Extract classifier head
classifier_head = model.get_layer("sequential_3")

# This is the last conv layer in MobileNetV2
last_conv_layer_name = "Conv_1"

# Load class names
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

def translate_text(text):
    try:
        return GoogleTranslator(source='auto', target='hi').translate(text)
    except Exception:
        return text

# ----------------------------
# 📄 PDF Generator
# ----------------------------
def generate_pdf(patient_name, patient_age, image, predictions, lang="English", qr_type="info", gradcam_image=None):
    pdf = FPDF()
    pdf.add_page()

    # Text labels
    if lang == "Hindi":
        title = "आंख की बीमारी डिटेक्शन रिपोर्ट"
        name_label = "नाम"
        age_label = "आयु"
        date_label = "तारीख"
        disease_label = "पहचानी गई बीमारियाँ"
        confidence_label = "विश्वास स्तर"
        qr_label = "अधिक जानकारी के लिए स्कैन करें"
        gradcam_label = "मॉडल का फोकस (Grad-CAM)"
    else:
        title = "Eye Disease Detection Report"
        name_label = "Name"
        age_label = "Age"
        date_label = "Date"
        disease_label = "Detected Diseases"
        confidence_label = "Confidence"
        qr_label = "Scan for More Info"
        gradcam_label = "Model Focus (Grad-CAM)"

    # Header
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, ln=True, align="C")

    # Patient Info
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"{name_label}: {patient_name}", ln=True)
    pdf.cell(0, 10, f"{age_label}: {patient_age}", ln=True)
    pdf.cell(0, 10, f"{date_label}: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

    # Retina image
    img_path = "temp_retina.jpg"
    image.save(img_path)
    pdf.image(img_path, x=60, y=None, w=90)
    os.remove(img_path)

    pdf.ln(10)
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"{disease_label}:", ln=True)

    pdf.set_font("Arial", "", 12)
    for pred in predictions:
        pdf.cell(0, 10, f"{pred['disease']} - {pred['confidence']:.2%}", ln=True)

    # 📊 Pie chart
    labels = [d['disease'] for d in predictions]
    values = [d['confidence'] for d in predictions]
    chart_path = "chart.png"
    plt.figure(figsize=(4, 4))
    plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title("Disease Confidence Chart")
    plt.tight_layout()
    plt.savefig(chart_path)
    plt.close()
    pdf.image(chart_path, x=40, y=None, w=120)
    os.remove(chart_path)

    # 🔥 Grad-CAM (after pie chart)
    if gradcam_image:
        gradcam_path = "gradcam_temp.jpg"
        gradcam_image.save(gradcam_path)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, gradcam_label, ln=True)
        pdf.image(gradcam_path, x=60, y=None, w=90)
        os.remove(gradcam_path)

    # 📌 QR Code
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, f"{qr_label}:", ln=True)
    if qr_type == "info":
        qr_url = "https://www.google.com/search?q=eye+diseases+info"
    else:
        lat, lon = get_location()
        if lat and lon:
            qr_url = f"https://www.google.com/maps?q=eye+hospital+near+{lat},{lon}"
        else:
            qr_url = "https://www.google.com/search?q=eye+hospital+near+me"

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
st.markdown("---")
st.header("🖼️ Retina Image Input" if language == "English" else "🖼️ रेटिना इमेज इनपुट")
input_mode = st.radio(
    "Select Input Mode:" if language == "English" else "इनपुट मोड चुनें:",
    ["Single Image" if language == "English" else "एकल इमेज",
     "Multiple Images (Batch)" if language == "English" else "एकाधिक इमेज (बैच)"],
    horizontal=True
)

st.markdown("---")
st.header("🧑 Patient Information" if language == "English" else "🧑 मरीज की जानकारी")
col1, col2 = st.columns(2)
with col1:
    patient_name = st.text_input("👤 Name" if language == "English" else "👤 नाम", "")
with col2:
    patient_age = st.text_input("🎂 Age" if language == "English" else "🎂 आयु", "")
    
# 🔁 Prediction Function

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

def make_gradcam_heatmap(img_array, feature_model, classifier_head, last_conv_layer_name, pred_index=None):
    # Create a model that maps input to last conv layer and feature output
    grad_model = tf.keras.models.Model(
        inputs=feature_model.input,
        outputs=[
            feature_model.get_layer(last_conv_layer_name).output,
            feature_model.output  # This gives (1, 7, 7, 1280)
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, conv_features = grad_model(img_array)
        tape.watch(conv_outputs)

        # 🟡 Apply Global Average Pooling before classifier
        gap_features = tf.reduce_mean(conv_outputs, axis=[1, 2])  # -> (1, 1280)

        predictions = classifier_head(gap_features)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Compute gradients
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + 1e-10)
    return heatmap.numpy()


def overlay_heatmap_on_image(img_pil, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = Image.fromarray((jet_heatmap * 255).astype(np.uint8))
    jet_heatmap = jet_heatmap.resize(img_pil.size)
    blended = Image.blend(img_pil.convert("RGB"), jet_heatmap, alpha=alpha)
    return blended


# ----------------------------
# 📸 SINGLE IMAGE MODE
# ----------------------------
if (input_mode == "Single Image" and language == "English") or (input_mode == "एकल इमेज" and language == "Hindi"):
    input_method = st.radio(
        "📷 Select Image Input Method" if language == "English" else "📷 इमेज इनपुट विधि चुनें",
        ["Upload from device" if language == "English" else "डिवाइस से अपलोड करें",
         "Capture with webcam (demo)" if language == "English" else "वेबकैम से कैप्चर करें"]
    )

    image = None
    if (input_method == "Upload from device" and language == "English") or (input_method == "डिवाइस से अपलोड करें" and language == "Hindi"):
        uploaded_file = st.file_uploader(
          "📥 Upload Retina Image" if language == "English" else "📥 रेटिना इमेज अपलोड करें", 
          type=["jpg", "jpeg", "png"],
          key="single_upload"
       )   
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
    elif (input_method == "Capture with webcam (demo)" and language == "English") or (input_method == "वेबकैम से कैप्चर करें" and language == "Hindi"):
        image = capture_webcam_image()

    if image:
        st.image(image, caption="🖼️ Input Image" if language == "English" else "🖼️ इनपुट इमेज", use_container_width=True)
        selected, confidence, info = predict_image(image)
        img_resized = image.resize((224, 224))
        img_array = np.asarray(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

# 🔥 Generate Grad-CAM heatmap
        heatmap = make_gradcam_heatmap(
         img_array,
         feature_model=feature_model,
         classifier_head=classifier_head,
         last_conv_layer_name="Conv_1"
        )

# Replace with your actual last conv layer name
        gradcam_image = overlay_heatmap_on_image(image, heatmap)

# 📸 Display Grad-CAM image
        st.image(gradcam_image, caption="🔥 Grad-CAM Heatmap (Model Focus)", use_container_width=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        phone = st.session_state.get("phone", "")
        email = st.session_state.get("email", "")

        # Append prediction to CSV
        record = {
           "Timestamp": timestamp,
           "Name": patient_name,
           "Age": patient_age,
           "Phone": phone,
           "Email": email,
           "Disease": selected,
           "Confidence": round(confidence * 100, 2)
         }

        df = pd.read_csv(RECORDS_FILE)
        df = pd.concat([df, pd.DataFrame([record])], ignore_index=True)
        df.to_csv(RECORDS_FILE, index=False)


        if language == "Hindi":
            translated_desc = translate_text(info['desc'])
            translated_treat = translate_text(info['treat'])

        tab1, tab2 = st.tabs([
            "🔍 Prediction" if language == "English" else "🔍 भविष्यवाणी",
            "📌 Explanation" if language == "English" else "📌 व्याख्या"
        ])
        # 📍 Show Nearby Eye Hospitals Using Location
        st.markdown("---")
        st.subheader("🏥 Nearby Eye Hospitals" if language == "English" else "🏥 पास के नेत्र अस्पताल")

        lat, lon = get_location()
        if lat and lon:
            maps_url = f"https://www.google.com/maps?q=eye+hospital+near+{lat},{lon}&output=embed"
            st.components.v1.html(f"""
            <iframe width="100%" height="400"
            src="{maps_url}">
             </iframe>
            """, height=400)
        else:
             st.warning("📍 Unable to detect location. Please check your connection or VPN." if language == "English"
               else "📍 स्थान का पता नहीं चल सका। कृपया अपना कनेक्शन या वीपीएन जांचें।")

        with tab1:
            st.subheader("🔍 Prediction Result" if language == "English" else "🔍 भविष्यवाणी परिणाम")
            st.write(f"🩺 **Detected Disease:** {selected}" if language == "English" else f"🩺 **पहचानी गई बीमारी:** {selected}")
            st.write(f"📊 **Confidence:** {confidence:.2%}" if language == "English" else f"📊 **विश्वास स्तर:** {confidence:.2%}")
        with tab2:
            st.subheader("🧠 Disease Explanation" if language == "English" else "🧠 बीमारी की व्याख्या")
            if language == "English":
                st.write(f"📌 {info['desc']}")
                st.write(f"💊 {info['treat']}")
            else:
                st.write(f"📌 {translated_desc}")
                st.write(f"💊 {translated_treat}")

        if st.button("📄 Generate PDF Report" if language == "English" else "📄 पीडीएफ रिपोर्ट बनाएं"):
            st.session_state["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            desc = info['desc'] if language == "English" else translated_desc
            treat = info['treat'] if language == "English" else translated_treat
            pdf_path = generate_pdf(
               patient_name, 
               patient_age, 
               image, 
               [{"disease": selected, "confidence": confidence}],  # 👈 wrap as list of dicts
               lang=language,
               gradcam_image=gradcam_image
            )
            with open(pdf_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                href = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="Eye_Report.pdf">📥 Download Report</a>'
                st.markdown(href, unsafe_allow_html=True)
            os.remove(pdf_path)

# ----------------------------
# 📦 BATCH MODE
# ----------------------------
elif (input_mode == "Multiple Images (Batch)" and language == "English") or (input_mode == "एकाधिक इमेज (बैच)" and language == "Hindi"):
    uploaded_files = st.file_uploader(
        "📥 Upload Retina Images (Multiple)" if language == "English" else "📥 रेटिना इमेज अपलोड करें (एकाधिक)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="batch_upload"
    )
    
    if uploaded_files:
        st.markdown("### 🔍 Batch Prediction Results" if language == "English" else "### 🔍 बैच भविष्यवाणी परिणाम")
        records = []

        for i, file in enumerate(uploaded_files):
            filename = getattr(file, "name", f"Image_{i+1}")

            try:
                image = Image.open(file).convert("RGB")
            except Exception as e:
                st.warning(f"❌ Could not open image {filename}. Error: {e}")
                continue

            # 🔮 Prediction
            selected, confidence, info = predict_image(image)

            # 🔥 Grad-CAM Heatmap
            img_resized = image.resize((224, 224))
            img_array = np.asarray(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            heatmap = make_gradcam_heatmap(
                img_array,
                feature_model=feature_model,
                classifier_head=classifier_head,
                last_conv_layer_name="Conv_1"
            )
            gradcam_image = overlay_heatmap_on_image(image, heatmap)

            # 🗂️ Save to CSV
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            record = {
                "Timestamp": timestamp,
                "Name": patient_name,
                "Age": patient_age,
                "Phone": st.session_state.get("phone", ""),
                "Email": st.session_state.get("email", ""),
                "Disease": selected,
                "Confidence": round(confidence * 100, 2)
            }

            df_existing = pd.read_csv(RECORDS_FILE)
            df_existing = pd.concat([df_existing, pd.DataFrame([record])], ignore_index=True)
            df_existing.to_csv(RECORDS_FILE, index=False)

            # 🌐 Translation
            translated_desc, translated_treat = info['desc'], info['treat']
            if language == "Hindi":
                translated_desc = translate_text(info['desc'])
                translated_treat = translate_text(info['treat'])

            # 🖼️ Show Images (no use_container_width for Streamlit Cloud!)
            try:
                st.image(image, caption=f"🖼️ {filename}")
                st.image(gradcam_image, caption="🔥 Grad-CAM Heatmap")
            except Exception as e:
                st.warning(f"⚠️ Could not display image {filename}. Error: {e}")

            # 🧠 Show Prediction
            st.write(f"🩺 **Prediction:** {selected}" if language == "English" else f"🩺 **भविष्यवाणी:** {selected}")
            st.write(f"📊 **Confidence:** {confidence:.2%}" if language == "English" else f"📊 **विश्वास स्तर:** {confidence:.2%}")
            st.write(f"📌 **Description:** {info['desc']}" if language == "English" else f"📌 **विवरण:** {translated_desc}")
            st.write(f"💊 **Treatment:** {info['treat']}" if language == "English" else f"💊 **उपचार:** {translated_treat}")

            # 📄 PDF Download
            if st.button(f"📄 Generate PDF for {filename}" if language == "English" else f"📄 {filename} के लिए पीडीएफ बनाएं", key=f"pdf_{i}"):
                st.session_state["timestamp"] = timestamp
                desc = info['desc'] if language == "English" else translated_desc
                treat = info['treat'] if language == "English" else translated_treat
                predictions = [{"disease": selected, "confidence": confidence}]
                pdf_path = generate_pdf(patient_name, patient_age, image, predictions, lang=language, gradcam_image=gradcam_image)

                with open(pdf_path, "rb") as f:
                    base64_pdf = base64.b64encode(f.read()).decode('utf-8')
                    href = f'<a href="data:application/octet-stream;base64,{base64_pdf}" download="Report_{filename}.pdf">📥 Download Report</a>'
                    st.markdown(href, unsafe_allow_html=True)
                os.remove(pdf_path)

            # 📋 Table
            records.append({
                "Image" if language == "English" else "इमेज": filename,
                "Prediction" if language == "English" else "भविष्यवाणी": selected,
                "Confidence" if language == "English" else "विश्वास स्तर": f"{confidence:.2%}"
            })

        # ✅ Display Summary Table
        st.markdown("### 📊 Summary Table" if language == "English" else "### 📊 सारांश तालिका")
        st.dataframe(records)


# ----------------------------
# 📂 Patient Prediction History Viewer
# ----------------------------
st.markdown("---")
st.header("📂 Prediction History" if language == "English" else "📂 भविष्यवाणी इतिहास")

df = pd.read_csv(RECORDS_FILE)

if df.empty:
    st.info("No records found.")
else:
    with st.expander("🔍 Filter Records"):
        name_filter = st.text_input("🔎 Search by Name")
        disease_filter = st.selectbox("🦠 Filter by Disease", ["All"] + df["Disease"].unique().tolist())
        date_range = st.date_input("📅 Filter by Date Range", [])

        if name_filter:
            df = df[df["Name"].str.contains(name_filter, case=False)]
        if disease_filter != "All":
            df = df[df["Disease"] == disease_filter]
        if len(date_range) == 2:
            start, end = date_range
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            df = df[(df['Timestamp'].dt.date >= start) & (df['Timestamp'].dt.date <= end)]

    st.dataframe(df)

    # Optional CSV export
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download CSV", csv, "patient_records.csv", "text/csv")

# ----------------------------
# 💬 Voice Assistant
# ----------------------------

faq_answers = {
    "glaucoma": "Glaucoma is a group of eye conditions that damage the optic nerve. It often has no early symptoms. Regular eye exams and medications can manage it.",
    "diabetic": "Diabetic retinopathy is caused by long-term diabetes. It damages blood vessels in the retina and can be managed with laser treatment and injections.",
    "cataract": "A cataract is a cloudy area in the eye's lens. It's common with aging and may require surgery for clear vision.",
    "amd": "Age-related Macular Degeneration affects the macula. It leads to central vision loss. Treatment includes anti-VEGF injections.",
    "retinitis pigmentosa": "Retinitis Pigmentosa is a genetic condition causing vision loss over time. There is no cure, but low vision aids can help.",
    "retinal detachment": "Retinal Detachment occurs when the retina lifts from the eye wall. It requires emergency surgery to restore vision.",
    "papilledema": "Papilledema is swelling of the optic disc due to increased brain pressure. It requires immediate medical attention.",
    "pseudopapilledema": "Pseudopapilledema mimics papilledema but is usually benign. Monitoring is recommended.",
    "hospital": "You can check the Nearby Eye Hospitals section above based on your location.",
    "eye": "Please specify the eye disease for more information. For example: glaucoma, cataract, AMD, etc.",
    "default": "Sorry, I couldn't understand your question. Try asking about a specific eye disease like 'What is glaucoma?'"
}
st.markdown("---")
st.header("💬 Ask About Eye Diseases (Voice Assistant)" if language == "English" else "💬 आंख की बीमारियों के बारे में पूछें (वॉयस असिस्टेंट)")

audio = mic_recorder(
    start_prompt="🎙️ Click to Record" if language == "English" else "🎙️ रिकॉर्ड करने के लिए क्लिक करें",
    stop_prompt="⏹️ Stop Recording" if language == "English" else "⏹️ रिकॉर्डिंग रोकें",
    key="voice"
)

if audio:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_audio:
        tmp_audio.write(audio["bytes"])
        tmp_audio_path = tmp_audio.name

    st.audio(audio["bytes"], format="audio/wav")

    with st.spinner("🧠 Transcribing..." if language == "English" else "🧠 ट्रांसक्राइब किया जा रहा है..."):
        model_whisper = whisper.load_model("base")
        result = model_whisper.transcribe(tmp_audio_path)
        question = result["text"]

    st.success(f"🔊 You said: {question}" if language == "English" else f"🔊 आपने कहा: {question}")

    # ==== LOCAL FAQ LOGIC ====
    answer = faq_answers["default"]
    for keyword in faq_answers:
        if keyword != "default" and keyword in question.lower():
            answer = faq_answers[keyword]
            break

    st.markdown("### 🤖 Assistant's Answer:" if language == "English" else "### 🤖 असिस्टेंट का उत्तर:")
    st.write(answer)

    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(answer)
    engine.runAndWait()
    os.remove(tmp_audio_path)
else:
    st.info("Click the mic and ask your eye disease question." if language == "English" else "माइक पर क्लिक करें और अपनी आंख की बीमारी का प्रश्न पूछें।")
