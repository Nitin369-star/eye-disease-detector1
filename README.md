# 🧠 EyeScope AI - Retina Disease Detector

[![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)](https://www.python.org/downloads/release/python-3100/)

An intelligent deep learning web app that can detect **Retina Diseases** from retinal fundus images using a pre-trained CNN model and Grad-CAM heatmaps for visual explainability.

> ⚠️ Note: Live deployment on Streamlit Cloud is currently under maintenance due to compatibility issues. Please refer to the screenshots below for a demo.

---

## 📸 Screenshots

| Upload Image Mode | Prediction with Heatmap |
|-------------------|--------------------------|
| ![Upload Mode](<img width="1726" height="851" alt="image" src="https://github.com/user-attachments/assets/164315bf-cbd1-4c9d-a594-f1b2886468dd" />
) | ![Prediction](<img width="1327" height="875" alt="image" src="https://github.com/user-attachments/assets/808446d9-be13-4f61-8729-25f47c0b8467" />
) | ![Uploading image.png…](<img width="530" height="870" alt="image" src="https://github.com/user-attachments/assets/cee1d8a9-2f30-462f-bfc8-0049f7f7dc1b" /> | ![live location](<img width="1194" height="805" alt="image" src="https://github.com/user-attachments/assets/a8a6327b-2471-4bbd-ab23-ee235f7f0a10" />)
)


---

## 📌 Supported Eye Diseases

- 👁️ Normal
- 👁️‍🗨️ Glaucoma
- 🩸 Diabetic Retinopathy
- 🌫️ Cataract
- 🧓 AMD (Age-related Macular Degeneration)
- 🚨 Papilledema
- ❗ Pseudopapilledema

---

## 📷 Features

### ✅ Core Functionality
- Retina disease prediction using a Keras `.h5` model.
- Confidence score and disease explanation.
- Grad-CAM heatmap visualization to show model focus.
- English & Hindi bilingual interface.

### 🖼️ Input Options
- Upload retina images (single or batch)
- Optional webcam capture (local only)

### 📂 Patient Management
- Name, Age, Phone, Email collection
- Timestamped prediction history with filtering options
- CSV export of records

### 📄 PDF Report
- Auto-generated report with:
  - Retina image
  - Grad-CAM heatmap
  - Disease info + treatment
  - QR code and download option

### 📍 Nearby Hospital Finder
- Detects your location
- Shows nearby eye hospitals via embedded Google Maps

---

## 🛠️ Tech Stack

| Component        | Tool/Library                      |
|------------------|-----------------------------------|
| Frontend         | [Streamlit](https://streamlit.io) |
| Deep Learning    | TensorFlow / Keras                |
| Image Processing | OpenCV, PIL                       |
| Grad-CAM         | Custom `make_gradcam_heatmap()`   |
| PDF Export       | ReportLab                         |
| Translation      | Deep Translator (Google Translate)|
| Location         | IP-based geolocation + Maps       |

---

## 🧪 Model Details

- Input Shape: `224x224x3`
- Model Type: CNN trained via [Teachable Machine](https://teachablemachine.withgoogle.com/)
- Format: `keras_model.h5`
- Classes: 7 Retina Disease Types

---

## 🧰 Installation & Run Locally

### 📦 Prerequisites

```bash
git clone https://github.com/Nitin369-star/eye-disease-detector1.git
cd eye-disease-detector1
pip install -r requirements.txt
streamlit run app.py

