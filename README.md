# 🧠 EyeScope AI - Retina Disease Detector

[![Streamlit App](https://img.shields.io/badge/Live%20App-Open-green?style=for-the-badge&logo=streamlit)](https://eye-disease-detector1-369.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)](https://www.python.org/downloads/release/python-3100/)

An intelligent deep learning web app that can detect **Retina Diseases** from retinal fundus images using a pre-trained CNN model and Grad-CAM heatmaps for visual explainability.

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

## 🚀 Live Demo

🔗 **Try it now:** [eye-disease-detector1-369.streamlit.app](https://eye-disease-detector1-369.streamlit.app/)

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
