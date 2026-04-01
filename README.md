🧠 Tweet Emotion Detector

An end-to-end Natural Language Processing (NLP) application that classifies the emotional subtext of short English text into six categories: **Joy, Sadness, Anger, Fear, Love, and Surprise.**

## 🚀 Project Overview
This project demonstrates a complete Machine Learning lifecycle—from raw data ingestion using Hugging Face, through feature engineering and model training with Scikit-Learn, to a functional web dashboard built with Streamlit.

### Key Features:
* **High Efficiency:** Uses a lightweight LinearSVC model that trains in seconds and runs on standard CPUs.
* **Context Awareness:** Utilizes TF-IDF with N-grams (bi-grams) to understand word pairs.
* **Confidence Visualization:** Provides a dynamic probability breakdown for every prediction.
* **Balanced Classification:** Specifically tuned to handle imbalanced datasets (e.g., accurately detecting rare emotions like 'Surprise').

---

## 🛠️ Tech Stack
* **Language:** Python 3.9+
* **ML Library:** Scikit-Learn
* **Data Handling:** Pandas, Datasets (Hugging Face)
* **Frontend:** Streamlit
* **Model Persistence:** Joblib

---

## 📂 Project Structure
* `train_model.py`: Script to download the dataset, train the model, and save the pipeline.
* `app.py`: The Streamlit web application code.
* `emotion_model.pkl`: The serialized (saved) trained model.
* `requirements.txt`: List of necessary Python dependencies.

---

## ⚙️ Methodology

### 1. Data Preprocessing
The model uses the **dair-ai/emotion** dataset (16,000 training samples).
* **Tokenization:** Text is broken into unigrams and bigrams.
* **Stop-Word Removal:** Common English filler words are removed to reduce noise.
* **TF-IDF Vectorization:** Words are converted into mathematical weights based on their uniqueness across the dataset (15,000 features).

### 2. The Model
The core classifier is a **Linear Support Vector Classifier (LinearSVC)**.
* **Regularization ($C=0.5$):** Implemented to prevent overfitting and improve generalization.
* **Class Balancing:** `class_weight='balanced'` was used to ensure the model doesn't ignore minority classes like "Surprise."
* **Calibration:** Wrapped in `CalibratedClassifierCV` to translate raw distances into 0-100% probability scores.

---

## 🏃 How to Run Locally

1. **Install Dependencies:**
   pip install -r requirements.txt
   
Train the Model: python train_model.py

Launch the Web App: streamlit run app.py
