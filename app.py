import streamlit as st
import joblib
import pandas as pd

# --- 1. PAGE CONFIGURATION ---
# This must be the first Streamlit command. It sets the browser tab title and layout.
st.set_page_config(
    page_title="Emotion Detector",
    page_icon="🧠",
    layout="wide" # Uses more of the screen width, great for presentations
)

# --- 2. SETUP & CACHING ---
@st.cache_resource
def load_model():
    return joblib.load('emotion_model.pkl')

model = load_model()

emotion_mapping = {0: "Sadness", 1: "Joy", 2: "Love", 3: "Anger", 4: "Fear", 5: "Surprise"}
emoji_mapping = {"Sadness": "😢", "Joy": "😄", "Love": "😍", "Anger": "😡", "Fear": "😨", "Surprise": "😲"}

# --- 3. SIDEBAR (Great for your professor!) ---
with st.sidebar:
    st.header("About the Project")
    st.write("This application detects the underlying emotion in short English text using Natural Language Processing.")
    
    st.subheader("⚙️ Under the Hood")
    st.write("This pipeline uses advanced feature engineering and a robust linear classifier.")
    st.write("- **Feature Engineering:** TF-IDF (15,000 features)")
    st.write("- **Context Awareness:** 1-gram & 2-gram extraction")
    st.write("- **Noise Reduction:** English stop-word removal")
    st.write("- **Model:** Calibrated Linear Support Vector Classifier (LinearSVC)")
    st.write("- **Class Balancing:** Adjusted weights to accurately detect rare emotions like *Surprise* and *Love*.")
    st.write("- **Accuracy:** ~88-90% on test data")
    
    st.subheader("📊 The Dataset")
    st.write("Trained on the `dair-ai/emotion` dataset from Hugging Face, containing 16,000 labeled informal text messages.")
    st.divider()
    st.caption("Developed for Academic Demo")

# --- 4. MAIN APP UI ---
st.title("🧠 Tweet Emotion Detector")
st.markdown("Enter a sentence below, and the machine learning pipeline will predict the primary emotion and display its confidence levels.")

# Added a clean info box to replace the dropdown
st.info("💡 Try typing a complex sentence, or test how the model handles conflicting emotions!")

user_input = st.text_area("Analyze Text:", height=100, placeholder="Type your sentence or tweet here...")

# --- 5. PREDICTION LOGIC & RESULTS ---
# type="primary" makes the button pop with a solid background color
if st.button("Analyze Emotion", type="primary"): 
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # The model logic stays exactly the same! 
        # The new pipeline handles the N-grams and SVC silently in the background.
        prediction = model.predict([user_input])[0]
        predicted_emotion = emotion_mapping[prediction]
        probabilities = model.predict_proba([user_input])[0]
        
        st.divider() # Adds a nice visual line break
        st.subheader("Analysis Results")
        
        # Use columns to lay out the results side-by-side
        col1, col2 = st.columns([1, 2]) # col2 (the chart) will be twice as wide as col1 (the result)
        
        with col1:
            st.write("### Primary Emotion")
            # Using st.metric makes the main result look like a professional dashboard widget
            st.metric(label="Detected", value=f"{predicted_emotion} {emoji_mapping[predicted_emotion]}")
            st.write(f"**Top Confidence:** {max(probabilities)*100:.1f}%")
            
        with col2:
            st.write("### Probability Breakdown")
            prob_df = pd.DataFrame({
                "Emotion": list(emotion_mapping.values()),
                "Probability": probabilities
            })
            
            # UI Polish: Sort the dataframe so the highest probability is at the top of the chart!
            prob_df = prob_df.sort_values(by="Probability", ascending=False).set_index("Emotion")
            
            st.bar_chart(prob_df)

# --- 6. FOOTER ---
st.markdown("---")
st.caption("Built with Python, Scikit-Learn, and Streamlit.")