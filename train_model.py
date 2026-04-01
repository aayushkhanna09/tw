import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("Starting Phase 2: Loading Data...")

# 1. Load the dataset from Hugging Face
# We specify "dair-ai/emotion" as the dataset name.
dataset = load_dataset("dair-ai/emotion", trust_remote_code=True)

# 2. Convert the Hugging Face dataset format into pandas DataFrames
df_train = pd.DataFrame(dataset['train'])
df_test = pd.DataFrame(dataset['test'])

# 3. Create a mapping dictionary for the labels
emotion_mapping = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# Add a new column to our DataFrame that shows the readable emotion name
df_train['emotion_name'] = df_train['label'].map(emotion_mapping)

# 4. Print some statistics and data to verify it worked
print("\n--- Training Data Overview ---")
print(f"Number of training tweets: {len(df_train)}")
print("\nEmotion Distribution in Training Data:")
print(df_train['emotion_name'].value_counts())

print("\n--- Sneak Peek at the First 5 Tweets ---")
# pd.set_option is just to make sure the text doesn't get cut off in the terminal
pd.set_option('display.max_colwidth', None) 
print(df_train[['text', 'emotion_name']].head())

print("\nStarting Phase 3: Training the Model...")

# 1. Separate the features (text) and the target labels (numbers 0-5)
X_train = df_train['text']
y_train = df_train['label'] # We must train on the numbers, not the string names!

X_test = df_test['text']
y_test = df_test['label']

# 2. Create the UPGRADED Machine Learning Pipeline
print("Building upgraded pipeline with N-grams and a Calibrated SVM...")

model_pipeline = make_pipeline(
    TfidfVectorizer(
        max_features=15000,      # Increased from 10k to hold more word pairs
        ngram_range=(1, 2),      # Looks at single words AND two-word phrases (bi-grams)
        stop_words='english'     # Removes filler words like 'the', 'is', 'at'
    ),
    # We use LinearSVC for better text classification boundaries, 
    # and wrap it in CalibratedClassifierCV so predict_proba() still works for your UI chart!
    CalibratedClassifierCV(LinearSVC(C=0.5, max_iter=2000, class_weight='balanced')) 
)

# 3. Train the model! 
print("Learning patterns from 16,000 tweets. This will take a few seconds...")
model_pipeline.fit(X_train, y_train)
print("Training Complete!")

# 4. Evaluate the model on the unseen test data
print("\nEvaluating Model on Test Data...")
predictions = model_pipeline.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Overall Accuracy: {accuracy * 100:.2f}%\n")

# Print a detailed report showing performance for each specific emotion
# We pass our target_names so the report shows the readable emotions instead of 0-5
print("Detailed Classification Report:")
target_names = [emotion_mapping[i] for i in range(6)]
print(classification_report(y_test, predictions, target_names=target_names))

print("\nStarting Phase 4: Saving the Model...")

# Save the trained pipeline to a file named 'emotion_model.pkl'
joblib.dump(model_pipeline, 'emotion_model.pkl')

print("Model successfully saved as 'emotion_model.pkl'!")
print("Backend training is complete. You can now run your Streamlit app.")