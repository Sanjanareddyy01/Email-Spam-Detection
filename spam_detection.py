import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")

print(data.head())
print(data.columns)

# Convert labels
data["Category"] = data["Category"].map({
    "ham": 0,
    "spam": 1
})

print("\nAfter Encoding:\n")
print(data.head())
import re

def clean_text(text):
    text = text.lower()                      # Convert to lowercase
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text) # Remove punctuation, keep numbers
    return text

data["Message"] = data["Message"].apply(clean_text)

print("\nAfter Cleaning:\n")
print(data.head())
# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data["Message"])
y = data["Category"]

print("\nShape of X:", X.shape)
print("Shape of y:", y.shape)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Train the Naive Bayes model
model = MultinomialNB()

model.fit(X_train, y_train)

print("Model trained successfully!")

# Predict on the test data
predictions = model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy: {accuracy * 100:.2f}%")
email = ["Congratulations! You have won ₹50,000. Claim now!"]

email_vector = vectorizer.transform(email)

prediction = model.predict(email_vector)

print(prediction)