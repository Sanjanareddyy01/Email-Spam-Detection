import pandas as pd
import os
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from utils import clean_text


# -----------------------------------
# Load Dataset
# -----------------------------------

print("\nLoading dataset...")

data = pd.read_csv(
    "spam.csv",
    encoding="latin-1"
)


# Keep only required columns

data = data[['Category', 'Message']]

data.columns = [
    "label",
    "message"
]


# Convert labels

data["label"] = data["label"].map({

    "ham": 0,

    "spam": 1

})


print("Dataset Loaded Successfully")

print(data.head())



# -----------------------------------
# Clean Text
# -----------------------------------

print("\nCleaning emails...")


data["message"] = data["message"].apply(
    clean_text
)



# -----------------------------------
# Split Dataset
# -----------------------------------

X = data["message"]

y = data["label"]


X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42

)



# -----------------------------------
# TF-IDF Vectorization
# -----------------------------------

print("\nApplying TF-IDF...")


vectorizer = TfidfVectorizer(

    max_features=5000

)


X_train_vector = vectorizer.fit_transform(
    X_train
)


X_test_vector = vectorizer.transform(
    X_test
)



# -----------------------------------
# Train Model
# -----------------------------------

print("\nTraining model...")


model = MultinomialNB()


model.fit(

    X_train_vector,

    y_train

)



# -----------------------------------
# Evaluation
# -----------------------------------

print("\nEvaluating model...")


prediction = model.predict(

    X_test_vector

)


accuracy = accuracy_score(

    y_test,

    prediction

)


print("\n-----------------------------")

print(
    f"Accuracy : {accuracy*100:.2f}%"
)

print("-----------------------------")


print("\nClassification Report:")

print(

    classification_report(

        y_test,

        prediction

    )

)



print("\nConfusion Matrix:")

print(

    confusion_matrix(

        y_test,

        prediction

    )

)


# -----------------------------------
# Save Model
# -----------------------------------

print("\nSaving model...")

metrics = {
    "accuracy": round(accuracy * 100, 2),
    "confusion_matrix": confusion_matrix(
        y_test,
        prediction
    ).tolist(),
    "dataset_size": len(data)
}


with open("models/metrics.json", "w") as file:
    json.dump(metrics, file, indent=4)


joblib.dump(
    model,
    "models/spam_model.pkl"
)


joblib.dump(
    vectorizer,
    "models/vectorizer.pkl"
)


print("\n✅ Training Completed")

print("Saved Files:")
print("✔ models/spam_model.pkl")
print("✔ models/vectorizer.pkl")
print("✔ models/metrics.json")