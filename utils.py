import re
import pandas as pd

# -----------------------------------
# Clean Email Text
# -----------------------------------

def clean_text(text):
    """
    Clean email text for NLP.
    """

    text = str(text)

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"www\S+", "", text)

    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# -----------------------------------
# Email Statistics
# -----------------------------------

def get_email_statistics(text):

    words = text.split()

    return {

        "Characters": len(text),

        "Words": len(words),

        "Unique Words": len(set(words)),

        "Reading Time": max(1, round(len(words) / 200 * 60))

    }


# -----------------------------------
# Spam Keywords
# -----------------------------------

SPAM_KEYWORDS = [

    "free",

    "winner",

    "won",

    "prize",

    "cash",

    "urgent",

    "offer",

    "claim",

    "click",

    "bonus",

    "lottery",

    "congratulations",

    "gift",

    "reward",

    "money"

]


def detect_spam_keywords(text):

    text = text.lower()

    found = []

    for word in SPAM_KEYWORDS:

        if word in text:

            found.append(word)

    return found


# -----------------------------------
# Prediction Label
# -----------------------------------

def prediction_label(value):

    if value == 1:

        return "Spam"

    return "Not Spam"
# -----------------------------------
# Model Loading
# -----------------------------------

import joblib


def load_model():

    model = joblib.load(
        "models/spam_model.pkl"
    )

    vectorizer = joblib.load(
        "models/vectorizer.pkl"
    )

    return model, vectorizer



# -----------------------------------
# Email Prediction
# -----------------------------------

def predict_email(text):

    model, vectorizer = load_model()


    cleaned_text = clean_text(text)


    vector = vectorizer.transform(
        [cleaned_text]
    )


    prediction = model.predict(
        vector
    )[0]


    probability = model.predict_proba(
        vector
    )[0]


    confidence = float(
    round(
        max(probability) * 100,
        2
    )
)


    return {

        "result": prediction_label(prediction),

        "confidence": confidence,

        "keywords": detect_spam_keywords(text)

    }