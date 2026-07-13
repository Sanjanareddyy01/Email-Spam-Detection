import streamlit as st
# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="AI Email Spam Detector",
    page_icon="📧",
    layout="wide"
)
import plotly.express as px
import json

from utils import (
    predict_email,
    get_email_statistics
)


def load_css():

    with open("assets/style.css") as f:

        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )


load_css()
import streamlit as st
import json

from utils import (
    predict_email,
    get_email_statistics
)





# -----------------------------------
# Load Metrics
# -----------------------------------

with open("models/metrics.json", "r") as file:
    metrics = json.load(file)



# -----------------------------------
# Header
# -----------------------------------

st.markdown('<div class="hero">', unsafe_allow_html=True)

col1, col2 = st.columns([1,5])

with col1:
    st.image("assets/logo.png", width=110)

with col2:
    st.markdown("""
    <h1>AI Email Spam Detector</h1>
    <p>
    Detect spam emails instantly using
    Machine Learning and Natural Language Processing
    </p>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="hero-badges">
<span>96.41% Accuracy</span>
<span>5572 Emails</span>
<span>TF-IDF NLP</span>
<span>Naive Bayes</span>
</div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------
# Pipeline
# -----------------------------------

st.markdown("""
<div class='pipeline'>

📧 Email
→
🧹 Cleaning
→
🔤 TF-IDF
→
🤖 Naive Bayes
→
🚨 Classification

</div>
""", unsafe_allow_html=True)

# -----------------------------------
# Dashboard Cards
# -----------------------------------

fig = px.pie(

    values=[4825, 747],

    names=["Ham", "Spam"],

    hole=0.65

)

st.plotly_chart(
    fig,
    use_container_width=True
)
col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Accuracy",
        f"{metrics['accuracy']}%"
    )


with col2:

    st.metric(
        "Dataset",
        f"{metrics['dataset_size']} Emails"
    )


with col3:

    st.metric(
        "Algorithm",
        "Naive Bayes"
    )


with col4:

    st.metric(
        "Features",
        "TF-IDF NLP"
    )



st.divider()



# -----------------------------------
# Email Input
# -----------------------------------

st.subheader("📨 Paste your Email")


email_text = st.text_area(

    "Enter email content",

    height=200,

    placeholder=
    "Paste your email message here..."

)



# -----------------------------------
# Statistics
# -----------------------------------

if email_text:

    stats = get_email_statistics(
        email_text
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Words",
        stats["Words"]
    )

    c2.metric(
        "Characters",
        stats["Characters"]
    )

    c3.metric(
        "Reading Time",
        f"{stats['Reading Time']} sec"
    )

st.divider()



# -----------------------------------
# Prediction
# -----------------------------------

result = None
if st.button(
    "🔍 Analyze Email",
    use_container_width=True
):

    if email_text.strip() == "":

        st.warning(
            "Please enter an email first."
        )


    else:

        with st.spinner(
            "Analyzing email..."
        ):

            result = predict_email(
                email_text
            )



        st.divider()



        if result["result"] == "Spam":

            st.markdown(f"""
<div class="result-spam">

<h2>🚨 SPAM DETECTED</h2>

<h3>{result['confidence']}% Confidence</h3>

</div>
""", unsafe_allow_html=True)

        else:

            st.markdown(f"""
<div class="result-safe">

<h2>✅ NOT SPAM</h2>

<h3>{result['confidence']}% Confidence</h3>

</div>
""", unsafe_allow_html=True)



        st.subheader(
            "Probability"
        )


        st.progress(
            result["confidence"] / 100
        )


        st.write(
            f"{result['confidence']}%"
        )



    st.subheader(
    "Detected Spam Keywords"
)

if result is not None and result["keywords"]:

    html = ""

    for word in result["keywords"]:

        html += f"""
        <span class="keyword">
        {word}
        </span>
        """

    st.markdown(
        html,
        unsafe_allow_html=True
    )

else:

    st.write(
        "No suspicious keywords detected."
    )

# -----------------------------------
# Footer
# -----------------------------------

st.markdown("""
<hr>

<center>

Built with Python • NLP • TF-IDF • Naive Bayes • Streamlit

</center>
""", unsafe_allow_html=True)