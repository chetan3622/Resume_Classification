import streamlit as st
import joblib
import re

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Resume Classifier",
    page_icon="📄",
    layout="centered"
)

# ---------------- LOAD FILES ----------------
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")
le = joblib.load("label_encoder.pkl")

# ---------------- CLEAN FUNCTION ----------------
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = " ".join(text)
    return text

# ---------------- STYLE ----------------
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 700px;
}

h1 {
    text-align: center;
    font-weight: 700;
}

.upload-box {
    border: 1px solid #e6e6e6;
    padding: 25px;
    border-radius: 12px;
    text-align: center;
    background-color: #fafafa;
}

.result {
    margin-top: 25px;
    padding: 20px;
    border-radius: 12px;
    background-color: #f0f9ff;
    border-left: 5px solid #0ea5e9;
    font-size: 20px;
    font-weight: 600;
}

.footer {
    text-align: center;
    color: #888;
    font-size: 13px;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("📄 Resume Classification System")
st.caption("Upload a resume to predict its category using machine learning")

# ---------------- UPLOAD ----------------
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload Resume (.txt file)", type=["txt"])
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if uploaded_file is not None:
    text = uploaded_file.read().decode()

    cleaned = clean_text(text)
    vector = tfidf.transform([cleaned])

    prediction = model.predict(vector)
    category = le.inverse_transform(prediction)

    # Confidence (if supported)
    try:
        probs = model.predict_proba(vector)
        confidence = max(probs[0]) * 100
        result_text = f"{category[0]}  |  Confidence: {confidence:.2f}%"
    except:
        result_text = category[0]

    st.markdown(f'<div class="result">🎯 Prediction: {result_text}</div>', unsafe_allow_html=True)

    # Preview
    with st.expander("📄 View Resume"):
        st.text(text[:1200])

# ---------------- FOOTER ----------------
st.markdown('<div class="footer">ML Project • Resume Classification</div>', unsafe_allow_html=True)