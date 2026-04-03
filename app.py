import streamlit as st
import joblib
import re
import PyPDF2
from docx import Document

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Resume Classifier",
    page_icon="🤖",
    layout="wide"
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


# ---------------- CSS ----------------
st.markdown("""
<style>
.stApp {
background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1c1c1c);
background-size: 400% 400%;
animation: gradient 12s ease infinite;
color:white;
}

@keyframes gradient {
0% {background-position:0% 50%;}
50% {background-position:100% 50%;}
100% {background-position:0% 50%;}
}

.result {
background: linear-gradient(90deg,#00c6ff,#0072ff);
padding: 25px;
border-radius: 12px;
text-align:center;
font-size:26px;
font-weight:600;
color:white;
}

.title {
text-align:center;
font-size:48px;
font-weight:700;
}

.subtitle {
text-align:center;
color:#cfcfcf;
margin-bottom:30px;
}

.footer {
text-align:center;
color:#aaa;
margin-top:40px;
}
</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------
st.markdown('<div class="title">🤖 Resume Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload resume and detect job category</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

# ---------------- FILE UPLOAD ----------------
with col1:
    st.subheader("📤 Upload Resume")
    uploaded_file = st.file_uploader(
        "Upload Resume",
        type=["txt", "pdf", "docx"]
    )


# ---------------- PREDICTION ----------------
with col2:
    st.subheader("🎯 Prediction")

    if uploaded_file is not None:

        # TXT
        if uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode()

        # PDF
        elif uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

        # DOCX
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(uploaded_file)
            text = ""
            for para in doc.paragraphs:
                text += para.text

        cleaned = clean_text(text)
        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)
        category = le.inverse_transform(prediction)

        try:
            probs = model.predict_proba(vector)
            confidence = max(probs[0]) * 100
            result = f"{category[0]} <br><span style='font-size:16px'>Confidence: {confidence:.2f}%</span>"
        except:
            result = category[0]

        st.markdown(f'<div class="result">{result}</div>', unsafe_allow_html=True)

    else:
        st.info("Upload resume to see prediction")


# ---------------- PREVIEW ----------------
if uploaded_file is not None:
    st.markdown("### 📄 Resume Preview")
    st.text_area("", text, height=200)


# ---------------- FOOTER ----------------
st.markdown('<div class="footer">AI Resume Screening System</div>', unsafe_allow_html=True)
