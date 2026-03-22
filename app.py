import streamlit as st
import joblib
import re

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Resume Classifier",
    page_icon="📄",
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


# ---------------- CUSTOM STYLE ----------------
st.markdown("""
<style>

.main-title{
font-size:42px;
font-weight:700;
text-align:center;
margin-bottom:5px;
}

.sub-title{
text-align:center;
color:gray;
margin-bottom:30px;
}

.upload-card{
padding:30px;
border-radius:15px;
border:1px solid #e6e6e6;
background:white;
box-shadow:0px 2px 8px rgba(0,0,0,0.05);
}

.result-card{
padding:30px;
border-radius:15px;
background:linear-gradient(135deg,#4facfe,#00f2fe);
color:white;
text-align:center;
font-size:24px;
font-weight:600;
box-shadow:0px 4px 15px rgba(0,0,0,0.15);
}

.footer{
text-align:center;
color:gray;
margin-top:40px;
}

</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------
st.markdown('<div class="main-title">AI Resume Classification System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Upload resume and identify job category using Machine Learning</div>', unsafe_allow_html=True)

# ---------------- LAYOUT ----------------
col1, col2 = st.columns(2)

# ---------------- UPLOAD SECTION ----------------
with col1:
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.subheader("📤 Upload Resume")
    uploaded_file = st.file_uploader("Upload resume (.txt)", type=["txt"])
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- RESULT SECTION ----------------
with col2:
    st.subheader("🎯 Prediction Result")

    if uploaded_file is not None:
        text = uploaded_file.read().decode()

        cleaned = clean_text(text)
        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)
        category = le.inverse_transform(prediction)

        try:
            probs = model.predict_proba(vector)
            confidence = max(probs[0]) * 100
            result = f"{category[0]} <br> <span style='font-size:16px'>Confidence : {confidence:.2f}%</span>"
        except:
            result = category[0]

        st.markdown(f'<div class="result-card">{result}</div>', unsafe_allow_html=True)

# ---------------- PREVIEW ----------------
if uploaded_file is not None:
    st.markdown("### 📄 Resume Preview")
    st.text(text[:1500])


# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown('<div class="footer">Machine Learning Project | Resume Classification using NLP</div>', unsafe_allow_html=True)
