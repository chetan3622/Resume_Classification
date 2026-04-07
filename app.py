import streamlit as st
import joblib
import re
import pdfplumber
from docx import Document
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="RESUME CLASSIFIER",
    page_icon="",
    layout="wide"
)

# ---------------- CHECK MODEL FILES ----------------
required_files = ["model.pkl", "tfidf.pkl", "label_encoder.pkl"]

for file in required_files:
    if not os.path.exists(file):
        st.error(f" Missing file: {file}")
        st.info("Please run train_model.py first to generate model files.")
        st.stop()

# ---------------- LOAD FILES ----------------
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")
le = joblib.load("label_encoder.pkl")

# ---------------- CLEAN FUNCTION ----------------
def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    text = text.split()
    text = " ".join(text)
    return text

# ---------------- FILE TEXT EXTRACTION ----------------
def extract_text(uploaded_file):
    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type == "txt":
        return uploaded_file.read().decode("utf-8")

    elif file_type == "pdf":
        text = ""
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text

    elif file_type == "docx":
        doc = Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs])

    return ""

# ---------------- SKILL DATABASE ----------------
skill_db = {
    "SQL Developer": ["sql", "mysql", "oracle", "joins", "stored procedures", "etl", "database"],
    "PeopleSoft": ["peoplesoft", "fscm", "hrms", "erp", "workflow", "support"],
    "Workday": ["workday", "hcm", "payroll", "integrations", "reporting", "business process"],
    "Internship": ["python", "java", "html", "css", "project", "internship"],
    "Web Developer": ["html", "css", "javascript", "react", "bootstrap", "frontend"],
    "React Developer": ["react", "reactjs", "redux", "hooks", "jsx", "frontend", "javascript"],
    "Data Science": ["python", "machine learning", "pandas", "numpy", "data analysis", "visualization"],
    "Java Developer": ["java", "spring boot", "jdbc", "hibernate", "rest api", "oop"],
    "Python Developer": ["python", "django", "flask", "api", "automation", "backend"]
}

# ---------------- SKILL MATCH FUNCTION ----------------
def find_skills(text, predicted_role):
    text = text.lower()
    role_skills = skill_db.get(predicted_role, [])
    matched = [skill for skill in role_skills if skill.lower() in text]
    missing = [skill for skill in role_skills if skill.lower() not in text]
    return matched, missing

# ---------------- KEYWORD-BASED ROLE CHECK ----------------
def keyword_based_role(text):
    text = text.lower()
    role_scores = {}
    for role, skills in skill_db.items():
        score = sum(1 for skill in skills if skill.lower() in text)
        role_scores[role] = score

    best_role = max(role_scores, key=role_scores.get)
    return best_role, role_scores[best_role]

# ---------------- RESUME SCORE FUNCTION ----------------
def calculate_score(matched, total):
    if total == 0:
        return 0
    return int((len(matched) / total) * 100)

# ---------------- CLEAN SIMPLE CSS ----------------
st.markdown("""
<style>
/* ===== PAGE BACKGROUND ===== */
html, body, [data-testid="stAppViewContainer"], .main, .block-container {
    background-color: #0f172a !important;
    color: #ffffff !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827 !important;
}

/* Main container */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px !important;
}

/* Title */
.title {
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: #facc15;
    margin-bottom: 8px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: #d1d5db;
    font-size: 18px;
    margin-bottom: 28px;
}

/* Normal boxes */
.card {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 22px;
    box-shadow: none;
}

/* Prediction result box */
.result {
    background-color: #facc15;
    color: #111827;
    border-radius: 14px;
    padding: 22px;
    text-align: center;
    font-size: 26px;
    font-weight: 800;
    box-shadow: none;
}

/* All text */
h1, h2, h3, h4, h5, h6, p, label, span, div {
    color: #ffffff;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background-color: #111827 !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    padding: 12px !important;
}

/* Text area */
textarea {
    background-color: #111827 !important;
    color: #ffffff !important;
    border-radius: 10px !important;
    border: 1px solid #334155 !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background-color: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
    padding: 14px !important;
    box-shadow: none !important;
}

/* Buttons */
.stButton > button {
    background-color: #facc15 !important;
    color: #111827 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #cbd5e1;
    margin-top: 40px;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title"> RESUME CLASSIFIER</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze your resume with Machine Learning, NLP, and smart skill matching</div>', unsafe_allow_html=True)

# ---------------- MAIN TOP SECTION ----------------
col1, col2 = st.columns(2)

# ---------------- LEFT SECTION ----------------
with col1:
    st.subheader("Upload Resume")
    with st.container(border=True):
        uploaded_file = st.file_uploader("Upload your resume", type=["txt", "pdf", "docx"])

# ---------------- RIGHT SECTION ----------------
with col2:
    st.subheader(" Prediction")
    with st.container(border=True):
        if uploaded_file is not None:
            text = extract_text(uploaded_file)

            if text.strip():
                cleaned = clean_text(text)
                vector = tfidf.transform([cleaned])

                # ML Prediction
                prediction = model.predict(vector)
                ml_category = le.inverse_transform(prediction)[0]

                try:
                    probs = model.predict_proba(vector)
                    confidence = max(probs[0]) * 100
                except:
                    confidence = 0

                # Keyword-based correction
                keyword_category, keyword_score = keyword_based_role(text)

                if keyword_score >= 3:
                    category = keyword_category
                else:
                    category = ml_category

                matched_skills, missing_skills = find_skills(text, category)
                score = calculate_score(matched_skills, len(skill_db.get(category, [])))

                result = f"""
                {category} <br>
                <span style='font-size:16px'>Confidence: {confidence:.2f}%</span><br>
                <span style='font-size:16px'>Resume Match Score: {score}%</span>
                """
                st.markdown(f'<div class="result">{result}</div>', unsafe_allow_html=True)
            else:
                st.error("Could not extract text from the uploaded file.")
        else:
            st.info("Upload resume to see prediction")

# ---------------- PREVIEW + ANALYSIS ----------------
if uploaded_file is not None and 'text' in locals():
    st.markdown("###  Resume Preview")
    st.text_area("", text, height=220)

    if text.strip():
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("###  Detected Skills")
            if matched_skills:
                for skill in matched_skills:
                    st.write(f"- {skill}")
            else:
                st.write("No matching skills found.")

        with col4:
            st.markdown("###  Missing Skills")
            if missing_skills:
                for skill in missing_skills:
                    st.write(f"- {skill}")
            else:
                st.write("No major skills missing.")

        st.markdown("###  Suggestion")
        if score >= 80:
            st.success("Your resume is strong for this role.")
        elif score >= 50:
            st.warning("Your resume is moderately aligned. Add more relevant skills.")
        else:
            st.error("Your resume needs improvement for this role. Add more role-specific skills.")

# ---------------- FEATURES ----------------
st.markdown("###  Features")
f1, f2, f3 = st.columns(3)
f1.metric("Model Type", "ML Classifier")
f2.metric("Text Vectorizer", "TF-IDF")
f3.metric("Prediction", "Multi-Category")

# ---------------- FOOTER ----------------
st.markdown('<div class="footer">Resume Classifier | Machine Learning + NLP Project</div>', unsafe_allow_html=True)