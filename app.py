import streamlit as st
import pandas as pd
import re
import os
import PyPDF2
import docx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords

# --- INITIALIZATION & NLTK ---
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

STOPWORDS = set(stopwords.words('english'))

# --- 1. DATASET ---
def get_job_dataset():
    data = {
        'title': [
            'Frontend Developer', 'Data Scientist', 'Python Backend Engineer', 
            'UI/UX Designer', 'DevOps Engineer', 'Project Manager', 
            'Marketing Specialist', 'HR Manager', 'Fullstack Developer', 
            'Cybersecurity Analyst', 'Cloud Architect', 'Mobile App Developer',
            'Business Analyst', 'Machine Learning Engineer', 'Quality Assurance'
        ],
        'description': [
            'React, Vue, CSS expertise and responsive design.',
            'Python, SQL, machine learning models, Random Forest, XGBoost.',
            'Django, Flask, FastAPI and PostgreSQL database management.',
            'Figma, Adobe XD, wireframes and user research.',
            'AWS, Docker, Kubernetes, and CI/CD automation.',
            'Agile methodology, leading teams, and managing backlogs.',
            'SEO, SEM, content strategy, and Google Analytics.',
            'Talent acquisition, employee relations, and HR strategy.',
            'MERN stack (MongoDB, Express, React, Node) development.',
            'Network security, penetration testing, incident response.',
            'Azure/GCP architecture, scalability, and cost-optimization.',
            'iOS and Android using Flutter, React Native, or Swift.',
            'Business analysis, data documentation, technical solutions.',
            'PyTorch/TensorFlow, NLP, and Computer Vision.',
            'Automated testing with Selenium, Cypress, and unit tests.'
        ]
    }
    return pd.DataFrame(data)

# --- 2. LOGIC ---
def extract_text(file):
    text = ""
    try:
        if file.name.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        elif file.name.endswith('.docx'):
            doc = docx.Document(file)
            text = " ".join([para.text for para in doc.paragraphs])
    except Exception as e:
        st.error(f"Error reading file: {e}")
    return text

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())
    tokens = text.split()
    return " ".join([w for w in tokens if w not in STOPWORDS])

def recommend_jobs(user_input, df):
    processed_jobs = df['description'].apply(preprocess_text)
    processed_user = preprocess_text(user_input)
    
    vectorizer = TfidfVectorizer()
    all_texts = pd.concat([pd.Series([processed_user]), processed_jobs], ignore_index=True)
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    top_indices = similarities.argsort()[-5:][::-1]
    
    results = []
    for idx in top_indices:
        results.append({
            'title': df.iloc[idx]['title'],
            'description': df.iloc[idx]['description'],
            'score': round(similarities[idx] * 100, 1)
        })
    return results

# --- 3. UI/UX DESIGN (High Contrast Edition) ---
st.set_page_config(page_title="AI Career Matcher", layout="centered")

st.markdown("""
<style>
    /* Gradient Background */
    .stApp { 
        background: linear-gradient(135deg, #e0eafc 0%, #cfdef3 100%); 
    }

    /* Header Styling */
    h1 {
        color: #1a2a6c !important;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        text-align: center;
    }

    /* Custom Job Card */
    .job-card {
        background-color: #ffffff;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.08);
        margin-bottom: 20px;
        border-left: 8px solid #1a2a6c;
        transition: transform 0.3s ease;
    }

    .job-card:hover {
        transform: scale(1.02);
    }

    /* Text Colors for Visibility */
    .job-title { 
        color: #1a2a6c !important; 
        font-size: 1.4rem; 
        font-weight: 700; 
        margin-bottom: 10px;
    }

    .job-desc { 
        color: #34495e !important; 
        font-size: 1rem; 
        line-height: 1.5;
        margin-bottom: 15px;
    }

    .match-score-badge { 
        display: inline-block;
        background-color: #1a2a6c;
        color: #ffffff !important; 
        padding: 5px 15px;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
    }

    /* Tab Label Styling */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        color: #1a2a6c;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.markdown("<h1>💼 AI Job Recommender</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #5d6d7e; font-size: 1.1rem; margin-bottom: 40px;'>Intelligent skill matching for the modern workforce.</p>", unsafe_allow_html=True)

    df = get_job_dataset()
    user_text = ""

    tab1, tab2 = st.tabs(["📄 Upload Resume", "⌨️ Enter Skills"])
    
    with tab1:
        file = st.file_uploader("Upload PDF or DOCX", type=["pdf", "docx"])
        if file:
            user_text = extract_text(file)
            if user_text: st.success("Resume data captured!")

    with tab2:
        skills = st.text_area("List your skills (e.g., Python, Figma, React)...", height=150)
        if skills: user_text = skills

    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔍 Find My Best Matches", use_container_width=True):
        if not user_text.strip():
            st.error("Please provide a resume or skill list to proceed.")
        else:
            with st.spinner("Analyzing your profile..."):
                results = recommend_jobs(user_text, df)
                st.markdown("### 🎯 Recommended Opportunities")
                for job in results:
                    st.markdown(f"""
                    <div class="job-card">
                        <div class="job-title">{job['title']}</div>
                        <div class="job-desc">{job['description']}</div>
                        <div class="match-score-badge">Match Score: {job['score']}%</div>
                    </div>
                    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()