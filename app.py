import io
import pdfplumber
import docx
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Matcher", layout="wide")
st.title("AI Resume Matcher")

st.write(
    "Upload a Job Description (PDF or DOCX) and multiple resumes. "
    "The app will rank candidates by similarity using TF-IDF and cosine similarity."
)

# ----------- File Uploads -----------
jd_file = st.file_uploader("Upload Job Description (PDF or DOCX)", type=["pdf", "docx"])
resume_files = st.file_uploader(
    "Upload Resumes (PDF or DOCX)", type=["pdf", "docx"], accept_multiple_files=True
)

# ----------- Helper Functions -----------
def extract_text(file):
    """Extract text from PDF or DOCX."""
    name = file.name.lower()
    if name.endswith(".pdf"):
        text = []
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                txt = page.extract_text() or ""
                text.append(txt)
        return "\n".join(text)
    elif name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        return ""

def clean(text):
    return (text or "").replace("\x00", " ").strip()

# ----------- Processing -----------
if st.button("Match") and jd_file and resume_files:
    with st.spinner("Processing..."):
        jd_text = clean(extract_text(jd_file))
        docs = [jd_text]
        names = ["Job Description"]

        for f in resume_files:
            names.append(f.name)
            docs.append(clean(extract_text(f)))

        # TF-IDF similarity
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        X = vectorizer.fit_transform(docs)
        jd_vec = X[0:1]
        resume_vecs = X[1:]
        sims = cosine_similarity(jd_vec, resume_vecs).flatten()

        results = pd.DataFrame({
            "Candidate": names[1:],
            "Match Score (%)": (sims * 100).round(1)
        }).sort_values("Match Score (%)", ascending=False)

    st.success("Done!")
    st.dataframe(results, use_container_width=True)

    st.download_button(
        "Download Results as CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="resume_matches.csv",
        mime="text/csv"
    )

st.caption("Note: Works best with text-based PDFs or DOCX files. For scanned/image resumes, OCR support may be added later.")