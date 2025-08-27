# Matchly

[![Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A simple web application to match resumes against a job description.  
Upload a job description and multiple resumes (PDF or DOCX) and the app ranks candidates by similarity using TF-IDF and cosine similarity.  

**Live Demo:** https://YOUR-RENDER-URL.onrender.com  
**Repository:** https://github.com/YOUR-USERNAME/AIResumeMatcher

---

## Features
- Upload job description and multiple resumes (PDF/DOCX)
- Extracts text with pdfplumber and python-docx
- Computes similarity with TF-IDF + cosine similarity
- Displays ranked results in a table
- Option to export results as CSV
- Runs easily on Streamlit Cloud or Render free tier

---

## Quickstart (Run Locally)

1. Clone the repo:
git clone https://github.com/YOUR-USERNAME/AIResumeMatcher.git
cd AIResumeMatcher

2. Create and activate a virtual environment:
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

3. Install requirements:
pip install -r requirements.txt

4. Run the app:
streamlit run app.py

---

## Deployment

### Streamlit Cloud
- Connect this repository in Streamlit Cloud
- It will auto-detect app.py
- Set Python version to 3.11

### Render
- Build command:
pip install -r requirements.txt

- Start command:
streamlit run app.py --server.port $PORT --server.address 0.0.0.0

---

## Roadmap
- Skill extraction and weighted scoring
- Explanations for match results
- OCR support for scanned resumes
- Optionally use embeddings (pgvector / sentence-transformers)

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.