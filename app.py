from __future__ import annotations

import base64
import html
import io
import re
from typing import List, Tuple

import pdfplumber
import docx
import pandas as pd
from fastapi import FastAPI, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


APP_NAME = "Matchly — AI Resume Matcher"
MAX_FILES = 50            # maximum number of resumes per request
MAX_FILE_MB = 10          # per-file limit
USE_SKILLS_WEIGHTING = True  # set False to disable keyword weighting


# --------------------------- FastAPI setup ---------------------------

app = FastAPI(title=APP_NAME)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------- HTML (inline) ---------------------------

PAGE_HEAD = f"""
<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{APP_NAME}</title>
<style>
:root {{ --fg:#0b0f14; --muted:#5b6776; --pri:#0f62fe; --bg:#fff; --br:#e5e8ec; }}
* {{ box-sizing: border-box; font-family: system-ui, -apple-system, Segoe UI, Roboto, Inter, Arial; }}
body {{ margin: 32px; background: var(--bg); color: var(--fg); max-width: 1100px; }}
h1 {{ margin: 0 0 12px 0; font-size: 26px; }}
p.muted {{ color: var(--muted); }}
.card {{ border: 1px solid var(--br); border-radius: 12px; padding: 18px; margin: 18px 0; }}
input[type=file], button {{ padding: 10px 12px; border-radius: 8px; border: 1px solid #d7dbe0; }}
button {{ background: var(--pri); border-color: var(--pri); color: #fff; cursor: pointer; }}
button:hover {{ filter: brightness(0.95); }}
table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
th, td {{ border: 1px solid var(--br); padding: 10px; text-align: left; }}
th {{ background: #f8fafc; }}
.alert {{ padding: 12px; border-radius: 8px; background: #fff7ed; border: 1px solid #fed7aa; }}
.success {{ padding: 12px; border-radius: 8px; background: #ecfdf5; border: 1px solid #a7f3d0; }}
small.code {{ font-family: ui-monospace, Menlo, Consolas, monospace; color: #444; }}
footer {{ margin: 28px 0; color: #6b7280; font-size: 14px; }}
label b {{ display: inline-block; margin-bottom: 6px; }}
.grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
@media (max-width: 900px) {{
  .grid {{ grid-template-columns: 1fr; }}
}}
</style></head><body>
<h1>{APP_NAME}</h1>
<p class="muted">Upload a Job Description (PDF/DOCX) and multiple resumes. We rank with TF-IDF cosine similarity{', plus skills weighting' if USE_SKILLS_WEIGHTING else ''}. No paid APIs.</p>
"""

PAGE_FORM = f"""
<div class="card">
  <form method="post" action="/match" enctype="multipart/form-data">
    <div class="grid">
      <div>
        <label><b>Job Description (PDF or DOCX)</b></label>
        <input type="file" name="jd" accept=".pdf,.docx" required>
      </div>
      <div>
        <label><b>Resumes (PDF or DOCX, multiple)</b></label>
        <input type="file" name="resumes" accept=".pdf,.docx" multiple required>
      </div>
    </div>
    <p style="margin-top:12px">
      <button type="submit">Match</button>
    </p>
    <p class="muted">Limits: up to {MAX_FILES} resumes, ≤ {MAX_FILE_MB} MB each. Image-only PDFs aren’t supported in this version.</p>
  </form>
</div>
"""

PAGE_TAIL = """
<footer>© 2025 Matchly. Built with FastAPI.</footer>
</body></html>
"""


# --------------------------- Helpers ---------------------------

def _extract_text_from_bytes(b: bytes, filename: str) -> str:
    """Extract text from PDF/DOCX bytes. Returns empty string if no text found."""
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        try:
            text = []
            with pdfplumber.open(io.BytesIO(b)) as pdf:
                for p in pdf.pages:
                    text.append(p.extract_text() or "")
            return "\n".join(text)
        except Exception:
            return ""
    if name.endswith(".docx"):
        try:
            d = docx.Document(io.BytesIO(b))
            return "\n".join([p.text for p in d.paragraphs])
        except Exception:
            return ""
    return ""


COMMON_SKILL_WORDS = set("""
python java javascript typescript go rust c++ sql nosql mysql postgresql postgres mongodb redis
aws gcp azure docker kubernetes k8s terraform ansible linux windows macos bash powershell
spark hadoop kafka airflow dbt databricks snowflake bigquery redshift clickhouse hive presto trino
pandas numpy scipy scikit-learn sklearn pytorch tensorflow keras nltk spacy transformers
ml ops mlops devops cicd git github gitlab jira confluence tableau powerbi metabase grafana
nlp cv llm llama gpt bert rag embeddings vector pgvector elastic opensearch
microservices rest grpc fastapi flask django streamlit gradio
security iam oauth jwt gdpr iso27001 soc2 pdpl sama
""".split())


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9\+#\-\.]+", (text or "").lower())
    return tokens


def _extract_skills_from_jd(jd_text: str, top_n: int = 30) -> List[str]:
    """Crude skill extraction: intersect frequent tokens with a known skill lexicon."""
    toks = _tokenize(jd_text)
    if not toks:
        return []
    # Frequency count
    freq = {}
    for t in toks:
        if t in COMMON_SKILL_WORDS and len(t) >= 2:
            freq[t] = freq.get(t, 0) + 1
    # Sort by frequency
    skills = sorted(freq, key=freq.get, reverse=True)[:top_n]
    return skills


def _skills_weight_score(resume_text: str, skills: List[str]) -> float:
    """Return a [0..1] score based on fraction of skills present in resume."""
    if not skills:
        return 0.0
    toks = set(_tokenize(resume_text))
    hit = sum(1 for s in skills if s in toks)
    return hit / max(1, len(skills))


def _match_core(jd_text: str, resumes: List[Tuple[str, str]]) -> pd.DataFrame:
    """
    resumes: list of (filename, text)
    Returns DataFrame with Candidate, CosineScore, SkillsScore, MatchScore(%)
    """
    docs = [jd_text] + [r_text for _, r_text in resumes]
    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(docs)
    cos = cosine_similarity(X[0:1], X[1:]).flatten()  # 0..1

    if USE_SKILLS_WEIGHTING:
        jd_skills = _extract_skills_from_jd(jd_text)
        skills_scores = [
            _skills_weight_score(resumes[i][1], jd_skills) for i in range(len(resumes))
        ]
        # Blend: 80% cosine, 20% skills
        blended = (0.8 * cos) + (0.2 * pd.Series(skills_scores))
    else:
        jd_skills = []
        skills_scores = [0.0] * len(resumes)
        blended = cos

    df = pd.DataFrame({
        "Candidate": [name for name, _ in resumes],
        "CosineScore": cos,
        "SkillsScore": skills_scores,
        "Match Score (%)": (blended * 100).round(1)
    }).sort_values("Match Score (%)", ascending=False).reset_index(drop=True)

    # Add a short explanation column
    if USE_SKILLS_WEIGHTING:
        expl = []
        for i, (name, text) in enumerate(resumes):
            present = [s for s in _extract_skills_from_jd(jd_text) if s in set(_tokenize(text))]
            expl.append(f"Skills matched: {', '.join(present[:8])}" if present else "Few/no listed skills matched")
        df.insert(2, "Explanation", expl)

    return df


def _csv_download_href(df: pd.DataFrame, filename="resume_matches.csv") -> str:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    b64 = base64.b64encode(csv_bytes).decode("ascii")
    return f'<a download="{filename}" href="data:text/csv;base64,{b64}"><button>Download CSV</button></a>'


def _err_box(msg: str) -> str:
    return f'<div class="alert">{html.escape(msg)}</div>'


# --------------------------- Routes ---------------------------

@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(PAGE_HEAD + PAGE_FORM + PAGE_TAIL)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.post("/match", response_class=HTMLResponse)
async def match_ui(request: Request, jd: UploadFile, resumes: List[UploadFile]):
    # Validate presence
    if not jd or not resumes:
        return HTMLResponse(PAGE_HEAD + _err_box("Upload a JD and at least one resume.") + PAGE_FORM + PAGE_TAIL)

    if len(resumes) > MAX_FILES:
        return HTMLResponse(PAGE_HEAD + _err_box(f"Too many resumes. Max is {MAX_FILES}.") + PAGE_FORM + PAGE_TAIL)

    # Read JD
    jd_bytes = await jd.read()
    if len(jd_bytes) > MAX_FILE_MB * 1024 * 1024:
        return HTMLResponse(PAGE_HEAD + _err_box(f"JD too large. Max {MAX_FILE_MB} MB.") + PAGE_FORM + PAGE_TAIL)
    jd_text = _extract_text_from_bytes(jd_bytes, jd.filename)

    if not jd_text.strip():
        return HTMLResponse(PAGE_HEAD + _err_box("Could not extract text from JD (is it scanned/image-only?).") + PAGE_FORM + PAGE_TAIL)

    # Read resumes
    resumes_texts: List[Tuple[str, str]] = []
    for r in resumes:
        b = await r.read()
        if len(b) > MAX_FILE_MB * 1024 * 1024:
            return HTMLResponse(PAGE_HEAD + _err_box(f"File {html.escape(r.filename)} exceeds {MAX_FILE_MB} MB.") + PAGE_FORM + PAGE_TAIL)
        text = _extract_text_from_bytes(b, r.filename)
        resumes_texts.append((r.filename, text))

    # Compute
    df = _match_core(jd_text, resumes_texts)
    table_html = df.to_html(index=False, classes="table")
    download = _csv_download_href(df)

    body = f"""
    {PAGE_HEAD}
    <div class="success">Processed {len(resumes_texts)} resume(s).</div>
    {PAGE_FORM}
    <div class="card">
      <h3>Results</h3>
      {download}
      {table_html}
      <p class="muted">Cosine = TF-IDF similarity. SkillsScore = fraction of JD skills found in resume.
      Final score blends both (80/20) when skills weighting is enabled.</p>
    </div>
    {PAGE_TAIL}
    """
    return HTMLResponse(body)


# --------------------------- JSON API ---------------------------

@app.post("/api/match")
async def api_match(jd: UploadFile, resumes: List[UploadFile]):
    """
    JSON API.
    Form-data:
      - jd: file
      - resumes: files[]
    Returns:
      {
        "candidates": [{"name": str, "cosine": float, "skills": float, "score_pct": float, "explanation": str}, ...],
        "count": int
      }
    """
    if not jd or not resumes:
        return JSONResponse({"error": "Upload 'jd' and at least one 'resumes' file."}, status_code=400)

    jd_bytes = await jd.read()
    jd_text = _extract_text_from_bytes(jd_bytes, jd.filename)
    if not jd_text.strip():
        return JSONResponse({"error": "Could not extract text from JD."}, status_code=400)

    res_list: List[Tuple[str, str]] = []
    for r in resumes:
        b = await r.read()
        res_list.append((r.filename, _extract_text_from_bytes(b, r.filename)))

    df = _match_core(jd_text, res_list)

    # Build JSON
    payload = []
    for _, row in df.iterrows():
        item = {
            "name": row["Candidate"],
            "cosine": float(row["CosineScore"]),
            "skills": float(row["SkillsScore"]),
            "score_pct": float(row["Match Score (%)"]),
        }
        if "Explanation" in df.columns:
            item["explanation"] = row["Explanation"]
        payload.append(item)

    return {"count": len(payload), "candidates": payload}
