import os
import re
import datetime
import streamlit as st
import pdfplumber
from docx import Document
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Analyzer", page_icon="📄", layout="wide")

# ─── Job Title → Required Skills Map ─────────────────────────────────────────
# When user types only a job title, we use this to know what skills to expect.

JOB_TITLE_SKILLS = {
    "python developer": {
        "technical": ["python","django","flask","fastapi","rest api","sql","git","docker","postgresql","redis","celery","pytest"],
        "soft":       ["problem solving","communication","teamwork","attention to detail"],
        "keywords":   ["backend","api","web","database","server","microservice","deployment","linux","bash"]
    },
    "frontend developer": {
        "technical": ["javascript","typescript","react","angular","vue.js","html","css","git","rest api","webpack","figma"],
        "soft":       ["creativity","communication","attention to detail","teamwork"],
        "keywords":   ["ui","ux","responsive","component","browser","accessibility","web","design"]
    },
    "full stack developer": {
        "technical": ["javascript","react","node.js","python","sql","mongodb","rest api","git","docker","html","css"],
        "soft":       ["problem solving","communication","teamwork","adaptability"],
        "keywords":   ["frontend","backend","api","database","deployment","web","server"]
    },
    "data analyst": {
        "technical": ["python","sql","pandas","numpy","matplotlib","tableau","power bi","excel","data analysis","data visualization"],
        "soft":       ["analytical thinking","communication","attention to detail","critical thinking"],
        "keywords":   ["dashboard","report","insight","trend","query","visualization","business","kpi","metric"]
    },
    "data scientist": {
        "technical": ["python","machine learning","deep learning","scikit-learn","tensorflow","pytorch","pandas","numpy","sql","statistics"],
        "soft":       ["analytical thinking","problem solving","communication","critical thinking"],
        "keywords":   ["model","prediction","feature","training","accuracy","dataset","experiment","research"]
    },
    "machine learning engineer": {
        "technical": ["python","machine learning","deep learning","tensorflow","pytorch","scikit-learn","docker","kubernetes","mlflow","aws"],
        "soft":       ["analytical thinking","problem solving","teamwork"],
        "keywords":   ["model","training","inference","pipeline","deployment","production","mlops","experiment"]
    },
    "data engineer": {
        "technical": ["python","sql","spark","hadoop","kafka","airflow","aws","azure","docker","postgresql","mongodb","etl"],
        "soft":       ["problem solving","attention to detail","teamwork"],
        "keywords":   ["pipeline","etl","warehouse","ingestion","batch","streaming","orchestration","datalake"]
    },
    "devops engineer": {
        "technical": ["docker","kubernetes","aws","azure","ci/cd","linux","bash","terraform","ansible","git","python"],
        "soft":       ["problem solving","communication","teamwork","adaptability"],
        "keywords":   ["deployment","automation","infrastructure","pipeline","monitoring","cloud","container","orchestration"]
    },
    "cloud engineer": {
        "technical": ["aws","azure","google cloud platform","docker","kubernetes","terraform","ci/cd","linux","python"],
        "soft":       ["problem solving","communication","adaptability"],
        "keywords":   ["cloud","infrastructure","scaling","serverless","migration","cost","security","monitoring"]
    },
    "backend developer": {
        "technical": ["python","java","node.js","rest api","sql","mongodb","postgresql","docker","git","redis"],
        "soft":       ["problem solving","communication","attention to detail","teamwork"],
        "keywords":   ["api","server","database","microservice","backend","authentication","security","performance"]
    },
    "java developer": {
        "technical": ["java","spring boot","sql","rest api","maven","git","docker","postgresql","junit","microservices"],
        "soft":       ["problem solving","teamwork","attention to detail"],
        "keywords":   ["backend","api","spring","microservice","deployment","oop","design pattern"]
    },
    "android developer": {
        "technical": ["java","kotlin","android","sql","rest api","git","firebase"],
        "soft":       ["creativity","problem solving","attention to detail"],
        "keywords":   ["mobile","app","ui","activity","fragment","sdk","play store","material design"]
    },
    "ios developer": {
        "technical": ["swift","objective-c","xcode","rest api","git","firebase","sql"],
        "soft":       ["creativity","problem solving","attention to detail"],
        "keywords":   ["mobile","app","ui","uikit","swiftui","app store","cocoapods"]
    },
    "nlp engineer": {
        "technical": ["python","natural language processing","machine learning","deep learning","tensorflow","pytorch","scikit-learn","pandas"],
        "soft":       ["analytical thinking","problem solving","research"],
        "keywords":   ["text","classification","sentiment","entity","transformer","bert","gpt","tokenization","language model"]
    },
    "ai engineer": {
        "technical": ["python","machine learning","deep learning","tensorflow","pytorch","computer vision","natural language processing","docker"],
        "soft":       ["analytical thinking","problem solving","communication"],
        "keywords":   ["model","inference","training","pipeline","research","production","generative","llm"]
    },
    "software engineer": {
        "technical": ["python","java","javascript","sql","git","docker","rest api","data structures","algorithms"],
        "soft":       ["problem solving","teamwork","communication","adaptability"],
        "keywords":   ["design","system","architecture","testing","debugging","code review","agile","scrum"]
    },
    "software developer": {
        "technical": ["python","java","javascript","sql","git","rest api"],
        "soft":       ["problem solving","teamwork","communication"],
        "keywords":   ["development","testing","debugging","agile","version control","deployment"]
    },
    "web developer": {
        "technical": ["html","css","javascript","react","node.js","sql","git","rest api"],
        "soft":       ["creativity","communication","attention to detail"],
        "keywords":   ["responsive","web","frontend","backend","browser","hosting","cms"]
    },
    "cybersecurity analyst": {
        "technical": ["linux","python","sql","git","bash","aws"],
        "soft":       ["analytical thinking","attention to detail","critical thinking","problem solving"],
        "keywords":   ["security","threat","vulnerability","penetration","firewall","siem","incident","compliance","encryption"]
    },
    "database administrator": {
        "technical": ["sql","postgresql","mysql","mongodb","redis","python","bash","aws"],
        "soft":       ["attention to detail","problem solving","analytical thinking"],
        "keywords":   ["database","query","performance","backup","replication","schema","index","tuning"]
    },
    "project manager": {
        "technical": ["jira","confluence","agile","scrum","pmp","ms office"],
        "soft":       ["leadership","communication","project management","decision making","time management","teamwork"],
        "keywords":   ["delivery","milestone","stakeholder","budget","risk","planning","coordination","sprint","roadmap"]
    },
    "business analyst": {
        "technical": ["sql","tableau","power bi","excel","jira","confluence"],
        "soft":       ["analytical thinking","communication","critical thinking","presentation skills","problem solving"],
        "keywords":   ["requirement","analysis","stakeholder","process","workflow","report","insight","documentation"]
    },
    "ui ux designer": {
        "technical": ["figma","photoshop","css","html","javascript"],
        "soft":       ["creativity","communication","attention to detail","empathy"],
        "keywords":   ["design","prototype","wireframe","user","interface","usability","research","interaction","accessibility"]
    },
    "qa engineer": {
        "technical": ["python","selenium","jira","git","sql","rest api","postman"],
        "soft":       ["attention to detail","analytical thinking","communication","problem solving"],
        "keywords":   ["testing","automation","test case","bug","regression","integration","quality","manual","api testing"]
    },
    "embedded systems engineer": {
        "technical": ["c++","python","linux","bash","git"],
        "soft":       ["problem solving","attention to detail","analytical thinking"],
        "keywords":   ["firmware","microcontroller","rtos","hardware","driver","protocol","iot","uart","spi","i2c"]
    },
}

def resolve_job_title(title_input):
    """
    Match user input to a known job title (fuzzy/partial match).
    Returns (matched_title, skills_dict) or (None, None).
    """
    if not title_input:
        return None, None
    inp = title_input.lower().strip()
    # Exact match first
    if inp in JOB_TITLE_SKILLS:
        return inp, JOB_TITLE_SKILLS[inp]
    # Partial match — check if input words appear in any known title
    for title, skills in JOB_TITLE_SKILLS.items():
        title_words = set(title.split())
        inp_words   = set(inp.split())
        if inp_words & title_words:          # at least one word in common
            return title, skills
    return None, None

def expand_jd_with_title(job_title, job_desc, skills_db):
    """
    If user gave a job title, add its required skills to the JD context.
    Returns an enriched combined text to analyze against.
    """
    matched_title, title_skills = resolve_job_title(job_title)
    if not title_skills:
        return job_desc or "", matched_title

    # Build a synthetic JD from title skills if no JD provided
    synthetic = " ".join(
        title_skills.get("technical", []) +
        title_skills.get("soft", []) +
        title_skills.get("keywords", [])
    )
    combined = (job_desc or "") + " " + synthetic
    return combined.strip(), matched_title

# ─── Load Skills ──────────────────────────────────────────────────────────────

def load_skills(path="skills.txt"):
    skills = {"technical": [], "soft": [], "certifications": []}
    current = "technical"
    if not os.path.exists(path):
        st.warning("skills.txt not found. Place it in the same folder as app.py")
        return skills
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("# Technical"):
                current = "technical"
            elif line.startswith("# Soft"):
                current = "soft"
            elif line.startswith("# Cert"):
                current = "certifications"
            elif not line.startswith("#"):
                skills[current].append(line.lower())
    return skills

# ─── Extract Text ─────────────────────────────────────────────────────────────

def extract_text(uploaded_file):
    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)
    elif ext == "docx":
        doc = Document(uploaded_file)
        return "\n".join(p.text for p in doc.paragraphs)
    elif ext == "txt":
        return uploaded_file.read().decode("utf-8", errors="ignore")
    return ""

# ─── Skill Matching ───────────────────────────────────────────────────────────

def match_skills(text, skills_db):
    text_lower = text.lower()
    found = {"technical": [], "soft": [], "certifications": []}
    for category, skill_list in skills_db.items():
        for skill in skill_list:
            if re.search(r"\b" + re.escape(skill) + r"\b", text_lower):
                found[category].append(skill.title())
    return found

# ─── Experience ───────────────────────────────────────────────────────────────

def estimate_experience(text):
    for pat in [r"(\d+)\+?\s*years?\s+of\s+(?:work\s+)?experience",
                r"(\d+)\+?\s*years?\s+experience",
                r"experience\s+of\s+(\d+)\+?\s*years?"]:
        m = re.search(pat, text.lower())
        if m:
            return int(m.group(1))
    cy = datetime.datetime.now().year
    ranges = re.findall(r"(20\d\d)\s*[-–to]+\s*(20\d\d|present|current|now)", text.lower())
    if ranges:
        total = sum(max(0, (cy if e in ("present","current","now") else int(e)) - int(s)) for s, e in ranges)
        return min(total, 30)
    return 0

# ─── Constants ────────────────────────────────────────────────────────────────

IMPACT_WORDS   = ["increased","decreased","reduced","improved","saved","grew","delivered",
                  "launched","optimized","achieved","built","led","managed","developed",
                  "designed","implemented","automated","streamlined","deployed","created"]
METRIC_PATTERN = re.compile(r'\d+\s*(%|percent|x\b|times|million|thousand|k\b|users|customers|projects)')
SECTION_HEADERS= ["education","experience","skills","projects","summary","objective",
                  "certifications","achievements","work history","internship","training"]
WEAK_WORDS     = ["responsible for","helped with","assisted in","worked on","involved in","participated in"]
ACTION_VERBS   = ["built","led","managed","designed","developed","implemented","deployed",
                  "created","launched","optimized","automated","improved","delivered","achieved",
                  "architected","mentored","scaled","reduced","increased","analyzed"]
STOPWORDS      = {"the","and","or","to","a","an","of","in","for","is","are","with","that",
                  "be","will","you","have","on","at","by","from","as","we","this","our",
                  "it","can","all","not","your","their","they","who","has","its","but",
                  "also","both","must","more","than","into","such","was","were","been",
                  "should","would","could","may","which","when","what","how","any","each",
                  "work","role","position","team","company","experience","required","preferred",
                  "ability","strong","knowledge","good","years","looking","join","seeking"}

# ─── JD / Title Keyword Extraction ────────────────────────────────────────────

def extract_jd_keywords(combined_text, skills_db):
    if not combined_text or not combined_text.strip():
        return [], [], []
    tl = combined_text.lower()
    jd_tech = [s for s in skills_db["technical"]      if re.search(r"\b" + re.escape(s) + r"\b", tl)]
    jd_soft = [s for s in skills_db["soft"]           if re.search(r"\b" + re.escape(s) + r"\b", tl)]
    jd_cert = [s for s in skills_db["certifications"] if re.search(r"\b" + re.escape(s) + r"\b", tl)]
    cleaned = re.sub(r"[^\w\s]", " ", tl)
    words   = [w for w in cleaned.split() if len(w) >= 3 and w not in STOPWORDS and not w.isdigit()]
    top_kw  = [w for w, _ in Counter(words).most_common(50)]
    return jd_tech, jd_soft + jd_cert, top_kw

# ─── Job-Aware Score Breakdown ────────────────────────────────────────────────

def score_breakdown(text, matched, years, jd_tech, jd_soft, jd_keywords, has_job_context):
    tl = text.lower()

    # 1. SKILLS (0-25)
    if has_job_context and jd_tech:
        resume_tech_lower = [s.lower() for s in matched["technical"]]
        matched_jd_tech   = sum(1 for s in jd_tech if s in resume_tech_lower)
        skill_score = min(25, int((matched_jd_tech / len(jd_tech)) * 25))
        skill_note  = f"{matched_jd_tech}/{len(jd_tech)} required tech skills matched"
    else:
        total = sum(len(v) for v in matched.values())
        skill_score = min(25, int((total / 20) * 25))
        skill_note  = f"{total} skills detected in resume"

    # 2. EXPERIENCE (0-20)
    if has_job_context and jd_keywords:
        exp_start = tl.find("experience")
        exp_text  = tl[exp_start:exp_start+2500] if exp_start != -1 else tl
        hits      = sum(1 for kw in jd_keywords if re.search(r"\b" + re.escape(kw) + r"\b", exp_text))
        exp_score = min(20, hits + min(6, years))
        exp_note  = f"{hits} job-relevant keywords in experience section"
    else:
        ac        = sum(1 for v in ACTION_VERBS if v in tl)
        exp_score = min(20, ac * 2 + min(10, years * 2))
        exp_note  = f"{ac} action verbs, {years} yr(s) experience"

    # 3. IMPACT (0-20) — always resume-based
    impact_hits  = sum(1 for w in IMPACT_WORDS if w in tl)
    metric_count = len(METRIC_PATTERN.findall(text))
    impact_score = min(20, impact_hits + metric_count * 3)
    impact_note  = f"{impact_hits} impact words, {metric_count} measurable metric(s)"

    # 4. SKILL RELEVANCE vs Structure (0-20)
    if has_job_context and (jd_tech or jd_soft):
        all_jd   = jd_tech + jd_soft
        res_all  = [s.lower() for s in matched["technical"] + matched["soft"] + matched["certifications"]]
        matched_all = sum(1 for s in all_jd if s in res_all)
        rel_score   = min(20, int((matched_all / max(len(all_jd), 1)) * 20))
        rel_note    = f"{matched_all}/{len(all_jd)} job-required skills present"
        label4      = "Skill Relevance"
    else:
        secs      = sum(1 for s in SECTION_HEADERS if s in tl)
        rel_score = min(20, secs * 3)
        rel_note  = f"{secs} standard sections found"
        label4    = "Structure"

    # 5. EDUCATION (0-15)
    edu_words = ["bachelor","master","phd","b.tech","m.tech","b.e","m.e","b.sc","m.sc",
                 "mba","degree","university","college","diploma","10th","12th","cgpa","gpa"]
    edu_count = sum(1 for w in edu_words if w in tl)
    edu_score = min(15, edu_count * 3)
    edu_note  = f"{edu_count} education keyword(s) found"

    bd = {
        "Skills":    (skill_score,  25, skill_note),
        "Experience":(exp_score,    20, exp_note),
        "Impact":    (impact_score, 20, impact_note),
        label4:      (rel_score,    20, rel_note),
        "Education": (edu_score,    15, edu_note),
    }
    return bd, sum(v for v, _, _ in bd.values())

# ─── ATS Score ────────────────────────────────────────────────────────────────

def calc_ats(text, matched):
    score, tl = 0, text.lower()
    if re.search(r"[\w.+-]+@[\w-]+\.\w+", text): score += 15
    if re.search(r"\+?\d[\d\s\-().]{8,}", text):  score += 10
    for kw in ["experience","education","skills"]:
        if kw in tl: score += 7
    score += min(20, sum(len(v) for v in matched.values()) * 2)
    wc = len(text.split())
    if 300 <= wc <= 1200: score += 10
    elif wc > 0: score += 5
    if not re.search(r"[\w.+-]+@[\w-]+\.\w+", text) and not re.search(r"\+?\d[\d\s\-().]{8,}", text):
        score -= 10
    return max(0, min(100, score))

# ─── Job Match Analysis — TF-IDF Cosine Similarity (ML Algorithm) ──────────────
#
# TF-IDF (Term Frequency - Inverse Document Frequency) is a classic NLP/ML
# technique that converts text into numeric vectors based on word importance.
# Cosine Similarity then measures the angle between those vectors — the closer
# to 1.0, the more similar the two documents are.
#
# This is a real ML algorithm from scikit-learn, not just keyword counting.

def job_match_analysis(resume_text, combined_text, skills_db):
    """
    ML-based job match using TF-IDF + Cosine Similarity.
    Also returns matched/missing keywords for display.
    """
    if not combined_text or not combined_text.strip():
        return None, [], [], 0

    # ── Step 1: TF-IDF Cosine Similarity Score ────────────────────────────────
    # Vectorize both texts using TF-IDF (unigrams + bigrams for multi-word skills)
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),      # capture both single words and pairs like "machine learning"
            stop_words="english",    # remove common English stop words
            max_features=5000,       # limit vocabulary size for efficiency
            sublinear_tf=True        # apply log normalization to term frequency
        )
        tfidf_matrix = vectorizer.fit_transform([resume_text.lower(), combined_text.lower()])
        similarity   = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        tfidf_score  = int(round(similarity * 100))
    except Exception:
        tfidf_score = 0

    # ── Step 2: Keyword gap analysis (for display — what's missing/present) ───
    jd_lower     = combined_text.lower()
    resume_lower = resume_text.lower()

    all_skills   = skills_db["technical"] + skills_db["soft"] + skills_db["certifications"]
    skills_in_jd = [s for s in all_skills if re.search(r"\b" + re.escape(s) + r"\b", jd_lower)]

    jd_kw_counter = Counter(re.sub(r"[^\w\s]", " ", jd_lower).split())
    top_jd_kw     = [w for w, _ in jd_kw_counter.most_common(60)
                     if len(w) >= 3 and w not in STOPWORDS]
    all_terms     = list(dict.fromkeys(skills_in_jd + top_jd_kw))

    matched_t, missing_t = [], []
    for term in all_terms:
        if re.search(r"\b" + re.escape(term) + r"\b", resume_lower):
            matched_t.append(term.title())
        else:
            missing_t.append(term.title())

    # ── Step 3: Combine TF-IDF score with keyword match for final score ───────
    keyword_score = int(len(matched_t) / len(all_terms) * 100) if all_terms else 0
    # Weighted blend: 60% TF-IDF (semantic similarity) + 40% keyword match
    final_score   = int(0.6 * tfidf_score + 0.4 * keyword_score)

    skill_set = set(s.lower() for s in all_skills)
    gap       = [t for t in missing_t if t.lower() in skill_set][:8]
    if len(gap) < 5:
        gap  += [t for t in missing_t if t not in gap][:5 - len(gap)]

    return final_score, gap, matched_t[:15], len(all_terms)

# ─── Weaknesses ───────────────────────────────────────────────────────────────

def get_weaknesses(text, matched, years):
    areas, tl = [], text.lower()
    if not re.search(r"[\w.+-]+@[\w-]+\.\w+", text):    areas.append("Missing email address")
    if not re.search(r"\+?\d[\d\s\-().]{8,}", text):     areas.append("Missing phone number")
    if not any(k in tl for k in ["linkedin","github","portfolio"]): areas.append("No LinkedIn / GitHub / Portfolio link")
    if not METRIC_PATTERN.search(text):                  areas.append("No measurable achievements — add numbers like '30% improvement'")
    if len(matched["soft"]) < 3:                         areas.append("Very few soft skills mentioned")
    if any(w in tl for w in WEAK_WORDS):                 areas.append("Weak phrases found — replace with strong action verbs")
    if len(text.split()) < 250:                          areas.append("Resume too short — add more detail")
    if "summary" not in tl and "objective" not in tl:   areas.append("Missing professional summary / objective section")
    if years == 0 and "intern" not in tl and "project" not in tl:
        areas.append("No experience/internship/projects found")
    return areas[:6]

# ─── Recommendations ──────────────────────────────────────────────────────────

def get_recommendations(text, matched, bd, years, jm_score):
    recs, tl = [], text.lower()
    impact_score = bd.get("Impact", (0,))[0]
    rel_score    = bd.get("Skill Relevance", bd.get("Structure", (0,)))[0]
    if impact_score < 10:      recs.append("Add quantifiable results e.g. 'Reduced load time by 40%'")
    if rel_score < 9:          recs.append("Add clear sections: Summary, Experience, Skills, Education, Projects")
    if len(matched["technical"]) < 4: recs.append("List all technical skills — even basic ones count")
    if "project" not in tl:   recs.append("Add a Projects section with tech used and outcomes")
    if years == 0:             recs.append("Add internships, freelance, or college projects under experience")
    if not any(k in tl for k in ["linkedin","github"]): recs.append("Add GitHub/LinkedIn profile URL")
    if jm_score is not None and jm_score < 50:
        recs.append("Tailor this resume — mirror keywords from the job description / title")
    if len(matched["certifications"]) == 0: recs.append("Add certifications — free ones from Google/Coursera help")
    return recs[:5]

# ─── Job Titles ───────────────────────────────────────────────────────────────

def get_job_titles(matched, years):
    tech = [s.lower() for s in matched["technical"]]
    c = []
    if any(s in tech for s in ["machine learning","deep learning","tensorflow","pytorch","scikit-learn"]): c.append(("Machine Learning Engineer",5))
    if any(s in tech for s in ["natural language processing","computer vision"]): c.append(("AI / NLP Engineer",5))
    if any(s in tech for s in ["data analysis","pandas","numpy","tableau","power bi"]): c.append(("Data Analyst",4))
    if any(s in tech for s in ["spark","hadoop","kafka"]): c.append(("Data Engineer",4))
    if any(s in tech for s in ["aws","docker","kubernetes","ci/cd","azure"]): c.append(("DevOps / Cloud Engineer",4))
    if any(s in tech for s in ["python","django","flask","fastapi"]): c.append(("Python Developer",3))
    if any(s in tech for s in ["react","angular","vue.js","javascript","html","css"]): c.append(("Frontend Developer",3))
    if any(s in tech for s in ["node.js","express.js","rest api"]): c.append(("Backend Developer",3))
    if any(s in tech for s in ["java","spring boot"]): c.append(("Java Developer",3))
    if any(s in tech for s in ["sql","mongodb","postgresql","mysql"]): c.append(("Database Developer",2))
    c.sort(key=lambda x: -x[1])
    titles = [t for t, _ in c[:3]] or ["Software Developer"]
    if years >= 8:   titles = ["Lead " + t for t in titles]
    elif years >= 5: titles = ["Senior " + t for t in titles]
    return titles[:3]

def grade(s):
    if s >= 80: return "🟢 Excellent"
    if s >= 65: return "🟡 Good"
    if s >= 50: return "🟠 Average"
    return "🔴 Needs Work"

def get_level(years, tc):
    if years == 0 and tc < 4: return "Fresher"
    if years <= 2:  return "Junior"
    if years <= 5:  return "Mid-Level"
    if years <= 9:  return "Senior"
    return "Lead / Principal"

# ─── UI ───────────────────────────────────────────────────────────────────────

st.title("📄 AI Resume Analyzer")
st.caption("Analyzes your resume against a **Job Title**, a **Job Description**, or **both**.")
st.divider()

c1, c2 = st.columns([1, 1])
with c1:
    uploaded_file = st.file_uploader("Upload Resume (PDF / DOCX / TXT)", type=["pdf","docx","txt"])

with c2:
    job_title = st.text_input(
        "Job Title (e.g. Python Developer, Data Analyst)",
        placeholder="Type the job title you are applying for..."
    )
    job_desc = st.text_area(
        "Job Description (optional — paste full JD for deeper analysis)",
        height=100,
        placeholder="Paste the job description here for keyword-level matching..."
    )

analyze = st.button("🔍 Analyze Resume", use_container_width=True, type="primary")
st.divider()

if analyze:
    if not uploaded_file:
        st.error("Please upload a resume file first.")
        st.stop()

    if not job_title.strip() and not job_desc.strip():
        st.warning("Tip: Enter a Job Title or Job Description for job-specific scoring. Analyzing resume quality only for now.")

    with st.spinner("Analyzing your resume..."):
        text = extract_text(uploaded_file)
        if not text.strip():
            st.error("Could not extract text. Try a different format.")
            st.stop()

        skills_db = load_skills("skills.txt")
        matched   = match_skills(text, skills_db)
        years     = estimate_experience(text)

        # Merge job title skills + job description into one context
        combined_context, matched_title = expand_jd_with_title(job_title, job_desc, skills_db)
        has_job_context = bool(combined_context.strip())

        jd_tech, jd_soft, jd_kw = extract_jd_keywords(combined_context, skills_db)
        breakdown, overall       = score_breakdown(text, matched, years, jd_tech, jd_soft, jd_kw, has_job_context)
        ats                      = calc_ats(text, matched)
        level                    = get_level(years, len(matched["technical"]))
        weaknesses               = get_weaknesses(text, matched, years)
        titles                   = get_job_titles(matched, years)
        jm_score, gap, jm_matched, jm_total = job_match_analysis(text, combined_context, skills_db)
        recs                     = get_recommendations(text, matched, breakdown, years, jm_score)

        lines = [l.strip() for l in text.splitlines() if l.strip()]
        name  = lines[0] if lines else "Unknown"
        if len(name) > 40 or "@" in name or sum(c.isdigit() for c in name) > 2:
            name = uploaded_file.name.rsplit(".", 1)[0]

    # ── Job Context Banner ──
    if matched_title:
        st.info(f"🎯 Analyzing against job title: **{matched_title.title()}**" +
                (" + your job description" if job_desc.strip() else ""))
    elif job_desc.strip():
        st.info("🎯 Analyzing against your pasted job description.")
    else:
        st.info("📋 No job title/description provided — showing general resume quality scores.")

    # ── Candidate ──
    st.subheader(f"👤 {name}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Experience Level", level)
    m2.metric("Years of Experience", f"{years} yr(s)")
    m3.metric("Word Count", len(text.split()))
    m4.metric("Skills Found", sum(len(v) for v in matched.values()))
    st.divider()

    # ── Scores ──
    st.subheader("📊 Scores")
    s1, s2, s3 = st.columns(3)
    s1.metric("Overall Score",    f"{overall} / 100", grade(overall))
    s2.metric("ATS Score",        f"{ats} / 100",     grade(ats))
    if jm_score is not None:
        s3.metric("Job Match Score (ML)",  f"{jm_score} / 100",  grade(jm_score))
    else:
        s3.metric("Job Match Score", "N/A", "Enter a job title or JD above")
    st.divider()

    # ── Score Breakdown ──
    if has_job_context:
        st.subheader("📈 Score Breakdown *(adjusted for this job)*")
        st.caption("Scores change based on the job title / description you entered.")
    else:
        st.subheader("📈 Score Breakdown *(resume-based)*")
        st.caption("Enter a job title or paste a JD above to get job-specific scores.")

    for label, (val, max_val, note) in breakdown.items():
        pct = val / max_val
        ca, cb = st.columns([3, 1])
        with ca:
            st.write(f"**{label}** — {val} / {max_val}")
            st.progress(pct)
            st.caption(f"📌 {note}")
        with cb:
            st.metric("", f"{int(pct*100)}%")
        st.write("")
    st.divider()

    # ── Job Match Detail ──
    if jm_score is not None:
        st.subheader(f"🎯 Job Match Analysis — {jm_score}%")
        st.caption(f"Score calculated using **TF-IDF Cosine Similarity** (ML) + keyword matching. Matched {len(jm_matched)} out of {jm_total} keywords from the job context.")
        jma, jmb = st.columns(2)
        with jma:
            st.markdown("**✅ Keywords You Have**")
            if jm_matched:
                mc = st.columns(3)
                for i, kw in enumerate(jm_matched): mc[i % 3].success(kw)
            else:
                st.info("No matching keywords found.")
        with jmb:
            st.markdown("**❌ Missing Keywords / Skills**")
            if gap:
                gc = st.columns(3)
                for i, kw in enumerate(gap): gc[i % 3].error(kw)
            else:
                st.success("No major gaps — great match!")
        st.divider()

    # ── Skills ──
    st.subheader("🛠️ Skills Detected in Resume")
    t1, t2, t3 = st.tabs([f"Technical ({len(matched['technical'])})",
                           f"Soft Skills ({len(matched['soft'])})",
                           f"Certifications ({len(matched['certifications'])})"])
    with t1:
        if matched["technical"]:
            cols = st.columns(4)
            for i, s in enumerate(matched["technical"]): cols[i%4].success(s)
        else:
            st.info("No technical skills detected. List languages/tools clearly in your resume.")
    with t2:
        if matched["soft"]:
            cols = st.columns(4)
            for i, s in enumerate(matched["soft"]): cols[i%4].info(s)
        else:
            st.info("No soft skills detected. Mention: communication, leadership, teamwork, etc.")
    with t3:
        if matched["certifications"]:
            cols = st.columns(4)
            for i, s in enumerate(matched["certifications"]): cols[i%4].warning(s)
        else:
            st.info("No certifications detected.")
    st.divider()

    # ── Job Titles ──
    st.subheader("💼 Best Fit Job Titles")
    if titles:
        jc = st.columns(len(titles))
        for i, t in enumerate(titles): jc[i].info(f"💼 {t}")
    else:
        st.info("Add more technical skills to determine best fit roles.")
    st.divider()

    # ── Weaknesses & Recommendations ──
    wa, wb = st.columns(2)
    with wa:
        st.subheader("⚠️ Improvement Areas")
        for w in weaknesses: st.warning(w)
        if not weaknesses:   st.success("✅ No major issues found!")
    with wb:
        st.subheader("✅ Recommendations")
        for r in recs:       st.info(f"💡 {r}")
        if not recs:         st.success("Your resume looks well-optimized!")
    st.divider()
    st.caption("Analysis complete. Best of luck with your job search! 🚀")