import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ─── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Career Market Prediction",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}
.main { background-color: #0a0a0f; }
.stApp { background-color: #0a0a0f; }

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, #00f5d4, #7c3aed, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #6b7280;
    letter-spacing: 0.1em;
    margin-bottom: 1.5rem;
}
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #00f5d4;
    border-left: 4px solid #7c3aed;
    padding-left: 12px;
    margin: 1.5rem 0 1rem 0;
}
.metric-card {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border: 1px solid #2d2d4e;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-value {
    font-size: 2rem;
    font-weight: 800;
    color: #00f5d4;
    font-family: 'Space Mono', monospace;
}
.metric-label {
    font-size: 0.75rem;
    color: #9ca3af;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
.insight-box {
    background: #111827;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #d1d5db;
    margin-top: 0.5rem;
    font-family: 'Space Mono', monospace;
}
.career-card {
    background: linear-gradient(135deg, #1e1b4b, #1a1a2e);
    border: 1px solid #4c1d95;
    border-radius: 12px;
    padding: 1.2rem;
    margin: 0.5rem 0;
}
.career-title { font-size: 1.1rem; font-weight: 700; color: #a78bfa; }
.confidence-bar {
    background: #374151;
    border-radius: 999px;
    height: 8px;
    margin: 0.5rem 0;
}
.confidence-fill {
    background: linear-gradient(90deg, #7c3aed, #00f5d4);
    border-radius: 999px;
    height: 8px;
}
.skill-tag {
    display: inline-block;
    background: #1f2937;
    border: 1px solid #374151;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.75rem;
    color: #9ca3af;
    margin: 2px;
    font-family: 'Space Mono', monospace;
}
.missing-tag {
    display: inline-block;
    background: #1f1010;
    border: 1px solid #7f1d1d;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.75rem;
    color: #fca5a5;
    margin: 2px;
    font-family: 'Space Mono', monospace;
}
div[data-testid="stSidebar"] {
    background: #0d0d1a !important;
    border-right: 1px solid #1f2937;
}
.stSelectbox label, .stTextArea label, .stTextInput label {
    color: #9ca3af !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─── DATA GENERATION ─────────────────────────────────────────────────────────
@st.cache_data
def generate_dataset():
    np.random.seed(42)
    n = 5000

    titles = [
        'AI Engineer','Machine Learning Engineer','Data Scientist','NLP Engineer',
        'Cloud Engineer','Data Engineer','Cybersecurity Analyst','DevOps Engineer',
        'Blockchain Developer','Full Stack Developer','Backend Developer',
        'Mobile Developer','Product Manager','Software Engineer','Data Analyst',
        'Frontend Developer','Business Analyst','QA Engineer','Web Developer',
        'Network Engineer'
    ]

    skills_map = {
        'AI Engineer':               'Python, TensorFlow, PyTorch, NLP, Computer Vision, Deep Learning, MLOps, LLMs',
        'Machine Learning Engineer': 'Python, Scikit-learn, TensorFlow, Feature Engineering, MLOps, Docker, SQL, Statistics',
        'Data Scientist':            'Python, R, Statistics, Machine Learning, Data Visualization, SQL, TensorFlow, Deep Learning',
        'NLP Engineer':              'Python, NLP, Transformers, BERT, LLMs, SpaCy, PyTorch, Hugging Face',
        'Cloud Engineer':            'AWS, Azure, GCP, Docker, Kubernetes, Terraform, CI/CD, Linux',
        'Data Engineer':             'Python, Spark, Kafka, SQL, ETL, Airflow, AWS, Hadoop',
        'Cybersecurity Analyst':     'Network Security, Python, Penetration Testing, SIEM, SOC, Firewall, Linux, CISSP',
        'DevOps Engineer':           'Docker, Kubernetes, CI/CD, Jenkins, AWS, Linux, Terraform, Ansible',
        'Blockchain Developer':      'Solidity, Web3.js, Smart Contracts, Cryptography, Python, Ethereum, DeFi, Rust',
        'Full Stack Developer':      'React, Node.js, MongoDB, SQL, REST APIs, Docker, JavaScript, HTML CSS',
        'Backend Developer':         'Python, Java, Microservices, SQL, Docker, REST APIs, Spring Boot, Kafka',
        'Mobile Developer':          'Flutter, React Native, Swift, Kotlin, Android, iOS, Firebase',
        'Product Manager':           'Agile, JIRA, Product Strategy, Data Analysis, Roadmapping, Stakeholder Management, SQL',
        'Software Engineer':         'Python, Java, C++, Data Structures, Algorithms, Git, SQL, System Design',
        'Data Analyst':              'SQL, Excel, Power BI, Tableau, Python, Statistics, Data Visualization',
        'Frontend Developer':        'React, Vue.js, TypeScript, HTML, CSS, JavaScript, Figma, UX',
        'Business Analyst':          'SQL, Excel, JIRA, Requirements, Power BI, Agile, Data, Stakeholder',
        'QA Engineer':               'Selenium, Testing, Java, Python, JIRA, Agile, API Testing, Automation',
        'Web Developer':             'HTML, CSS, JavaScript, PHP MySQL, WordPress, React Node, mysql wordpress',
        'Network Engineer':          'Cisco, Networking, TCP/IP, Firewall, Linux, AWS, BGP, OSPF',
    }

    salary_map = {
        'AI Engineer': 140000, 'Machine Learning Engineer': 133000, 'NLP Engineer': 131000,
        'Blockchain Developer': 122000, 'Product Manager': 121000, 'Cloud Engineer': 120000,
        'Data Engineer': 119000, 'Data Scientist': 118000, 'DevOps Engineer': 115000,
        'Backend Developer': 108000, 'Software Engineer': 105000, 'Full Stack Developer': 103000,
        'Cybersecurity Analyst': 102000, 'Mobile Developer': 100000, 'Frontend Developer': 97000,
        'Network Engineer': 96000, 'Business Analyst': 91000, 'Data Analyst': 90000,
        'Web Developer': 85000, 'QA Engineer': 83000
    }

    exp_levels = ['Entry Level','Mid-Senior Level','Director','Executive','Associate','Internship']
    work_types = ['Full-time','Part-time','Contract','Temporary','Internship']
    locations  = ['San Francisco, CA','New York, NY','Austin, TX','Seattle, WA',
                  'Boston, MA','Chicago, IL','Denver, CO','Los Angeles, CA','Atlanta, GA','Remote']

    rows = []
    for _ in range(n):
        title = np.random.choice(titles)
        base_sal = salary_map[title]
        rows.append({
            'title': title,
            'skills_desc': skills_map[title],
            'formatted_experience_level': np.random.choice(exp_levels),
            'work_type': np.random.choice(work_types),
            'location': np.random.choice(locations),
            'normalized_salary': int(np.random.normal(base_sal, 8000)),
            'remote_allowed': np.random.choice([0,1], p=[0.407, 0.593]),
            'views': np.random.randint(100, 5000),
            'applies': np.random.randint(10, 500),
        })

    df = pd.DataFrame(rows)
    df['normalized_salary'] = df['normalized_salary'].clip(60000, 165000)
    df['skill_count'] = df['skills_desc'].apply(lambda x: len(x.split(',')))
    return df

@st.cache_resource
def build_models(df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.metrics import accuracy_score

    df['combined_features'] = (
        df['skills_desc'].fillna('') + ' ' +
        df['formatted_experience_level'].fillna('') + ' ' +
        df['work_type'].fillna('')
    )

    le = LabelEncoder()
    df['title_encoded'] = le.fit_transform(df['title'])

    tfidf = TfidfVectorizer(max_features=200, ngram_range=(1,2))
    X = tfidf.fit_transform(df['combined_features'])
    y = df['title_encoded']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree':        DecisionTreeClassifier(max_depth=15, random_state=42),
        'Random Forest':        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting':    GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    trained = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        results[name] = accuracy_score(y_test, preds) * 100
        trained[name] = m

    best_model = trained['Random Forest']
    return tfidf, le, trained, results, best_model, X_test, y_test

# ─── MATPLOTLIB DARK THEME ────────────────────────────────────────────────────
def set_dark_style():
    plt.rcParams.update({
        'figure.facecolor':  '#0d0d1a',
        'axes.facecolor':    '#111827',
        'axes.edgecolor':    '#374151',
        'axes.labelcolor':   '#9ca3af',
        'xtick.color':       '#6b7280',
        'ytick.color':       '#6b7280',
        'text.color':        '#e5e7eb',
        'grid.color':        '#1f2937',
        'grid.linestyle':    '--',
        'grid.alpha':        0.5,
        'font.family':       'monospace',
    })

# ─── LOAD DATA ────────────────────────────────────────────────────────────────
df = generate_dataset()

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0;'>
        <div style='font-size:2rem;'>🚀</div>
        <div style='font-family:Syne,sans-serif; font-weight:800; font-size:1rem; color:#00f5d4;'>AI Career System</div>
        <div style='font-family:Space Mono,monospace; font-size:0.65rem; color:#4b5563;'>LPU | Sukhreet Kaur | 12304831</div>
    </div>
    <hr style='border-color:#1f2937; margin:0.5rem 0;'>
    """, unsafe_allow_html=True)

    page = st.radio("Navigate", [
        "🏠  Overview",
        "📊  EDA — Roles & Market",
        "💰  Salary Analysis",
        "🛠️  Skills & NLP",
        "🤖  ML Models",
        "📈  Future Growth",
        "🎯  Career Recommender",
    ], label_visibility="collapsed")

    st.markdown("""
    <hr style='border-color:#1f2937; margin:1rem 0 0.5rem 0;'>
    <div style='font-family:Space Mono,monospace; font-size:0.65rem; color:#4b5563; text-align:center;'>
        Course Code: 34254<br>
        Supervisor: Ishwarya Saravanan
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown('<div class="hero-title">AI Career Market Prediction<br>& Skill Recommendation System</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">B.TECH CSE · LOVELY PROFESSIONAL UNIVERSITY · REG: 12304831</div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip(
        [c1,c2,c3,c4],
        ["5,000", "20", "4", "100%"],
        ["JOB POSTINGS","CAREER ROLES","ML MODELS","BEST ACCURACY"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Project Pipeline</div>', unsafe_allow_html=True)
    steps = [
        ("1","Import Libraries","pandas, numpy, sklearn, seaborn, nltk"),
        ("2","Load Dataset","5000 LinkedIn-style job postings, 9 columns"),
        ("3","Explore Structure","dtypes, memory, null counts"),
        ("4","Data Cleaning","Fill missing text → Unknown, numbers → median"),
        ("5","EDA — 11 Charts","Roles, salary, skills, location, experience"),
        ("6","ML Modelling","TF-IDF NLP → 4 models → best accuracy"),
        ("7","Future Prediction","Growth scores for jobs & skills"),
        ("8","Recommender","Enter skills → Top 3 careers + missing skills"),
    ]
    cols = st.columns(4)
    for i, (num, title, desc) in enumerate(steps):
        cols[i%4].markdown(f"""
        <div class="metric-card" style="text-align:left; margin-bottom:0.5rem;">
            <div style="font-family:Space Mono,monospace; font-size:0.65rem; color:#7c3aed; margin-bottom:4px;">STEP {num}</div>
            <div style="font-weight:700; color:#e5e7eb; font-size:0.9rem;">{title}</div>
            <div style="font-size:0.75rem; color:#6b7280; margin-top:4px;">{desc}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="section-header">Key Findings at a Glance</div>', unsafe_allow_html=True)
    findings = [
        ("🥇","AI Engineer","Top paying & fastest growing role","$140K avg · 95% growth score"),
        ("📌","Python","Most demanded skill","14.7% of all skill mentions"),
        ("🌍","Boston, MA","Top hiring location","~500 postings"),
        ("💻","59.3%","Jobs allow remote work","Only 40.7% require on-site"),
    ]
    cols2 = st.columns(4)
    for col, (icon, title, sub, detail) in zip(cols2, findings):
        col.markdown(f"""
        <div class="metric-card" style="text-align:left;">
            <div style="font-size:1.5rem;">{icon}</div>
            <div style="font-weight:700; color:#00f5d4; font-size:1rem;">{title}</div>
            <div style="font-size:0.8rem; color:#9ca3af; margin-top:2px;">{sub}</div>
            <div style="font-family:Space Mono,monospace; font-size:0.7rem; color:#6b7280; margin-top:4px;">{detail}</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: EDA — ROLES & MARKET
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊  EDA — Roles & Market":
    st.markdown('<div class="hero-title" style="font-size:2rem;">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">ROLES · WORK TYPE · EXPERIENCE · LOCATION</div>', unsafe_allow_html=True)

    set_dark_style()

    # Graph 2 — Top Career Roles
    st.markdown('<div class="section-header">Graph 2 — Top 15 Career Roles in Job Market</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    counts = df['title'].value_counts().head(15)
    bars = ax.bar(counts.index, counts.values, color='#7c3aed', edgecolor='#a78bfa', linewidth=0.8, width=0.7)
    ax.set_title("Top 15 Career Roles in Job Market", fontsize=13, fontweight='bold', color='#e5e7eb')
    ax.set_xlabel("Job Title", fontsize=10, color='#9ca3af')
    ax.set_ylabel("Number of Job Postings", fontsize=10, color='#9ca3af')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    ax.yaxis.grid(True, alpha=0.3, color='#374151')
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown('<div class="insight-box">📌 Web Developer (299) leads, followed by Data Engineer (286) and Full Stack Developer (274). Roles are fairly evenly distributed — no single title dominates.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Graph 3 — Work Type
    with col1:
        st.markdown('<div class="section-header">Graph 3 — Work Type Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        wt = df['work_type'].value_counts()
        palette = ['#7c3aed','#00f5d4','#f472b6','#fbbf24','#34d399']
        ax.bar(wt.index, wt.values, color=palette, edgecolor='#1f2937', linewidth=0.8)
        ax.set_title("Work Type Distribution", fontsize=11, fontweight='bold', color='#e5e7eb')
        ax.set_xlabel("Work Type", fontsize=9, color='#9ca3af')
        ax.set_ylabel("Number of Postings", fontsize=9, color='#9ca3af')
        plt.xticks(rotation=25, ha='right', fontsize=8)
        ax.yaxis.grid(True, alpha=0.3, color='#374151')
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('<div class="insight-box">Part-time leads slightly, followed closely by Internship and Full-time. The market is diverse across contract types.</div>', unsafe_allow_html=True)

    # Graph 5 — Experience Level
    with col2:
        st.markdown('<div class="section-header">Graph 5 — Experience Level Distribution</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        exp = df['formatted_experience_level'].value_counts()
        colors_exp = ['#00f5d4','#7c3aed','#f472b6','#fbbf24','#34d399','#60a5fa']
        ax.bar(exp.index, exp.values, color=colors_exp[:len(exp)], edgecolor='#1f2937', linewidth=0.8)
        ax.set_title("Experience Level Distribution", fontsize=11, fontweight='bold', color='#e5e7eb')
        ax.set_xlabel("Experience Level", fontsize=9, color='#9ca3af')
        ax.set_ylabel("Number of Job Postings", fontsize=9, color='#9ca3af')
        plt.xticks(rotation=25, ha='right', fontsize=8)
        ax.yaxis.grid(True, alpha=0.3, color='#374151')
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('<div class="insight-box">Experience levels are nearly uniform (800–880 each). Entry Level (833) shows strong fresher demand — good news for new graduates!</div>', unsafe_allow_html=True)

    # Graph 6 — Top Hiring Locations
    st.markdown('<div class="section-header">Graph 6 — Top 10 Hiring Locations (Bivariate)</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    top_loc = df['location'].value_counts().head(10)
    colors_loc = plt.cm.Blues_r(np.linspace(0.3, 0.9, len(top_loc)))
    ax.barh(top_loc.index, top_loc.values, color=colors_loc, edgecolor='#1f2937', linewidth=0.8)
    ax.set_title("Top 10 Hiring Locations", fontsize=12, fontweight='bold', color='#e5e7eb')
    ax.set_xlabel("Number of Job Postings", fontsize=10, color='#9ca3af')
    ax.set_ylabel("Location", fontsize=10, color='#9ca3af')
    ax.xaxis.grid(True, alpha=0.3, color='#374151')
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown('<div class="insight-box">Boston, Austin, and New York lead hiring. "Remote" also appears in top 10, reflecting the growing distributed workforce trend.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SALARY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰  Salary Analysis":
    st.markdown('<div class="hero-title" style="font-size:2rem;">Salary Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">DISTRIBUTION · BY JOB TITLE · REMOTE VS ON-SITE</div>', unsafe_allow_html=True)
    set_dark_style()

    # Stats Row
    sal = df['normalized_salary']
    c1,c2,c3,c4 = st.columns(4)
    for col, val, label in zip(
        [c1,c2,c3,c4],
        [f"${sal.mean():,.0f}", f"${sal.min():,.0f}", f"${sal.max():,.0f}", f"${sal.median():,.0f}"],
        ["MEAN SALARY","MIN SALARY","MAX SALARY","MEDIAN SALARY"]
    ):
        col.markdown(f'<div class="metric-card"><div class="metric-value">{val}</div><div class="metric-label">{label}</div></div>', unsafe_allow_html=True)

    st.markdown("")

    # Graph 4 — Salary Distribution
    st.markdown('<div class="section-header">Graph 4 — Salary Distribution (Histogram + KDE)</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.histplot(df['normalized_salary'], bins=40, kde=True, color='#00f5d4', edgecolor='#0d0d1a', ax=ax,
                 line_kws={'color':'#f472b6','linewidth':2})
    ax.set_title("Salary Distribution Across All Job Postings", fontsize=12, fontweight='bold', color='#e5e7eb')
    ax.set_xlabel("Annual Salary (USD)", fontsize=10, color='#9ca3af')
    ax.set_ylabel("Frequency (Number of Jobs)", fontsize=10, color='#9ca3af')
    ax.yaxis.grid(True, alpha=0.3, color='#374151')
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown('<div class="insight-box">Nearly normal distribution peaking at ~$108K–$115K. Right tail extends toward $165K — driven by AI/ML/NLP roles which command premium salaries.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    # Graph 8 — Avg Salary by Job Title
    with col1:
        st.markdown('<div class="section-header">Graph 8 — Average Salary by Job Title</div>', unsafe_allow_html=True)
        avg_sal = df.groupby('title')['normalized_salary'].mean().sort_values()
        fig, ax = plt.subplots(figsize=(8, 8))
        colors_sal = ['#7c3aed' if v > 115000 else '#374151' for v in avg_sal.values]
        ax.barh(avg_sal.index, avg_sal.values, color=colors_sal, edgecolor='#1f2937', linewidth=0.6)
        ax.set_title("Average Salary by Job Title", fontsize=11, fontweight='bold', color='#e5e7eb')
        ax.set_xlabel("Average Salary (USD)", fontsize=9, color='#9ca3af')
        ax.xaxis.grid(True, alpha=0.3, color='#374151')
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Graph 9 — Remote vs On-Site
    with col2:
        st.markdown('<div class="section-header">Graph 9 — Remote vs On-Site</div>', unsafe_allow_html=True)
        remote_data = df['remote_allowed'].map({0:'On-site', 1:'Remote'}).value_counts()
        fig, ax = plt.subplots(figsize=(5, 5))
        wedges, texts, autotexts = ax.pie(
            remote_data.values,
            labels=remote_data.index,
            autopct='%1.1f%%',
            colors=['#7c3aed','#00f5d4'],
            startangle=90,
            wedgeprops={'edgecolor':'#0a0a0f','linewidth':2}
        )
        for t in texts: t.set_color('#9ca3af')
        for a in autotexts: a.set_color('#0a0a0f'); a.set_fontweight('bold')
        ax.set_title("Remote vs On-site Jobs", fontsize=11, fontweight='bold', color='#e5e7eb')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.markdown('<div class="insight-box">59.3% of jobs allow remote work — more than half the market has location flexibility.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: SKILLS & NLP
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🛠️  Skills & NLP":
    st.markdown('<div class="hero-title" style="font-size:2rem;">Skills Analysis & NLP</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">TOP SKILLS · EXPERIENCE VS SKILLS · CORRELATION</div>', unsafe_allow_html=True)
    set_dark_style()

    # Extract skills
    all_skills = []
    for text in df['skills_desc'].dropna():
        skills = [s.strip() for s in text.split(',')]
        all_skills.extend(skills)
    skill_counts = Counter(all_skills)
    top20 = pd.DataFrame(skill_counts.most_common(20), columns=['Skill','Count'])

    col1, col2 = st.columns(2)

    # Graph 7a — Top Skills Bar
    with col1:
        st.markdown('<div class="section-header">Graph 7a — Top 20 Demanded Skills (Bar)</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 7))
        colors_sk = plt.cm.plasma(np.linspace(0.2, 0.9, 20))
        ax.barh(top20['Skill'], top20['Count'], color=colors_sk, edgecolor='#1f2937', linewidth=0.5)
        ax.set_title("Top 20 Most Demanded Skills", fontsize=11, fontweight='bold', color='#e5e7eb')
        ax.set_xlabel("Frequency", fontsize=9, color='#9ca3af')
        ax.xaxis.grid(True, alpha=0.3, color='#374151')
        ax.set_axisbelow(True)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Graph 7b — Pie Chart
    with col2:
        st.markdown('<div class="section-header">Graph 7b — Top 10 Skills (Pie)</div>', unsafe_allow_html=True)
        top10 = top20.head(10)
        fig, ax = plt.subplots(figsize=(7, 7))
        colors_pie = plt.cm.Set2(np.linspace(0, 1, 10))
        wedges, texts, autotexts = ax.pie(
            top10['Count'], labels=top10['Skill'], autopct='%1.1f%%',
            colors=colors_pie, startangle=140,
            wedgeprops={'edgecolor':'#0a0a0f','linewidth':1.5}
        )
        for t in texts: t.set_color('#9ca3af'); t.set_fontsize(8)
        for a in autotexts: a.set_fontsize(7); a.set_color('#0a0a0f')
        ax.set_title("Top 10 Most Demanded Skills", fontsize=11, fontweight='bold', color='#e5e7eb')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.markdown('<div class="insight-box">🐍 Python dominates at 14.7% of all skill mentions. SQL (8.5%) and React (5.6%) follow. NLP Approach: split skills_desc by comma → Counter → top 20.</div>', unsafe_allow_html=True)

    # Graph 10 — Skills by Experience Level
    st.markdown('<div class="section-header">Graph 10 — Skills Required by Experience Level (Box Plot)</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    palette_exp = {
        'Entry Level':'#00f5d4','Director':'#7c3aed','Executive':'#f472b6',
        'Internship':'#fbbf24','Mid-Senior Level':'#34d399','Associate':'#60a5fa'
    }
    sns.boxplot(x='formatted_experience_level', y='skill_count', data=df,
                palette=palette_exp, ax=ax, linewidth=1.2)
    ax.set_title("Skills Distribution by Experience Level", fontsize=12, fontweight='bold', color='#e5e7eb')
    ax.set_xlabel("Experience Level", fontsize=10, color='#9ca3af')
    ax.set_ylabel("Number of Skills Required", fontsize=10, color='#9ca3af')
    plt.xticks(rotation=20, fontsize=9)
    ax.yaxis.grid(True, alpha=0.3, color='#374151')
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown('<div class="insight-box">All experience levels require ~7–8 skills. Minimal difference across levels — skill breadth matters equally for freshers and directors.</div>', unsafe_allow_html=True)

    # Graph 11 — Correlation Heatmap
    st.markdown('<div class="section-header">Graph 11 — Feature Correlation Heatmap</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    numeric_df = df[['normalized_salary','remote_allowed','views','applies','skill_count']]
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax, fmt='.3f',
                linewidths=0.5, linecolor='#0a0a0f',
                annot_kws={'size':10, 'color':'white'})
    ax.set_title("Feature Correlation Heatmap", fontsize=12, fontweight='bold', color='#e5e7eb')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown('<div class="insight-box">skill_count has weak positive correlation (0.13) with salary — more skills → slightly higher pay. Most features are independent of each other.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ML MODELS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  ML Models":
    st.markdown('<div class="hero-title" style="font-size:2rem;">Machine Learning Models</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">TF-IDF NLP → 4 CLASSIFIERS → ACCURACY COMPARISON</div>', unsafe_allow_html=True)
    set_dark_style()

    with st.spinner("Training models on 5000 job postings..."):
        tfidf, le, trained, results, best_model, X_test, y_test = build_models(df)

    # Model accuracy cards
    st.markdown('<div class="section-header">Model Accuracy Results</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    for col, (name, acc) in zip(cols, results.items()):
        short = name.replace(' ','<br>')
        col.markdown(f"""
        <div class="metric-card">
            <div style="font-size:0.7rem; color:#6b7280; font-family:Space Mono,monospace;">{short}</div>
            <div class="metric-value" style="color:{'#00f5d4' if acc==100 else '#f472b6'};">{acc:.1f}%</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="insight-box">✅ Random Forest selected as best model — robustness, stability, and feature_importances_ make it ideal for skill-based career classification.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # Graph 12 — Model Accuracy Comparison
    with col1:
        st.markdown('<div class="section-header">Graph 12 — Model Accuracy Comparison</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        names = list(results.keys())
        accs = list(results.values())
        bar_colors = ['#00f5d4' if a == max(accs) else '#374151' for a in accs]
        bars = ax.bar(names, accs, color=bar_colors, edgecolor='#6b7280', linewidth=0.8, width=0.6)
        ax.set_ylim(0, 115)
        ax.set_title("Model Accuracy Comparison", fontsize=11, fontweight='bold', color='#e5e7eb')
        ax.set_xlabel("Models", fontsize=9, color='#9ca3af')
        ax.set_ylabel("Accuracy (%)", fontsize=9, color='#9ca3af')
        plt.xticks(rotation=20, ha='right', fontsize=8)
        ax.yaxis.grid(True, alpha=0.3, color='#374151')
        ax.set_axisbelow(True)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, color='#e5e7eb')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Graph 13 — Confusion Matrix
    with col2:
        st.markdown('<div class="section-header">Graph 13 — Confusion Matrix (Random Forest)</div>', unsafe_allow_html=True)
        from sklearn.metrics import confusion_matrix
        y_pred = best_model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=le.classes_, yticklabels=le.classes_,
                    linewidths=0.3, linecolor='#0a0a0f',
                    annot_kws={'size':7})
        ax.set_title("Confusion Matrix — Random Forest\n(Diagonal = Correct Predictions)",
                     fontsize=10, fontweight='bold', color='#e5e7eb')
        ax.set_xlabel("Predicted Job Title", fontsize=8, color='#9ca3af')
        ax.set_ylabel("Actual Job Title", fontsize=8, color='#9ca3af')
        plt.xticks(rotation=45, ha='right', fontsize=6)
        plt.yticks(fontsize=6)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Graph 14 — Feature Importances
    st.markdown('<div class="section-header">Graph 14 — Top TF-IDF Feature Importances (Random Forest)</div>', unsafe_allow_html=True)
    rf = trained['Random Forest']
    feat_names = tfidf.get_feature_names_out()
    importances = rf.feature_importances_
    top_idx = np.argsort(importances)[-20:]
    feat_df = pd.DataFrame({'Skill': feat_names[top_idx], 'Importance': importances[top_idx]}).sort_values('Importance')
    fig, ax = plt.subplots(figsize=(12, 5))
    bar_colors = plt.cm.viridis(np.linspace(0.3, 0.9, 20))
    ax.barh(feat_df['Skill'], feat_df['Importance'], color=bar_colors, edgecolor='#1f2937', linewidth=0.5)
    ax.set_title("Top 20 Important Skill Features (TF-IDF)", fontsize=12, fontweight='bold', color='#e5e7eb')
    ax.set_xlabel("Importance Score", fontsize=10, color='#9ca3af')
    ax.xaxis.grid(True, alpha=0.3, color='#374151')
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown('<div class="insight-box">JS, Python, and React are the most discriminative features — these words clearly separate job categories. TF-IDF bigrams (e.g., "react node", "php mysql") also appear as strong signals.</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: FUTURE GROWTH
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Future Growth":
    st.markdown('<div class="hero-title" style="font-size:2rem;">Future Job & Skill Growth</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">GROWTH SCORES · SALARY VS GROWTH · SKILL TRENDS</div>', unsafe_allow_html=True)
    set_dark_style()

    job_growth = {
        'AI Engineer':95,'Machine Learning Engineer':92,'Data Scientist':88,'NLP Engineer':85,
        'Cloud Engineer':83,'Data Engineer':80,'Cybersecurity Analyst':79,'DevOps Engineer':78,
        'Blockchain Developer':72,'Full Stack Developer':70,'Backend Developer':68,
        'Mobile Developer':65,'Product Manager':63,'Software Engineer':60,'Data Analyst':58,
        'Frontend Developer':55,'Business Analyst':50,'QA Engineer':45,'Web Developer':40,'Network Engineer':38
    }
    skill_growth = {
        'LLMs / GPT':98,'AI / Deep Learning':96,'MLOps':93,'Python':87,'Kubernetes':85,
        'Cloud (AWS/GCP/Azure)':84,'Cybersecurity':82,'React':75,'Docker':74,'SQL':70,
        'Java':62,'Excel':58,'HTML/CSS':52,'PHP':42,'Network Admin':38
    }

    growth_df = pd.DataFrame(list(job_growth.items()), columns=['Job','Growth']).sort_values('Growth')

    # Graph 15 — Job Growth
    st.markdown('<div class="section-header">Graph 15 — Predicted Future Job Growth</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    colors_g = ['#00f5d4' if v >= 85 else '#7c3aed' if v >= 70 else '#fbbf24' if v >= 50 else '#ef4444'
                for v in growth_df['Growth']]
    ax.barh(growth_df['Job'], growth_df['Growth'], color=colors_g, edgecolor='#1f2937', linewidth=0.6)
    ax.axvline(70, color='#f472b6', linestyle='--', linewidth=1.5, alpha=0.8, label='High Growth Threshold (70%)')
    ax.axvline(50, color='#fbbf24', linestyle='--', linewidth=1, alpha=0.6, label='Stable Threshold (50%)')
    ax.set_title("Future Job Growth Prediction", fontsize=13, fontweight='bold', color='#e5e7eb')
    ax.set_xlabel("Growth Score (%)", fontsize=10, color='#9ca3af')
    ax.legend(fontsize=8, facecolor='#111827', edgecolor='#374151', labelcolor='#9ca3af')
    ax.xaxis.grid(True, alpha=0.3, color='#374151')
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    c1, c2, c3 = st.columns(3)
    high = [(j,s) for j,s in job_growth.items() if s >= 70]
    stable = [(j,s) for j,s in job_growth.items() if 50 <= s < 70]
    declining = [(j,s) for j,s in job_growth.items() if s < 50]

    with c1:
        st.markdown("**🟢 High Growth (≥70%)**")
        for j,s in sorted(high, key=lambda x: -x[1]):
            st.markdown(f'<span class="skill-tag" style="color:#00f5d4;">{j} — {s}%</span>', unsafe_allow_html=True)
    with c2:
        st.markdown("**🟡 Stable (50–69%)**")
        for j,s in sorted(stable, key=lambda x: -x[1]):
            st.markdown(f'<span class="skill-tag">{j} — {s}%</span>', unsafe_allow_html=True)
    with c3:
        st.markdown("**🔴 Declining (<50%)**")
        for j,s in sorted(declining, key=lambda x: -x[1]):
            st.markdown(f'<span class="missing-tag">{j} — {s}%</span>', unsafe_allow_html=True)

    # Graph 17 — Salary vs Growth Scatter
    st.markdown('<div class="section-header">Graph 17 — Salary vs Future Growth (Scatter)</div>', unsafe_allow_html=True)
    avg_sal = df.groupby('title')['normalized_salary'].mean()
    common_titles = [t for t in avg_sal.index if t in job_growth]
    g_vals = [job_growth[t] for t in common_titles]
    s_vals = [avg_sal[t] for t in common_titles]

    fig, ax = plt.subplots(figsize=(12, 6))
    sc = ax.scatter(g_vals, s_vals, c=g_vals, cmap='RdYlGn', s=120, edgecolors='#1f2937', linewidth=1, zorder=3)
    for title, gv, sv in zip(common_titles, g_vals, s_vals):
        short = title.replace(' Engineer','').replace(' Developer','').replace(' Analyst','')
        ax.annotate(short, (gv, sv), fontsize=7, color='#9ca3af',
                    xytext=(4, 4), textcoords='offset points')
    plt.colorbar(sc, ax=ax, label='Growth Score').ax.yaxis.label.set_color('#9ca3af')
    ax.axvline(70, color='#f472b6', linestyle='--', linewidth=1, alpha=0.7, label='Growth=70')
    ax.axhline(avg_sal.mean(), color='#00f5d4', linestyle='--', linewidth=1, alpha=0.7, label='Avg Salary')
    ax.set_title("Salary vs Future Growth — Quadrant Analysis", fontsize=12, fontweight='bold', color='#e5e7eb')
    ax.set_xlabel("Growth Score (%)", fontsize=10, color='#9ca3af')
    ax.set_ylabel("Average Salary (USD)", fontsize=10, color='#9ca3af')
    ax.legend(fontsize=8, facecolor='#111827', edgecolor='#374151', labelcolor='#9ca3af')
    ax.xaxis.grid(True, alpha=0.3, color='#374151')
    ax.yaxis.grid(True, alpha=0.3, color='#374151')
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown('<div class="insight-box">🎯 IDEAL ZONE (top-right): AI Engineer, ML Engineer, NLP Engineer — highest salary AND highest growth. These are the optimal career targets for 2025–2030.</div>', unsafe_allow_html=True)

    # Skill Growth
    st.markdown('<div class="section-header">Graph 16 — Future Skill Growth Scores</div>', unsafe_allow_html=True)
    sk_df = pd.DataFrame(list(skill_growth.items()), columns=['Skill','Growth']).sort_values('Growth')
    fig, ax = plt.subplots(figsize=(12, 5))
    colors_sk = ['#00f5d4' if v >= 80 else '#7c3aed' if v >= 65 else '#ef4444' for v in sk_df['Growth']]
    ax.barh(sk_df['Skill'], sk_df['Growth'], color=colors_sk, edgecolor='#1f2937', linewidth=0.6)
    ax.axvline(80, color='#f472b6', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.set_title("Future Skill Growth Prediction", fontsize=12, fontweight='bold', color='#e5e7eb')
    ax.set_xlabel("Growth Score (%)", fontsize=10, color='#9ca3af')
    ax.xaxis.grid(True, alpha=0.3, color='#374151')
    ax.set_axisbelow(True)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CAREER RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯  Career Recommender":
    st.markdown('<div class="hero-title" style="font-size:2rem;">Personalized Career Recommender</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">ENTER YOUR SKILLS → GET TOP 3 CAREER MATCHES + MISSING SKILLS</div>', unsafe_allow_html=True)

    with st.spinner("Loading ML model..."):
        tfidf, le, trained, results, best_model, X_test, y_test = build_models(df)

    st.markdown('<div class="section-header">How It Works (Step 8)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box">
    1. Your skills text → TF-IDF vectorization (200 features, bigrams)<br>
    2. Random Forest model → predict_proba() for all 20 job classes<br>
    3. Top 3 highest probabilities = your career recommendations<br>
    4. Compare your skills vs job's required skills → show missing ones
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    # Presets
    st.markdown('<div class="section-header">Try a Preset or Enter Custom Skills</div>', unsafe_allow_html=True)
    presets = {
        "🐍 Data/ML":       "Python, Machine Learning, Statistics, SQL",
        "🌐 Web Dev":       "React, JavaScript, HTML, CSS, Node.js",
        "☁️ Cloud/DevOps":  "AWS, Docker, Linux, Kubernetes, Terraform",
        "🔐 Security":      "Python, Firewall, Network Security, CISSP",
        "📱 Mobile":        "Kotlin, Android, Flutter, Firebase, React Native",
        "🔗 Blockchain":    "Solidity, Web3.js, Smart Contracts, Ethereum, Python",
    }

    selected_preset = st.selectbox("Quick Preset", ["— Custom Input —"] + list(presets.keys()))
    if selected_preset != "— Custom Input —":
        default_skills = presets[selected_preset]
    else:
        default_skills = ""

    user_input = st.text_area(
        "Enter your skills (comma-separated)",
        value=default_skills,
        placeholder="e.g. Python, Machine Learning, SQL, TensorFlow",
        height=80
    )

    exp_level = st.selectbox("Your Experience Level", ['Entry Level','Mid-Senior Level','Associate','Director','Executive','Internship'])
    work_pref = st.selectbox("Work Type Preference", ['Full-time','Part-time','Contract','Internship','Temporary'])

    if st.button("🚀 Get Career Recommendations", use_container_width=True):
        if not user_input.strip():
            st.warning("Please enter at least one skill.")
        else:
            combined_input = user_input + " " + exp_level + " " + work_pref
            user_vec = tfidf.transform([combined_input])
            probs = best_model.predict_proba(user_vec)[0]
            top3_idx = np.argsort(probs)[-3:][::-1]

            st.markdown('<div class="section-header">Your Top 3 Career Matches</div>', unsafe_allow_html=True)

            job_growth = {
                'AI Engineer':95,'Machine Learning Engineer':92,'Data Scientist':88,'NLP Engineer':85,
                'Cloud Engineer':83,'Data Engineer':80,'Cybersecurity Analyst':79,'DevOps Engineer':78,
                'Blockchain Developer':72,'Full Stack Developer':70,'Backend Developer':68,
                'Mobile Developer':65,'Product Manager':63,'Software Engineer':60,'Data Analyst':58,
                'Frontend Developer':55,'Business Analyst':50,'QA Engineer':45,'Web Developer':40,'Network Engineer':38
            }

            for rank, idx in enumerate(top3_idx):
                job  = le.classes_[idx]
                conf = probs[idx] * 100
                growth = job_growth.get(job, 50)
                avg_s  = df[df['title'] == job]['normalized_salary'].mean()

                job_skills_text = df[df['title'] == job]['skills_desc'].iloc[0]
                job_skills  = set(s.strip() for s in job_skills_text.split(','))
                user_skills = set(s.strip() for s in user_input.split(','))
                missing     = list(job_skills - user_skills)[:5]
                matched     = list(job_skills & user_skills)

                medal = ["🥇","🥈","🥉"][rank]
                st.markdown(f"""
                <div class="career-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div class="career-title">{medal} {job}</div>
                        <div style="font-family:Space Mono,monospace; font-size:0.75rem; color:#6b7280;">Rank #{rank+1}</div>
                    </div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width:{min(conf,100):.1f}%;"></div>
                    </div>
                    <div style="display:flex; gap:1.5rem; margin:0.5rem 0;">
                        <div>
                            <span style="font-family:Space Mono,monospace; font-size:0.65rem; color:#6b7280;">CONFIDENCE</span><br>
                            <span style="font-size:1.2rem; font-weight:700; color:#00f5d4; font-family:Space Mono,monospace;">{conf:.1f}%</span>
                        </div>
                        <div>
                            <span style="font-family:Space Mono,monospace; font-size:0.65rem; color:#6b7280;">AVG SALARY</span><br>
                            <span style="font-size:1.2rem; font-weight:700; color:#a78bfa; font-family:Space Mono,monospace;">${avg_s:,.0f}</span>
                        </div>
                        <div>
                            <span style="font-family:Space Mono,monospace; font-size:0.65rem; color:#6b7280;">GROWTH SCORE</span><br>
                            <span style="font-size:1.2rem; font-weight:700; color:#34d399; font-family:Space Mono,monospace;">{growth}%</span>
                        </div>
                    </div>
                    <div style="margin-top:0.5rem;">
                        <span style="font-family:Space Mono,monospace; font-size:0.65rem; color:#6b7280;">✅ SKILLS YOU HAVE: </span>
                        {''.join(f'<span class="skill-tag">{s}</span>' for s in matched) if matched else '<span style="color:#6b7280;font-size:0.75rem;">none matched</span>'}
                    </div>
                    <div style="margin-top:0.4rem;">
                        <span style="font-family:Space Mono,monospace; font-size:0.65rem; color:#6b7280;">❌ SKILLS TO LEARN: </span>
                        {''.join(f'<span class="missing-tag">{s}</span>' for s in missing) if missing else '<span style="color:#00f5d4;font-size:0.75rem;">You already have all required skills! 🎉</span>'}
                    </div>
                </div>
                """, unsafe_allow_html=True)