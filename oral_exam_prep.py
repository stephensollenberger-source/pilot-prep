import json
import os
import traceback
from datetime import datetime
from pathlib import Path

import anthropic
import certifi
import httpx
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────────

SESSIONS_FILE = Path("pilot_prep_sessions.json")

SUBJECTS = {
    "Private Pilot": [
        "Regulations (FARs)",
        "Airspace",
        "Weather",
        "Aircraft Systems",
        "Aerodynamics",
        "Navigation",
        "Airport Operations",
        "Cross-Country Planning",
        "Emergency Procedures",
    ],
    "Instrument Rating": [
        "IFR Regulations",
        "Weather & Meteorology",
        "Instrument Approaches",
        "Navigation & GPS",
        "Holding Patterns",
        "IFR Charts & Enroute",
        "Pitot-Static & Gyro Systems",
        "Emergency Procedures",
    ],
    "Commercial": [
        "Commercial Regulations",
        "Advanced Aerodynamics",
        "Aircraft Systems",
        "Navigation",
        "Commercial Flight Operations",
        "Night Operations",
        "High-Performance Aircraft",
        "Weather",
        "Emergency Procedures",
    ],
    "ATP": [
        "Part 121/135 Regulations",
        "CRM & SRM",
        "Turbine Systems",
        "High Altitude Operations",
        "Advanced Weather",
        "Performance & Limitations",
        "Weight & Balance",
        "International Operations",
        "Advanced Navigation",
        "Emergency Procedures",
    ],
}

CERT_STANDARDS = {
    "Private Pilot": "Private Pilot ACS (FAA-S-ACS-6)",
    "Instrument Rating": "Instrument Rating ACS (FAA-S-ACS-8)",
    "Commercial": "Commercial Pilot ACS (FAA-S-ACS-7)",
    "ATP": "Airline Transport Pilot ACS (FAA-S-ACS-11)",
}

EVALUATION_SCHEMA = {
    "type": "object",
    "properties": {
        "rating": {
            "type": "string",
            "enum": ["Pass", "Needs Work", "Unsatisfactory"],
        },
        "feedback": {"type": "string"},
        "key_points_covered": {"type": "array", "items": {"type": "string"}},
        "key_points_missed": {"type": "array", "items": {"type": "string"}},
        "examiner_note": {"type": "string"},
    },
    "required": ["rating", "feedback", "key_points_covered", "key_points_missed", "examiner_note"],
    "additionalProperties": False,
}

RATING_EMOJI = {"Pass": "🟢", "Needs Work": "🟡", "Unsatisfactory": "🔴"}
RATING_POINTS = {"Pass": 2, "Needs Work": 1, "Unsatisfactory": 0}

# ── Persistence ────────────────────────────────────────────────────────────────

def load_sessions() -> list:
    if not SESSIONS_FILE.exists():
        return []
    try:
        return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def save_session(cert: str, subject: str, num_questions: int, questions: list) -> None:
    sessions = load_sessions()
    sessions.append({
        "timestamp": datetime.now().isoformat(),
        "cert": cert,
        "subject": subject,
        "num_questions": num_questions,
        "questions": questions,
    })
    SESSIONS_FILE.write_text(json.dumps(sessions, indent=2), encoding="utf-8")


def readiness_score() -> dict:
    sessions = load_sessions()
    total_q = sum(len(s["questions"]) for s in sessions)
    if not sessions or total_q == 0:
        return {"score": 0, "total_q": 0, "total_s": 0}
    total_pts = sum(
        RATING_POINTS.get(q["rating"], 0)
        for s in sessions for q in s["questions"]
    )
    score = round((total_pts / (total_q * 2)) * 100)
    return {"score": score, "total_q": total_q, "total_s": len(sessions)}


def weak_areas() -> list:
    """Returns list of {subject, pass_rate, total} sorted worst-first, filtered < 70%."""
    stats: dict[str, list] = {}
    for s in load_sessions():
        subj = s["subject"]
        if subj not in stats:
            stats[subj] = [0, 0]  # [passes, total]
        for q in s["questions"]:
            stats[subj][1] += 1
            if q["rating"] == "Pass":
                stats[subj][0] += 1
    result = [
        {"subject": subj, "pass_rate": v[0] / v[1], "total": v[1]}
        for subj, v in stats.items() if v[1] > 0
    ]
    return sorted(
        [r for r in result if r["pass_rate"] < 0.70],
        key=lambda x: x["pass_rate"],
    )


# ── Claude API ─────────────────────────────────────────────────────────────────

def _client() -> anthropic.Anthropic:
    return anthropic.Anthropic(http_client=httpx.Client(verify=certifi.where()))


def _system(cert: str, subject: str) -> list:
    return [{
        "type": "text",
        "text": (
            f"You are an FAA Designated Pilot Examiner (DPE) conducting a realistic oral "
            f"examination for a student seeking their {cert} certificate. "
            f"You are currently testing the subject area: {subject}.\n\n"
            f"Rules:\n"
            f"- Ask one question at a time and wait for the student to respond\n"
            f"- After each response, ask exactly one follow-up — do not comment on whether "
            f"the student's answer was correct or incorrect before the follow-up is answered\n"
            f"- Use precise FAA terminology; reference specific FARs, AIM chapters, and "
            f"ACS task codes when relevant\n"
            f"- Hold the student to {CERT_STANDARDS[cert]} standards\n"
            f"- Be rigorous but conversational — match the tone of a real checkride\n\n"
            f"Never break character. You are the examiner."
        ),
        "cache_control": {"type": "ephemeral"},
    }]


def generate_question(cert: str, subject: str, previous: list) -> str:
    avoid = (
        "\n\nDo not repeat any of these previously asked questions:\n"
        + "\n".join(f"- {q}" for q in previous)
        if previous else ""
    )
    resp = _client().messages.create(
        model="claude-opus-4-7",
        max_tokens=256,
        system=_system(cert, subject),
        messages=[{
            "role": "user",
            "content": (
                f"Begin the oral exam. Ask me one question about {subject}."
                f"{avoid}\n\nOutput only the question — no preamble or label."
            ),
        }],
    )
    return resp.content[0].text.strip()


def generate_followup(cert: str, subject: str, question: str, answer: str) -> str:
    resp = _client().messages.create(
        model="claude-opus-4-7",
        max_tokens=256,
        system=_system(cert, subject),
        messages=[
            {"role": "user", "content": f"Begin the oral exam. Ask me about {subject}."},
            {"role": "assistant", "content": question},
            {"role": "user", "content": answer},
        ],
    )
    return resp.content[0].text.strip()


def evaluate(
    cert: str, subject: str,
    question: str, answer: str,
    followup: str, followup_answer: str,
) -> dict:
    resp = _client().messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=_system(cert, subject),
        output_config={"format": {"type": "json_schema", "schema": EVALUATION_SCHEMA}},
        messages=[{
            "role": "user",
            "content": (
                f"Evaluate this student's complete performance.\n\n"
                f"**Question:** {question}\n**Answer:** {answer}\n\n"
                f"**Follow-up:** {followup}\n**Follow-up answer:** {followup_answer}\n\n"
                f"Rate as Pass, Needs Work, or Unsatisfactory. Be honest and specific."
            ),
        }],
    )
    return json.loads(resp.content[0].text)


# ── HTML components ────────────────────────────────────────────────────────────

def _card(text: str, label: str, border: str, bg: str, label_color: str):
    st.markdown(
        f'<div style="background:{bg}; border-left:4px solid {border}; '
        f'border-radius:0 6px 6px 0; padding:14px 18px; margin:6px 0 14px 0;">'
        f'<div style="color:{label_color}; font-size:0.7em; font-weight:700; '
        f'letter-spacing:0.12em; text-transform:uppercase; margin-bottom:8px;">{label}</div>'
        f'<div style="color:#e6edf3; line-height:1.7;">{text}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def examiner_card(text: str):
    _card(text, "Examiner", "#1f6feb", "#0d2137", "#58a6ff")


def followup_card(text: str):
    _card(text, "Follow-up", "#8957e5", "#150d2a", "#a371f7")


def student_card(text: str, label: str = "Your Answer"):
    st.markdown(
        f'<div style="background:#161b22; border:1px solid #30363d; '
        f'border-radius:6px; padding:12px 16px; margin:4px 0 12px 0;">'
        f'<div style="color:#8b949e; font-size:0.7em; font-weight:700; '
        f'letter-spacing:0.12em; text-transform:uppercase; margin-bottom:6px;">{label}</div>'
        f'<div style="color:#8b949e; line-height:1.6; font-size:0.93em;">{text}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def rating_badge(rating: str):
    configs = {
        "Pass":            ("#0f2a1a", "#3fb950", "#3fb950", "✓  Pass"),
        "Needs Work":      ("#2d1f00", "#d29922", "#d29922", "△  Needs Work"),
        "Unsatisfactory":  ("#2d0d0d", "#f85149", "#f85149", "✗  Unsatisfactory"),
    }
    bg, border, fg, label = configs.get(rating, ("#161b22", "#30363d", "#8b949e", rating))
    st.markdown(
        f'<div style="display:inline-flex; align-items:center; '
        f'background:{bg}; border:1px solid {border}; border-radius:6px; '
        f'padding:8px 18px; margin:8px 0 16px 0; '
        f'color:{fg}; font-weight:700; font-size:1em; letter-spacing:0.03em;">'
        f'{label}</div>',
        unsafe_allow_html=True,
    )


def readiness_display(score: int, total_q: int, total_s: int):
    color = "#3fb950" if score >= 75 else "#d29922" if score >= 50 else "#f85149"
    subtitle = f"{total_q} questions · {total_s} {'session' if total_s == 1 else 'sessions'}"
    st.markdown(
        f'<div style="text-align:center; background:#161b22; border:1px solid #30363d; '
        f'border-radius:8px; padding:32px 20px 24px 20px; margin-bottom:4px;">'
        f'<div style="color:#8b949e; font-size:0.72em; font-weight:700; '
        f'letter-spacing:0.15em; text-transform:uppercase; margin-bottom:12px;">Readiness Score</div>'
        f'<div style="font-size:5em; font-weight:800; color:{color}; line-height:1; '
        f'font-variant-numeric:tabular-nums;">{score}</div>'
        f'<div style="color:#30363d; font-size:0.85em; margin-top:6px;">out of 100</div>'
        f'<div style="color:#484f58; font-size:0.78em; margin-top:4px;">{subtitle}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── CSS ────────────────────────────────────────────────────────────────────────

CSS = """<style>
/* ── Base ──────────────────────────────────────────────────────── */
.stApp,
[data-testid="stApp"],
[data-testid="stAppViewContainer"] { background-color: #0d1117 !important; }

/* ── App header — Streamlit 1.50 uses emotion CSS injected into <head>
     which wins the cascade at equal specificity, so we need !important
     on every selector that could match it. In v1.50 the element is
     <header class="stAppHeader" data-testid="stHeader"> — there is no
     stDecoration testid; it was removed.                               */
[data-testid="stHeader"],
[data-testid="stAppHeader"],
header[data-testid="stHeader"],
header.stAppHeader,
.stAppHeader,
[data-testid="stToolbar"] {
    background: #0d1117 !important;
    background-color: #0d1117 !important;
    border-bottom: 1px solid #21262d !important;
    box-shadow: none !important;
}
/* Belt-and-suspenders for older Streamlit that had stDecoration */
[data-testid="stDecoration"] {
    display: none !important;
    height: 0 !important;
    overflow: hidden !important;
}

/* ── Sidebar ────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: #161b22 !important;
    border-right: 1px solid #21262d !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label { color: #8b949e !important; }

/* ── Typography ─────────────────────────────────────────────────── */
h1 { color: #f0f6fc !important; font-weight: 800 !important; letter-spacing: -0.02em !important; }
h2 { color: #e6edf3 !important; font-weight: 700 !important; }
h3 { color: #cdd9e5 !important; font-weight: 600 !important; }
p, li { color: #cdd9e5; }
.stMarkdown p { color: #cdd9e5; }
.stCaption p { color: #484f58 !important; font-size: 0.8em !important; }

/* ── Tabs ───────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #21262d;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    color: #484f58;
    font-weight: 600;
    font-size: 0.88em;
    padding: 8px 20px;
    background: transparent;
}
.stTabs [data-baseweb="tab"]:hover { color: #8b949e; background: transparent; }
.stTabs [aria-selected="true"] {
    color: #58a6ff !important;
    border-bottom: 2px solid #1f6feb;
    background: transparent;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 20px; }

/* ── Inputs ─────────────────────────────────────────────────────── */
.stTextArea textarea {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    color: #e6edf3 !important;
    font-size: 0.95em !important;
    line-height: 1.65 !important;
    caret-color: #58a6ff;
}
.stTextArea textarea:focus {
    border-color: #1f6feb !important;
    box-shadow: 0 0 0 3px rgba(31, 111, 235, 0.12) !important;
}
.stTextArea textarea::placeholder { color: #30363d !important; }
.stTextArea label {
    color: #484f58 !important;
    font-size: 0.78em !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

/* ── Selectbox ──────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
    color: #e6edf3 !important;
}
[data-testid="stSelectbox"] label {
    color: #484f58 !important;
    font-size: 0.78em !important;
    font-weight: 700 !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}

/* ── Buttons ────────────────────────────────────────────────────── */
.stButton > button {
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.88em !important;
    border: 1px solid #30363d !important;
    background-color: #21262d !important;
    color: #cdd9e5 !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover:not(:disabled) {
    border-color: #8b949e !important;
    background-color: #30363d !important;
}
[data-testid="stBaseButton-primary"],
.stButton > button[kind="primary"] {
    background: #1f6feb !important;
    border-color: #1f6feb !important;
    color: #fff !important;
}
[data-testid="stBaseButton-primary"]:hover:not(:disabled),
.stButton > button[kind="primary"]:hover:not(:disabled) {
    background: #388bfd !important;
    border-color: #388bfd !important;
    box-shadow: 0 0 0 3px rgba(31, 111, 235, 0.3) !important;
}
.stButton > button:disabled { opacity: 0.35 !important; }

/* ── Progress ───────────────────────────────────────────────────── */
/* stProgress renders: <div.stProgress> → <div> label → <div> ProgressBar
   The old ".stProgress > div { height:4px }" collapsed the text label.
   Target only the bar track (last child) and leave the label alone.  */
.stProgress,
[data-testid="stProgress"] { background: transparent !important; }

/* Text label — restore natural height, transparent background */
[data-testid="stProgress"] > div:first-child {
    background: transparent !important;
    height: auto !important;
}

/* Bar track (last div child of stProgress) */
[data-testid="stProgress"] > div:last-child {
    background-color: #21262d !important;
    border-radius: 999px !important;
    height: 4px !important;
    overflow: hidden !important;
}
[data-testid="stProgress"] > div:last-child > div,
[data-testid="stProgress"] > div:last-child > div > div {
    background: #1f6feb !important;
    height: 4px !important;
    border-radius: 999px !important;
}
/* Fallback for stProgressBar testid (older Streamlit) */
[data-testid="stProgressBar"] {
    background-color: #21262d !important;
    border-radius: 999px !important;
    height: 4px !important;
}

/* ── Metrics ────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    padding: 20px !important;
}
[data-testid="stMetricValue"] { color: #f0f6fc !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] {
    color: #484f58 !important;
    font-size: 0.75em !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* ── Expanders ──────────────────────────────────────────────────── */
details[data-testid="stExpander"] {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
}
details[data-testid="stExpander"] summary { color: #8b949e !important; font-weight: 600 !important; }
details[data-testid="stExpander"] summary:hover { color: #cdd9e5 !important; }

/* ── Divider ────────────────────────────────────────────────────── */
hr { border-color: #21262d !important; margin: 20px 0 !important; }

/* ── Alerts ─────────────────────────────────────────────────────── */
[data-testid="stAlert"] { border-radius: 6px !important; }

/* ── Hide Streamlit decoration bar ──────────────────────────────── */
[data-testid="stDecoration"] { display: none !important; }

/* ── Transparent structural containers ──────────────────────────── */
[data-testid="stMain"],
[data-testid="block-container"],
.block-container,
.stVerticalBlock,
.element-container { background: transparent !important; }

/* ── Hero section ───────────────────────────────────────────────── */
.hero-wrap {
    position: relative; overflow: hidden; border-radius: 12px;
    background: linear-gradient(160deg, #020810 0%, #0d1e35 55%, #091828 100%);
    margin-bottom: 32px; min-height: 210px; display: flex; align-items: center;
}
.hero-text { position: relative; z-index: 3; padding: 36px 28px 36px 32px; max-width: 54%; }
.hero-label {
    color: #1f6feb; font-size: 0.62em; font-weight: 800;
    letter-spacing: 0.2em; text-transform: uppercase; margin-bottom: 10px;
}
h1.hero-title {
    color: #f0f6fc !important; font-size: 2.3em !important; font-weight: 900 !important;
    letter-spacing: -0.02em !important; line-height: 1.1 !important;
    margin: 0 0 14px 0 !important; padding: 0 !important;
}
p.hero-sub {
    color: #8b949e !important; font-size: 0.85em !important;
    line-height: 1.6 !important; margin: 0 !important;
}
.hero-art {
    position: absolute; right: -15px; bottom: 0; width: 58%;
    z-index: 2; pointer-events: none;
}
.hero-art svg { width: 100%; height: auto; display: block; }
.hero-star {
    position: absolute; border-radius: 50%; background: white; z-index: 1;
    animation: twinkle var(--dur,3s) ease-in-out var(--dly,0s) infinite;
    pointer-events: none;
}
@keyframes twinkle {
    0%, 100% { opacity: 0.1; transform: scale(0.8); }
    50%       { opacity: 0.9; transform: scale(1.2); }
}
</style>"""


# ── Hero HTML ──────────────────────────────────────────────────────────────────

HERO_HTML = """<div class="hero-wrap">
<div class="hero-star" style="width:2px;height:2px;top:10%;left:4%;--dur:3.1s;--dly:0s;"></div>
<div class="hero-star" style="width:1px;height:1px;top:22%;left:12%;--dur:4.5s;--dly:1.2s;"></div>
<div class="hero-star" style="width:2px;height:2px;top:6%;left:25%;--dur:2.8s;--dly:0.5s;"></div>
<div class="hero-star" style="width:1px;height:1px;top:35%;left:8%;--dur:5.2s;--dly:2.1s;"></div>
<div class="hero-star" style="width:2px;height:2px;top:16%;left:42%;--dur:3.7s;--dly:0.9s;"></div>
<div class="hero-star" style="width:1px;height:1px;top:30%;left:20%;--dur:4.1s;--dly:1.7s;"></div>
<div class="hero-star" style="width:2px;height:2px;top:8%;left:52%;--dur:3.4s;--dly:0.3s;"></div>
<div class="hero-star" style="width:1px;height:1px;top:44%;left:5%;--dur:5.8s;--dly:2.5s;"></div>
<div class="hero-star" style="width:2px;height:2px;top:20%;left:34%;--dur:2.9s;--dly:1.1s;"></div>
<div class="hero-star" style="width:1px;height:1px;top:50%;left:18%;--dur:4.6s;--dly:0.7s;"></div>
<div class="hero-star" style="width:2px;height:2px;top:40%;left:46%;--dur:3.2s;--dly:1.8s;"></div>
<div class="hero-star" style="width:1px;height:1px;top:58%;left:28%;--dur:5.0s;--dly:0.2s;"></div>
<div class="hero-star" style="width:2px;height:2px;top:65%;left:10%;--dur:3.8s;--dly:2.3s;"></div>
<div class="hero-star" style="width:1px;height:1px;top:72%;left:38%;--dur:4.3s;--dly:1.5s;"></div>
<div class="hero-star" style="width:2px;height:2px;top:14%;left:62%;--dur:2.7s;--dly:0.8s;"></div>
<div class="hero-star" style="width:1px;height:1px;top:26%;left:72%;--dur:6.0s;--dly:3.0s;"></div>
<div class="hero-star" style="width:2px;height:2px;top:48%;left:80%;--dur:3.5s;--dly:1.4s;"></div>
<div class="hero-star" style="width:1px;height:1px;top:60%;left:90%;--dur:4.8s;--dly:2.8s;"></div>
<div class="hero-star" style="width:2px;height:2px;top:80%;left:55%;--dur:3.0s;--dly:0.6s;"></div>
<div class="hero-star" style="width:1px;height:1px;top:88%;left:70%;--dur:4.4s;--dly:1.9s;"></div>
<div class="hero-art">
<svg viewBox="0 0 500 220" xmlns="http://www.w3.org/2000/svg">
<defs>
<linearGradient id="hsky" x1="0" y1="0" x2="0" y2="1">
  <stop offset="0%" stop-color="#020810"/>
  <stop offset="100%" stop-color="#0d1e35"/>
</linearGradient>
<clipPath id="aiClip">
  <circle cx="368" cy="104" r="71"/>
</clipPath>
</defs>
<rect width="500" height="220" fill="url(#hsky)"/>
<circle cx="30" cy="18" r="1.2" fill="white" opacity="0.7"/>
<circle cx="88" cy="10" r="0.8" fill="white" opacity="0.5"/>
<circle cx="148" cy="32" r="1.5" fill="white" opacity="0.8"/>
<circle cx="215" cy="16" r="1" fill="white" opacity="0.6"/>
<circle cx="278" cy="38" r="0.8" fill="white" opacity="0.4"/>
<circle cx="342" cy="12" r="1.2" fill="white" opacity="0.7"/>
<circle cx="398" cy="28" r="1" fill="white" opacity="0.5"/>
<circle cx="458" cy="8" r="1.5" fill="white" opacity="0.8"/>
<circle cx="488" cy="44" r="0.8" fill="white" opacity="0.6"/>
<circle cx="55" cy="60" r="1" fill="white" opacity="0.5"/>
<circle cx="178" cy="68" r="0.8" fill="white" opacity="0.4"/>
<circle cx="362" cy="62" r="1.3" fill="white" opacity="0.6"/>
<circle cx="472" cy="75" r="1" fill="white" opacity="0.5"/>
<rect x="0" y="187" width="500" height="33" fill="#10141d"/>
<rect x="0" y="187" width="500" height="2" fill="#1a2030" opacity="0.8"/>
<rect x="10" y="198" width="32" height="4" fill="#b09010" rx="2" opacity="0.7"/>
<rect x="62" y="198" width="32" height="4" fill="#b09010" rx="2" opacity="0.7"/>
<rect x="114" y="198" width="32" height="4" fill="#b09010" rx="2" opacity="0.7"/>
<rect x="166" y="198" width="32" height="4" fill="#b09010" rx="2" opacity="0.7"/>
<rect x="218" y="198" width="32" height="4" fill="#b09010" rx="2" opacity="0.7"/>
<rect x="270" y="198" width="32" height="4" fill="#b09010" rx="2" opacity="0.7"/>
<rect x="322" y="198" width="32" height="4" fill="#b09010" rx="2" opacity="0.7"/>
<rect x="374" y="198" width="32" height="4" fill="#b09010" rx="2" opacity="0.7"/>
<rect x="426" y="198" width="32" height="4" fill="#b09010" rx="2" opacity="0.7"/>
<rect x="478" y="198" width="20" height="4" fill="#b09010" rx="2" opacity="0.7"/>
<!-- Attitude Indicator (Artificial Horizon) centered at (368, 104) r=75 -->
<!-- Outer bezel -->
<circle cx="368" cy="104" r="75" fill="#0e1118" stroke="#2a3040" stroke-width="2"/>
<!-- Clipped earth/sky/instruments group -->
<g clip-path="url(#aiClip)">
  <!-- Sky: upper half -->
  <rect x="297" y="33" width="142" height="71" fill="#1d50a0"/>
  <!-- Earth: lower half -->
  <rect x="297" y="104" width="142" height="71" fill="#7a3c12"/>
  <!-- Pitch ladder at +20° (each 5° ≈ 7px) -->
  <line x1="328" y1="64" x2="408" y2="64" stroke="white" stroke-width="1" opacity="0.5"/>
  <text x="316" y="67" font-size="7" fill="white" opacity="0.5" text-anchor="middle">20</text>
  <text x="420" y="67" font-size="7" fill="white" opacity="0.5" text-anchor="middle">20</text>
  <!-- Pitch ladder at +15° -->
  <line x1="338" y1="71" x2="398" y2="71" stroke="white" stroke-width="1" opacity="0.6"/>
  <!-- Pitch ladder at +10° -->
  <line x1="333" y1="78" x2="403" y2="78" stroke="white" stroke-width="1.2" opacity="0.7"/>
  <text x="321" y="81" font-size="7" fill="white" opacity="0.7" text-anchor="middle">10</text>
  <text x="415" y="81" font-size="7" fill="white" opacity="0.7" text-anchor="middle">10</text>
  <!-- Pitch ladder at +5° -->
  <line x1="343" y1="85" x2="393" y2="85" stroke="white" stroke-width="1" opacity="0.6"/>
  <!-- Horizon line -->
  <line x1="297" y1="104" x2="439" y2="104" stroke="white" stroke-width="2.5"/>
  <!-- Pitch ladder at -5° -->
  <line x1="343" y1="123" x2="393" y2="123" stroke="white" stroke-width="1" opacity="0.6"/>
  <!-- Pitch ladder at -10° -->
  <line x1="333" y1="130" x2="403" y2="130" stroke="white" stroke-width="1.2" opacity="0.7"/>
  <text x="321" y="133" font-size="7" fill="white" opacity="0.7" text-anchor="middle">10</text>
  <text x="415" y="133" font-size="7" fill="white" opacity="0.7" text-anchor="middle">10</text>
  <!-- Pitch ladder at -15° -->
  <line x1="338" y1="137" x2="398" y2="137" stroke="white" stroke-width="1" opacity="0.6"/>
  <!-- Pitch ladder at -20° -->
  <line x1="328" y1="144" x2="408" y2="144" stroke="white" stroke-width="1" opacity="0.5"/>
  <text x="316" y="147" font-size="7" fill="white" opacity="0.5" text-anchor="middle">20</text>
  <text x="420" y="147" font-size="7" fill="white" opacity="0.5" text-anchor="middle">20</text>
  <!-- Miniature aircraft symbol (yellow) -->
  <!-- Left wing bar -->
  <line x1="335" y1="104" x2="356" y2="104" stroke="#f0c020" stroke-width="3" stroke-linecap="round"/>
  <!-- Left wing tip notch up -->
  <line x1="335" y1="104" x2="335" y2="99" stroke="#f0c020" stroke-width="2.5" stroke-linecap="round"/>
  <!-- Right wing bar -->
  <line x1="380" y1="104" x2="401" y2="104" stroke="#f0c020" stroke-width="3" stroke-linecap="round"/>
  <!-- Right wing tip notch up -->
  <line x1="401" y1="104" x2="401" y2="99" stroke="#f0c020" stroke-width="2.5" stroke-linecap="round"/>
  <!-- Center fuselage dot -->
  <circle cx="368" cy="104" r="3" fill="#f0c020"/>
</g>
<!-- Bank angle triangle pointer at 12 o'clock -->
<polygon points="368,30 363,40 373,40" fill="white" opacity="0.9"/>
<!-- Bank angle tick at -60° -->
<line x1="305" y1="46" x2="311" y2="55" stroke="white" stroke-width="1.5" opacity="0.7"/>
<!-- Bank angle tick at -30° -->
<line x1="332" y1="33" x2="335" y2="43" stroke="white" stroke-width="1.5" opacity="0.7"/>
<!-- Bank angle tick at +30° -->
<line x1="404" y1="33" x2="401" y2="43" stroke="white" stroke-width="1.5" opacity="0.7"/>
<!-- Bank angle tick at +60° -->
<line x1="431" y1="46" x2="425" y2="55" stroke="white" stroke-width="1.5" opacity="0.7"/>
<!-- Inner ring highlight -->
<circle cx="368" cy="104" r="71" fill="none" stroke="rgba(255,255,255,0.12)" stroke-width="1.5"/>
</svg>
</div>
<div class="hero-text">
<div class="hero-label">AI-Powered Checkride Prep</div>
<h1 class="hero-title">Pilot Oral<br>Exam Prep</h1>
<p class="hero-sub">Realistic AI examiner &middot; Structured feedback &middot; ACS standards</p>
</div>
</div>"""


# ── App setup ──────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Pilot Oral Exam Prep",
    page_icon="✈️",
    layout="centered",
)
st.markdown(CSS, unsafe_allow_html=True)

if not os.environ.get("ANTHROPIC_API_KEY"):
    st.error(
        "**ANTHROPIC_API_KEY not set.** "
        "Run with: `ANTHROPIC_API_KEY=your-key streamlit run oral_exam_prep.py`"
    )
    st.stop()

# ── Session state ──────────────────────────────────────────────────────────────

_DEFAULTS = {
    "phase": "setup",
    "exam_step": "question",
    "cert": None,
    "subject": None,
    "num_questions": 10,
    "question_num": 0,
    "current_question": None,
    "student_answer": "",
    "current_followup": None,
    "student_followup": "",
    "current_eval": None,
    "history": [],
    "asked_questions": [],
    "session_saved": False,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ✈️ Pilot Oral Exam Prep")
    st.caption("AI-powered checkride preparation")

    if st.session_state.phase == "exam":
        st.write("")
        done = len(st.session_state.history)
        total_q = st.session_state.num_questions
        st.caption(f"**{st.session_state.cert}** · {st.session_state.subject}")
        st.progress(done / total_q, text=f"{done} / {total_q} questions")
        st.write("")
        if st.button("↩ Restart Exam", use_container_width=True):
            for k in list(_DEFAULTS.keys()):
                st.session_state.pop(k, None)
            st.rerun()

# ── Tabs ───────────────────────────────────────────────────────────────────────

tab_exam, tab_progress = st.tabs(["✈️  Oral Exam", "📊  My Progress"])

# ══════════════════════════════════════════════════════════════════════════════
# EXAM TAB
# ══════════════════════════════════════════════════════════════════════════════

with tab_exam:

    # ── SETUP ─────────────────────────────────────────────────────────────────
    if st.session_state.phase == "setup":
        st.markdown(HERO_HTML, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            cert = st.selectbox("Certificate", list(SUBJECTS.keys()))
        with col2:
            subject = st.selectbox("Subject Area", SUBJECTS[cert])

        num_q = st.selectbox(
            "Questions per session",
            options=[5, 10, 20],
            index=1,
            help="Each question includes one follow-up, just like a real oral exam.",
        )

        st.write("")
        if st.button("Begin Oral Exam →", type="primary", use_container_width=True):
            st.session_state.cert = cert
            st.session_state.subject = subject
            st.session_state.num_questions = num_q
            st.session_state.phase = "exam"
            st.session_state.question_num = 1
            st.rerun()

    # ── EXAM ──────────────────────────────────────────────────────────────────
    elif st.session_state.phase == "exam":
        cert = st.session_state.cert
        subject = st.session_state.subject
        q_num = st.session_state.question_num
        total = st.session_state.num_questions

        st.markdown(
            f"<div style='color:#484f58; font-size:0.75em; font-weight:700; "
            f"letter-spacing:0.08em; text-transform:uppercase; margin-bottom:6px;'>"
            f"{cert} &nbsp;·&nbsp; {subject}</div>",
            unsafe_allow_html=True,
        )
        st.progress(len(st.session_state.history) / total, text=f"Question {q_num} of {total}")
        st.write("")

        # Generate question if needed
        if st.session_state.current_question is None:
            with st.spinner("Examiner is preparing a question…"):
                try:
                    q = generate_question(cert, subject, st.session_state.asked_questions)
                    st.session_state.current_question = q
                except Exception as e:
                    st.error(f"**Error generating question** — `{type(e).__name__}: {e}`")
                    with st.expander("Traceback"):
                        st.code(traceback.format_exc())
                    st.stop()

        step = st.session_state.exam_step

        # ── Question step ──
        if step == "question":
            examiner_card(st.session_state.current_question)
            answer = st.text_area(
                "Your answer",
                height=160,
                placeholder="Type your answer here…",
                key=f"ans_{q_num}",
            )
            if st.button("Submit Answer →", type="primary", disabled=not answer.strip()):
                st.session_state.student_answer = answer
                with st.spinner("Examiner is formulating a follow-up…"):
                    try:
                        fu = generate_followup(cert, subject, st.session_state.current_question, answer)
                        st.session_state.current_followup = fu
                    except Exception as e:
                        st.error(f"**Error generating follow-up** — `{type(e).__name__}: {e}`")
                        with st.expander("Traceback"):
                            st.code(traceback.format_exc())
                        st.stop()
                st.session_state.exam_step = "followup"
                st.rerun()

        # ── Follow-up step ──
        elif step == "followup":
            examiner_card(st.session_state.current_question)
            student_card(st.session_state.student_answer)
            followup_card(st.session_state.current_followup)
            fu_answer = st.text_area(
                "Your follow-up answer",
                height=140,
                placeholder="Type your follow-up answer here…",
                key=f"fu_{q_num}",
            )
            if st.button("Submit Follow-up →", type="primary", disabled=not fu_answer.strip()):
                st.session_state.student_followup = fu_answer
                with st.spinner("Evaluating your responses…"):
                    try:
                        ev = evaluate(
                            cert, subject,
                            st.session_state.current_question,
                            st.session_state.student_answer,
                            st.session_state.current_followup,
                            fu_answer,
                        )
                        st.session_state.current_eval = ev
                    except Exception as e:
                        st.error(f"**Error during evaluation** — `{type(e).__name__}: {e}`")
                        with st.expander("Traceback"):
                            st.code(traceback.format_exc())
                        st.stop()
                st.session_state.exam_step = "evaluation"
                st.rerun()

        # ── Evaluation step ──
        elif step == "evaluation":
            ev = st.session_state.current_eval
            rating = ev["rating"]

            examiner_card(st.session_state.current_question)
            student_card(st.session_state.student_answer)
            followup_card(st.session_state.current_followup)
            student_card(st.session_state.student_followup, "Your Follow-up Answer")
            st.divider()

            rating_badge(rating)
            st.markdown(ev["feedback"])
            st.write("")

            col1, col2 = st.columns(2)
            with col1:
                if ev["key_points_covered"]:
                    st.markdown(
                        "<div style='color:#3fb950; font-size:0.72em; font-weight:700; "
                        "letter-spacing:0.08em; text-transform:uppercase; margin-bottom:8px;'>"
                        "Demonstrated</div>",
                        unsafe_allow_html=True,
                    )
                    for pt in ev["key_points_covered"]:
                        st.markdown(f"- {pt}")
            with col2:
                if ev["key_points_missed"]:
                    st.markdown(
                        "<div style='color:#d29922; font-size:0.72em; font-weight:700; "
                        "letter-spacing:0.08em; text-transform:uppercase; margin-bottom:8px;'>"
                        "Missed / Incomplete</div>",
                        unsafe_allow_html=True,
                    )
                    for pt in ev["key_points_missed"]:
                        st.markdown(f"- {pt}")

            st.write("")
            st.caption(f"DPE note: {ev['examiner_note']}")
            st.write("")

            is_last = q_num >= total
            btn_label = "View Session Summary →" if is_last else f"Next Question ({q_num + 1} of {total}) →"
            if st.button(btn_label, type="primary", use_container_width=True):
                st.session_state.history.append({
                    "question_num": q_num,
                    "question": st.session_state.current_question,
                    "answer": st.session_state.student_answer,
                    "followup": st.session_state.current_followup,
                    "followup_answer": st.session_state.student_followup,
                    "rating": rating,
                    "feedback": ev["feedback"],
                    "covered": ev["key_points_covered"],
                    "missed": ev["key_points_missed"],
                })
                st.session_state.asked_questions.append(st.session_state.current_question)

                if is_last:
                    if not st.session_state.session_saved:
                        save_session(
                            cert, subject, total,
                            [
                                {
                                    "question_num": h["question_num"],
                                    "question": h["question"],
                                    "rating": h["rating"],
                                    "covered": h["covered"],
                                    "missed": h["missed"],
                                }
                                for h in st.session_state.history
                            ],
                        )
                        st.session_state.session_saved = True
                    st.session_state.phase = "summary"
                else:
                    st.session_state.question_num += 1
                    st.session_state.current_question = None
                    st.session_state.student_answer = ""
                    st.session_state.current_followup = None
                    st.session_state.student_followup = ""
                    st.session_state.current_eval = None
                    st.session_state.exam_step = "question"
                st.rerun()

    # ── SUMMARY ───────────────────────────────────────────────────────────────
    elif st.session_state.phase == "summary":
        history = st.session_state.history
        passes = sum(1 for h in history if h["rating"] == "Pass")
        needs_work = sum(1 for h in history if h["rating"] == "Needs Work")
        unsat = sum(1 for h in history if h["rating"] == "Unsatisfactory")
        total = len(history)

        st.title("Session Complete")
        st.caption(f"{st.session_state.cert} · {st.session_state.subject} · {total} questions")
        st.divider()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Pass", passes)
        with col2:
            st.metric("Needs Work", needs_work)
        with col3:
            st.metric("Unsatisfactory", unsat)

        st.write("")
        if unsat == 0 and needs_work <= 1:
            st.success("Strong performance — you're in good shape for this subject area.")
        elif unsat == 0:
            st.warning("Solid foundation, but a few areas need review before your checkride.")
        else:
            st.error("Significant gaps identified — focused study of this subject is recommended.")

        all_missed = []
        for h in history:
            if h["rating"] != "Pass":
                all_missed.extend(h["missed"])
        if all_missed:
            st.divider()
            st.markdown(
                "<div style='color:#d29922; font-size:0.72em; font-weight:700; "
                "letter-spacing:0.08em; text-transform:uppercase; margin-bottom:12px;'>"
                "Key Areas to Review</div>",
                unsafe_allow_html=True,
            )
            for pt in dict.fromkeys(all_missed):
                st.markdown(f"- {pt}")

        st.divider()
        st.subheader("Question Breakdown")
        for h in history:
            emoji = RATING_EMOJI[h["rating"]]
            preview = h["question"][:68] + "…" if len(h["question"]) > 68 else h["question"]
            with st.expander(f"Q{h['question_num']} {emoji} {h['rating']} — {preview}"):
                examiner_card(h["question"])
                student_card(h["answer"])
                followup_card(h["followup"])
                student_card(h["followup_answer"], "Your Follow-up Answer")
                st.divider()
                st.markdown(f"**Feedback:** {h['feedback']}")
                if h["covered"]:
                    st.caption("Demonstrated: " + " · ".join(h["covered"]))
                if h["missed"]:
                    st.caption("Missed: " + " · ".join(h["missed"]))

        st.divider()
        if st.button("Start New Session", type="primary", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PROGRESS TAB
# ══════════════════════════════════════════════════════════════════════════════

with tab_progress:
    sessions = load_sessions()

    if not sessions:
        st.markdown(
            "<div style='text-align:center; padding:56px 24px;'>"
            "<div style='font-size:2.5em; margin-bottom:14px;'>📊</div>"
            "<div style='color:#8b949e; font-size:1.05em; font-weight:600;'>No sessions yet</div>"
            "<div style='color:#484f58; font-size:0.88em; margin-top:8px;'>"
            "Complete your first oral exam session to see your progress here."
            "</div></div>",
            unsafe_allow_html=True,
        )
    else:
        rs = readiness_score()
        readiness_display(rs["score"], rs["total_q"], rs["total_s"])
        st.write("")

        # Weak areas
        wa = weak_areas()
        st.markdown(
            "<div style='color:#484f58; font-size:0.72em; font-weight:700; "
            "letter-spacing:0.1em; text-transform:uppercase; margin:24px 0 10px 0;'>"
            "Areas Needing Work</div>",
            unsafe_allow_html=True,
        )
        if not wa:
            st.success("All studied subject areas are at or above 70% pass rate.")
        else:
            for area in wa:
                pct = round(area["pass_rate"] * 100)
                color = "#f85149" if pct < 50 else "#d29922"
                st.markdown(
                    f'<div style="display:flex; justify-content:space-between; align-items:center; '
                    f'background:#161b22; border:1px solid #30363d; border-left:3px solid {color}; '
                    f'border-radius:0 6px 6px 0; padding:11px 16px; margin:5px 0;">'
                    f'<span style="color:#e6edf3; font-weight:600; font-size:0.93em;">{area["subject"]}</span>'
                    f'<span style="color:{color}; font-size:0.8em; font-weight:700;">'
                    f'{pct}% pass &nbsp;·&nbsp; {area["total"]}Q</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Session history
        st.divider()
        st.markdown(
            "<div style='color:#484f58; font-size:0.72em; font-weight:700; "
            "letter-spacing:0.1em; text-transform:uppercase; margin-bottom:10px;'>"
            "Session History</div>",
            unsafe_allow_html=True,
        )
        for sess in reversed(sessions):
            qs = sess["questions"]
            n = len(qs)
            n_pass = sum(1 for q in qs if q["rating"] == "Pass")
            pct = round(n_pass / n * 100) if n else 0
            ts = datetime.fromisoformat(sess["timestamp"]).strftime("%b %d, %Y")
            label = f"{ts}  ·  {sess['cert']}  ·  {sess['subject']}  ·  {n_pass}/{n} Pass"
            with st.expander(label):
                for q in qs:
                    emoji = RATING_EMOJI.get(q["rating"], "⚪")
                    st.markdown(f"**Q{q['question_num']}** {emoji} *{q['question']}*")
                    if q.get("covered"):
                        st.caption("Demonstrated: " + " · ".join(q["covered"]))
                    if q.get("missed"):
                        st.caption("Missed: " + " · ".join(q["missed"]))
                    st.write("")
