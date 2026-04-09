import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import tempfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import torch

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"D:\tesseract\tesseract.exe"

model_name = "microsoft/trocr-base-handwritten"
processor = TrOCRProcessor.from_pretrained(model_name)
model = VisionEncoderDecoderModel.from_pretrained(model_name)

st.set_page_config(page_title="AI Marks Analyzer", layout="centered", page_icon="🎯")

# ── LOADING SCREEN ──
st.markdown("""
<style>
#loader-overlay {
    position: fixed;
    inset: 0;
    background: #03060A;
    z-index: 99999;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 2rem;
    animation: loaderFadeOut 0.6s ease-out 2.4s forwards;
}
@keyframes loaderFadeOut {
    from { opacity: 1; pointer-events: all; }
    to   { opacity: 0; pointer-events: none; visibility: hidden; }
}
.loader-ring {
    position: relative;
    width: 90px;
    height: 90px;
}
.loader-ring::before,
.loader-ring::after {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 50%;
}
.loader-ring::before {
    border: 2px solid rgba(0,220,255,0.12);
}
.loader-ring::after {
    border: 2px solid transparent;
    border-top-color: #00DCFF;
    border-right-color: rgba(0,220,255,0.4);
    animation: spin 0.9s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.loader-ring-inner {
    position: absolute;
    inset: 14px;
    border-radius: 50%;
    border: 1.5px solid transparent;
    border-bottom-color: #6030FF;
    border-left-color: rgba(96,48,255,0.4);
    animation: spin 1.3s linear infinite reverse;
}
.loader-dot {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.3rem;
    animation: pulseDot 1.5s ease-in-out infinite;
}
@keyframes pulseDot {
    0%, 100% { transform: scale(0.85); opacity: 0.6; }
    50%       { transform: scale(1.1);  opacity: 1; }
}
.loader-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.4);
    animation: blink 1.2s step-start infinite;
}
@keyframes blink {
    0%, 100% { opacity: 0.4; }
    50%       { opacity: 0.9; }
}
.loader-brand {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.5rem;
    letter-spacing: -0.03em;
    color: #fff;
    margin-bottom: -1rem;
}
.loader-brand span {
    background: linear-gradient(135deg, #00DCFF, #6030FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.loader-scanline {
    width: 160px;
    height: 2px;
    background: linear-gradient(90deg, transparent, #00DCFF, transparent);
    border-radius: 2px;
    animation: scanPulse 1.5s ease-in-out infinite;
}
@keyframes scanPulse {
    0%, 100% { opacity: 0.2; transform: scaleX(0.4); }
    50%       { opacity: 1.0; transform: scaleX(1.0); }
}
</style>

<div id="loader-overlay">
    <div class="loader-brand">Marks <span>Analyzer</span></div>
    <div class="loader-ring">
        <div class="loader-ring-inner"></div>
        <div class="loader-dot">MA</div>
    </div>
    <div class="loader-scanline"></div>
    <div class="loader-text">Initializing AI model...</div>
</div>
""", unsafe_allow_html=True)

# ── MAIN STYLES ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

[data-testid="stAppViewContainer"] {
    background: #03060A;
    min-height: 100vh;
    font-family: 'JetBrains Mono', monospace;
    overflow-x: hidden;
}
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 90% 60% at 15% 5%,  rgba(0,220,255,0.09) 0%, transparent 55%),
        radial-gradient(ellipse 70% 50% at 85% 90%, rgba(120,60,255,0.1)  0%, transparent 55%),
        radial-gradient(ellipse 50% 40% at 50% 50%, rgba(0,150,255,0.04)  0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
    animation: bgPulse 8s ease-in-out infinite alternate;
}
@keyframes bgPulse {
    0%   { opacity: 0.7; }
    50%  { opacity: 1.0; }
    100% { opacity: 0.7; }
}
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        repeating-linear-gradient(0deg,  transparent, transparent 59px, rgba(0,220,255,0.03) 59px, rgba(0,220,255,0.03) 60px),
        repeating-linear-gradient(90deg, transparent, transparent 59px, rgba(0,220,255,0.03) 59px, rgba(0,220,255,0.03) 60px);
    pointer-events: none;
    z-index: 0;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] { display: none !important; }

.block-container {
    max-width: 840px !important;
    padding: 3.5rem 2rem 6rem !important;
    position: relative;
    z-index: 1;
}

/* ── HEADER ── */
.main-header {
    text-align: center;
    margin-bottom: 3.5rem;
    animation: fadeDown 0.7s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}
@keyframes fadeDown {
    from { opacity: 0; transform: translateY(-20px); }
    to   { opacity: 1; transform: translateY(0); }
}
.main-header .badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #00DCFF;
    background: rgba(0,220,255,0.1);
    border: 1px solid rgba(0,220,255,0.25);
    border-radius: 100px;
    padding: 0.35rem 1rem;
    margin-bottom: 1rem;
}
.main-header h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 3rem !important;
    color: #ffffff !important;
    letter-spacing: -0.04em !important;
    line-height: 1.05 !important;
    margin: 0 !important;
}
.main-header h1 span {
    background: linear-gradient(135deg, #00DCFF, #0077FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.main-header p {
    font-family: 'JetBrains Mono', monospace !important;
    color: rgba(255,255,255,0.4) !important;
    font-size: 0.78rem !important;
    margin-top: 0.8rem !important;
}

/* ── SECTION LABELS ── */
h2, .stSubheader p {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    color: #00DCFF !important;
    letter-spacing: 0.2em !important;
    text-transform: uppercase !important;
    margin: 2.5rem 0 1rem !important;
}
p, label, .stMarkdown p {
    color: rgba(255,255,255,0.55) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important;
    padding: 0.2rem !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.03) !important;
    border: 1.5px dashed rgba(255,255,255,0.1) !important;
    border-radius: 18px !important;
    padding: 1rem !important;
    transition: border-color 0.3s, background 0.3s, box-shadow 0.3s !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: rgba(0,220,255,0.35) !important;
    background: rgba(0,220,255,0.04) !important;
    box-shadow: 0 0 28px rgba(0,220,255,0.06) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: rgba(255,255,255,0.35) !important;
    font-size: 0.75rem !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: rgba(0,220,255,0.1) !important;
    color: #00DCFF !important;
    border: 1.5px solid rgba(0,220,255,0.3) !important;
    border-radius: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.73rem !important;
    padding: 0.45rem 1.1rem !important;
    letter-spacing: 0.05em !important;
    transition: background 0.2s, border-color 0.2s, box-shadow 0.2s, transform 0.15s !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background: rgba(0,220,255,0.2) !important;
    border-color: rgba(0,220,255,0.6) !important;
    box-shadow: 0 0 18px rgba(0,220,255,0.25) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stFileUploaderFile"] {
    background: rgba(0,220,255,0.07) !important;
    border: 1px solid rgba(0,220,255,0.2) !important;
    border-radius: 10px !important;
    color: rgba(255,255,255,0.8) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
}

/* ── DIVIDER ── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(0,220,255,0.15), transparent);
    margin: 2rem 0;
    animation: shimmer 3s ease-in-out infinite;
}
@keyframes shimmer {
    0%, 100% { opacity: 0.4; }
    50%       { opacity: 1.0; }
}

/* ── EVALUATE BUTTON ── */
.stButton > button {
    background: linear-gradient(135deg, #00DCFF 0%, #0055FF 100%) !important;
    color: #020509 !important;
    border: none !important;
    border-radius: 14px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
    height: 3.6em !important;
    width: 100% !important;
    margin-top: 1rem !important;
    transition: opacity 0.2s, transform 0.15s, box-shadow 0.2s !important;
    animation: btnGlow 3s ease-in-out infinite alternate !important;
}
@keyframes btnGlow {
    from { box-shadow: 0 6px 30px rgba(0,200,255,0.3); }
    to   { box-shadow: 0 6px 40px rgba(0,120,255,0.55); }
}
.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-3px) !important;
    box-shadow: 0 12px 40px rgba(0,200,255,0.6) !important;
    color: #020509 !important;
}
.stButton > button:active { transform: translateY(1px) !important; }

/* ── SPINNER ── */
[data-testid="stSpinner"] > div { border-top-color: #00DCFF !important; }

/* ── SCORE CARD ── */
.score-wrapper {
    animation: slideUp 0.5s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    opacity: 0;
}
@keyframes slideUp {
    from { opacity: 0; transform: translateY(24px); }
    to   { opacity: 1; transform: translateY(0); }
}
.score-card {
    border-radius: 20px;
    padding: 2rem 2.4rem;
    margin-top: 1.2rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: relative;
    overflow: hidden;
}
.score-card::before {
    content: '';
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
        45deg, transparent, transparent 20px,
        rgba(255,255,255,0.012) 20px, rgba(255,255,255,0.012) 21px
    );
    pointer-events: none;
}
.score-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    opacity: 0.45;
    margin-bottom: 0.4rem;
    color: #fff !important;
}
.score-value {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 4.2rem;
    line-height: 1;
    color: #fff;
    letter-spacing: -0.04em;
}
.score-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    opacity: 0.4;
    color: #fff;
    margin-top: 0.5rem;
    letter-spacing: 0.1em;
}
.score-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.6rem 1.4rem;
    border-radius: 100px;
    border: 1.5px solid rgba(255,255,255,0.35);
    color: white;
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
}

/* ── MARKS CARD ── */
.marks-card {
    border-radius: 20px;
    padding: 1.6rem 2.4rem;
    margin-top: 1rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    position: relative;
    overflow: hidden;
    animation: slideUp 0.6s 0.15s cubic-bezier(0.16, 1, 0.3, 1) both;
}
.marks-left { display: flex; flex-direction: column; gap: 0.25rem; }
.marks-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.35) !important;
}
.marks-value {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 2.6rem;
    line-height: 1;
    letter-spacing: -0.03em;
}
.marks-total {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: rgba(255,255,255,0.3) !important;
    margin-top: 0.3rem;
}
.marks-bar-wrap {
    flex: 1;
    margin: 0 2rem;
}
.marks-bar-bg {
    height: 8px;
    background: rgba(255,255,255,0.07);
    border-radius: 100px;
    overflow: hidden;
}
.marks-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 1s cubic-bezier(0.16, 1, 0.3, 1);
}

/* ── ALERTS ── */
[data-testid="stAlert"] {
    border-radius: 14px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    border: none !important;
    background: rgba(255,80,80,0.1) !important;
}

/* ── NUMBER INPUT ── */
[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.05) !important;
    border: 1.5px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #fff !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: rgba(0,220,255,0.5) !important;
    box-shadow: 0 0 12px rgba(0,220,255,0.15) !important;
}

/* ── SCROLLBAR ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(0,220,255,0.18); border-radius: 3px; }

[data-testid="stHorizontalBlock"] { gap: 1rem !important; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ──
st.markdown("""
<div class="main-header">
    <div class="badge">✦ AI Assisted</div>
    <h1>Marks <span>Analyzer</span></h1>
    
</div>
""", unsafe_allow_html=True)

# ── TEACHER SECTION ──
st.subheader("👨🏻‍🏫  Teacher Upload")
col1, col2 = st.columns(2)
with col1:
    refference_pdf = st.file_uploader("Reference Answer PDF", type=["pdf"])
with col2:
    question_paper = st.file_uploader("Question Paper", type=["pdf","jpg","png"])

# ── TOTAL MARKS INPUT ──
st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)
total_marks = st.number_input(
    "Total Marks for this Paper",
    min_value=1,
    max_value=1000,
    value=100,
    step=1,
    help="Enter the maximum marks for this question paper"
)

st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

# ── STUDENT SECTION ──
st.subheader("🧑‍🎓  Student Upload")
answer_sheet = st.file_uploader("Answer Sheet", type=["jpg","png","pdf"])


# ── FUNCTIONS ──

def clean_text(text):
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    return text.lower()


def image_to_text(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    image = np.array(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    text = pytesseract.image_to_string(thresh)
    return text


def pdf_to_text(pdf_file):
    images = convert_from_bytes(
        pdf_file.read(),
        poppler_path=r"D:\Downloads\Release-25.12.0-0\poppler-25.12.0\Library\bin"
    )
    text = ""
    for img in images:
        text += image_to_text(img) + " "
    return text


def file_to_text(file):
    if file.type == "application/pdf":
        return pdf_to_text(file)
    else:
        image = Image.open(file).convert("RGB")
        return image_to_text(image)


def calculate_similarity(text1, text2):
    text1 = clean_text(text1)
    text2 = clean_text(text2)

    if len(text1.strip()) == 0 or len(text2.strip()) == 0:
        return 0

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    score = cosine_similarity(vectors[0], vectors[1])[0][0]
    return score * 100


def calculate_marks(similarity_score, total_marks):
    """
    Marks calculation based on similarity score.
    Uses a slightly curved formula so even partial answers get fair marks.
    """
    # Apply a mild curve: raw similarity can be harsh, so we soften it a bit
    # Formula: earned = total * (similarity/100) ^ 0.85
    # This means 60% similarity → ~62.5% marks, 80% → ~81.5%, 100% → 100%
    import math
    if similarity_score <= 0:
        return 0
    fraction = (similarity_score / 100) ** 0.85
    earned = round(total_marks * fraction, 1)
    return min(earned, total_marks)  # cap at total marks


# ── EVALUATE BUTTON ──
st.markdown("<div style='margin-top:0.5rem'></div>", unsafe_allow_html=True)

if st.button("Evaluate"):
    if refference_pdf is not None and answer_sheet is not None:
        with st.spinner():
            ref_text     = file_to_text(refference_pdf)
            student_text = file_to_text(answer_sheet)
            score        = calculate_similarity(ref_text, student_text)
            earned_marks = calculate_marks(score, total_marks)

        if score >= 80:
            grade, bar_color, icon = "Excellent", "#00E676", "🏆"
        elif score >= 60:
            grade, bar_color, icon = "Good", "#29B6F6", "✅"
        elif score >= 40:
            grade, bar_color, icon = "Average", "#FFA726", "⚡"
        else:
            grade, bar_color, icon = "Needs Work", "#EF5350", "📌"

        # ── SIMILARITY SCORE CARD (unchanged) ──
        st.markdown(f"""
        <div class="score-wrapper">
            <div class="score-card" style="
                background: linear-gradient(135deg, {bar_color}1A, {bar_color}0D);
                border: 1px solid {bar_color}55;
                box-shadow: 0 0 60px {bar_color}18, inset 0 1px 0 rgba(255,255,255,0.07);
            ">
                <div>
                    <p class="score-eyebrow">Similarity Score</p>
                    <p class="score-value">{score:.1f}<span style="font-size:1.8rem;opacity:0.4;font-weight:300;">%</span></p>
                    <p class="score-sub">TF-IDF · Cosine Similarity</p>
                </div>
                <div class="score-pill">{icon} {grade}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── MARKS CARD (new) ──
        bar_pct = (earned_marks / total_marks) * 100
        st.markdown(f"""
        <div class="marks-card">
            <div class="marks-left">
                <span class="marks-label">Marks Obtained</span>
                <span class="marks-value" style="color:{bar_color};">{earned_marks}</span>
                <span class="marks-total">out of {total_marks}</span>
            </div>
            <div class="marks-bar-wrap">
                <div class="marks-bar-bg">
                    <div class="marks-bar-fill"
                         style="width:{bar_pct:.1f}%;
                                background:linear-gradient(90deg,{bar_color}99,{bar_color});"></div>
                </div>
                <div style="display:flex;justify-content:space-between;margin-top:0.4rem;">
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:rgba(255,255,255,0.25);">0</span>
                    <span style="font-family:'JetBrains Mono',monospace;font-size:0.6rem;color:rgba(255,255,255,0.25);">{total_marks}</span>
                </div>
            </div>
            <div class="score-pill" style="border-color:{bar_color}55;min-width:fit-content;">
                {bar_pct:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.error("Upload both the reference answer and the student's answer sheet to begin.")
