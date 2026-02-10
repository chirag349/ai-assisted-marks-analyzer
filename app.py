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

st.set_page_config(
    page_title="AI Marks Analyzer",
    layout="centered"
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(-45deg,#0f2027,#203a43,#2c5364,#141E30);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
}
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
h1,h2,h3,h4,h5,h6,p,label { color: white !important; }
</style>
""", unsafe_allow_html=True)

st.title("AI-Assisted Answer Evaluation System")
st.subheader("Teacher 👨🏻‍🏫")

col1, col2 = st.columns(2)
with col1:
    refference_pdf = st.file_uploader("Upload Reference Answer (PDF)", type=["pdf"])
with col2:
    question_paper = st.file_uploader("Upload Question Paper", type=["pdf","jpg","png"])

st.subheader("Student 🧑‍🎓")
answer_sheet = st.file_uploader("Upload Answer Sheet", type=["jpg","png","pdf"])

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def ocr_core(img):
    return pytesseract.image_to_string(img, lang="eng", config="--psm 6")

def is_handwritten(path):
    img = cv2.imread(path, 0)
    edges = cv2.Canny(img, 50, 150)
    density = np.sum(edges) / edges.size
    return density > 0.02

def extract_reference_text(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + " "
    return text

def calculate_similarity(reference_text, student_text):
    documents = [reference_text, student_text]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return score

processor_hw = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model_hw = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

if st.button("Evaluate") and answer_sheet is not None and refference_pdf is not None:

    student_text = ""

    if answer_sheet.type == "application/pdf":
        pages = convert_from_bytes(
            answer_sheet.read(),
            poppler_path=r"C:\poppler-25.12.0\Library\bin"
        )
        for page in pages:
            student_text += ocr_core(page) + " "
    else:
        img = Image.open(answer_sheet)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp.name)
            temp_path = tmp.name

        if not is_handwritten(temp_path):
            student_text = ocr_core(img)
        else:
            pixel_values = processor_hw(images=img.convert("RGB"), return_tensors="pt").pixel_values
            ids = model_hw.generate(
                pixel_values,
                max_length=64,
                num_beams=5,
                early_stopping=True
            )
            student_text = processor_hw.batch_decode(ids, skip_special_tokens=True)[0]

    reference_text = extract_reference_text(refference_pdf)

    reference_text = clean_text(reference_text)
    student_text = clean_text(student_text)

    similarity_score = calculate_similarity(reference_text, student_text)

    max_marks = 10

    if similarity_score >= 0.95:
        final_marks = max_marks
    else:
        final_marks = round(similarity_score * max_marks, 2)

    st.subheader("Evaluation Result")
    st.write("Similarity Score:", round(similarity_score, 3))
    st.write("Final Marks:", final_marks, "/", max_marks)
