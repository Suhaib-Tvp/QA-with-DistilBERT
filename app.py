import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import matplotlib.pyplot as plt

# -----------------------------
# Load Model (cached for speed)
# -----------------------------
@st.cache_resource
def load_model():
    model_name = "distilbert-base-uncased-distilled-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return qa_pipeline

qa_pipeline = load_model()

# -----------------------------
# App Header
# -----------------------------
st.title("🛰️ Space Exploration QA System")
st.write("""
Ask questions about **space exploration** and get answers extracted from a context passage using DistilBERT.
""")

# -----------------------------
# Editable Context
# -----------------------------
context = st.text_area("Context Passage", value="""
Space exploration is the ongoing discovery and exploration of celestial structures in outer space
by means of continuously evolving and growing space technology. While the study of space is carried
out mainly by astronomers with telescopes, its physical exploration is conducted both by unmanned
robotic space probes and human spaceflight. Major milestones include the first satellite Sputnik in 1957,
the first human in space Yuri Gagarin in 1961, the Apollo 11 Moon landing in 1969,
and the ongoing Mars rover missions. Future exploration includes planned human missions to Mars and
the establishment of permanent lunar bases.
""", height=200)

# -----------------------------
# Single Question Input
# -----------------------------
question = st.text_input("Ask a question about space exploration:")

if question:
    result = qa_pipeline(question=question, context=context)
    st.subheader("Answer:")
    st.write(f"**{result['answer']}**")
    st.write(f"Confidence: {result['score']:.2f}")

# -----------------------------
# Multiple Questions Input
# -----------------------------
st.subheader("Multiple Questions (Optional)")

questions_text = st.text_area("Enter multiple questions (one per line):", value="""
Who was the first human in space?
When was the first satellite launched?
What planet has rover missions?
What future plans exist for space exploration?
""")

questions = [q.strip() for q in questions_text.split("\n") if q.strip()]

if questions:
    answers = [qa_pipeline(question=q, context=context) for q in questions]
    scores = [a["score"] for a in answers]
    labels = [f"Q{i+1}" for i in range(len(questions))]

    # -----------------------------
    # Plot confidence scores
    # -----------------------------
    fig, ax = plt.subplots()
    ax.bar(labels, scores, color="skyblue")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Confidence")
    ax.set_title("Confidence Scores for QA Predictions")
    st.pyplot(fig)
