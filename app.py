import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import matplotlib.pyplot as plt

# Load model (cache to avoid reloading every run)
@st.cache_resource
def load_model():
    model_name = "distilbert-base-uncased-distilled-squad"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return qa_pipeline

qa_pipeline = load_model()

# Streamlit App
st.title("🚀 Space Exploration QA System")
st.write("Ask questions about space exploration and get answers using a transformer-based model!")

# Context input
context = st.text_area("Enter context:", height=200, value="""
Space exploration is the ongoing discovery and exploration of celestial structures in outer space
by means of continuously evolving and growing space technology. While the study of space is carried
out mainly by astronomers with telescopes, its physical exploration is conducted both by unmanned
robotic space probes and human spaceflight. Major milestones include the first satellite Sputnik in 1957,
the first human in space Yuri Gagarin in 1961, the Apollo 11 Moon landing in 1969,
and the ongoing Mars rover missions. Future exploration includes planned human missions to Mars and
the establishment of permanent lunar bases.
""")

# Question input
question = st.text_input("Ask a question:")

if st.button("Get Answer"):
    if context.strip() == "" or question.strip() == "":
        st.warning("Please provide both context and a question!")
    else:
        result = qa_pipeline(question=question, context=context)
        st.success(f"**Answer:** {result['answer']}")
        st.info(f"**Confidence:** {result['score']:.2f}")

        # Bar chart of confidence (optional)
        st.subheader("Confidence Visualization")
        plt.bar([question], [result['score']], color="skyblue")
        plt.ylim(0, 1)
        plt.ylabel("Confidence")
        st.pyplot(plt)
