import streamlit as st
from transformers import pipeline

# Initialize Hugging Face pipelines
explainer = pipeline("text2text-generation", model="google/flan-t5-base")
quiz_generator = pipeline("text2text-generation", model="google/flan-t5-base")
summarizer = pipeline("text2text-generation", model="google/flan-t5-small")

# Helper functions
def explain_topic(topic):
    prompt = f"Explain in simple terms how {topic} works. Make it easy to understand but informative."
    result = explainer(prompt, max_new_tokens=50, no_repeat_ngram_size=3, do_sample=False, num_beams=5)
    return result[0]['generated_text'].strip()

def generate_quiz(topic):
    prompt = f"Create 3 multiple-choice questions with answers from the topic: {topic}"
    result = quiz_generator(prompt, max_new_tokens=300)
    return result[0]['generated_text'].strip()

def summarize_text(text):
    prompt = f"Summarize the following text in simple 1 line:\n{text}"
    result = summarizer(prompt, max_new_tokens=100)
    return result[0]['generated_text'].strip()

# Streamlit UI
st.title("AI Study Buddy 📚")
st.write("Learn, quiz yourself, and summarize topics easily!")

topic = st.text_input("Enter a topic:")

if topic:
    # Phase 1: Explanation
    st.subheader("Simple Explanation")
    explanation = explain_topic(topic)
    st.write(explanation)
    
    # Phase 2: Generate Quiz (based on topic)
    if st.button("Generate Questions"):
        st.subheader("Quiz Questions")
        quiz = generate_quiz(topic)  # <-- use topic directly
        st.write(quiz)

    # Phase 3: Summarize
    if st.button("Get Summary"):
        st.subheader("Summary")
        summary = summarize_text(explanation)
        st.write(summary)