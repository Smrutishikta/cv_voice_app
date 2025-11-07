# app.py
import streamlit as st
import speech_recognition as sr
from gtts import gTTS
import os
from utils import KnowledgeBase
from langchain.chains import RetrievalQA
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

st.title("🎤 Smruti’s Voice Chatbot")
st.write("Ask me anything about Smruti’s skills, education, or experience!")

# ----------- Load Knowledge Base -----------
kb = KnowledgeBase()
text_chunks = kb.text_chunks

# ----------- Setup LangChain FAISS Retriever -----------
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.from_texts(text_chunks, embedding_model)

# ----------- Setup LLM Pipeline -----------
llm_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=-1  # 0 for GPU
)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=False
)

# ------------------ User Input ------------------
st.subheader("💬 Ask via text")
user_text = st.text_input("Type your question here:")

st.subheader("🎙 Or ask via voice")
if st.button("Record Question"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening for 5 seconds...")
        try:
            audio = recognizer.listen(source, timeout=5)
            user_text = recognizer.recognize_google(audio)
            st.success(f"You said: {user_text}")
        except sr.WaitTimeoutError:
            st.error("No speech detected. Please try again.")
        except sr.UnknownValueError:
            st.error("Could not understand audio.")
        except sr.RequestError as e:
            st.error(f"Speech recognition error: {e}")

# ------------------ Generate Answer ------------------
if user_text:
    answer = qa_chain.run(user_text)
    st.subheader("Answer:")
    st.write(answer)

    # Generate and play speech
    tmp_path = os.path.join(os.path.abspath(os.getcwd()), "response.mp3")
    tts = gTTS(answer)
    tts.save(tmp_path)
    st.audio(tmp_path, format="audio/mp3")

