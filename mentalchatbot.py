# import os
# import streamlit as st
# import whisper
# import sounddevice as sd
# import soundfile as sf
# import tempfile
# import numpy as np
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.chains import RetrievalQA
# from langchain_community.vectorstores import FAISS
# from langchain_core.prompts import PromptTemplate
# from langchain_huggingface import HuggingFaceEndpoint
# from gtts import gTTS
# import pygame

# whisper_model = whisper.load_model("base")

# def record_and_transcribe(duration=4, samplerate=16000):
#     st.info("🎙️ Listening... Please speak now...")
#     try:
#         recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
#         sd.wait()

#         with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
#             sf.write(f.name, recording, samplerate)
#             audio_path = f.name

#         result = whisper_model.transcribe(audio_path, language="en")
#         os.remove(audio_path)
#         return result["text"]
#     except Exception as e:
#         return f"❌ Error during transcription: {e}"


# DB_FAISS_PATH = "vectorstore/db_faiss"

# @st.cache_resource
# def get_vectorstore():
#     embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
#     db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
#     return db

# def set_custom_prompt(custom_prompt_template):
#     prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])
#     return prompt

# def load_llm(huggingface_repo_id, HF_TOKEN):
#     llm = HuggingFaceEndpoint(
#         repo_id=huggingface_repo_id,
#         temperature=0.5,
#         task="text-generation",
#         model_kwargs={
#             "token": HF_TOKEN,
#             "max_length": "512"
#         }
#     )
#     return llm




# def main():
    
#     st.title("🧠 Ask Chatbot!")
#     st.caption("Type your query or use the mic to speak.")

#     if 'messages' not in st.session_state:
#         st.session_state.messages = []

#     for message in st.session_state.messages:
#         st.chat_message(message['role']).markdown(message['content'])

#     col1, col2 = st.columns([3, 1])
#     with col1:
#         user_text_input = st.chat_input("Type your prompt or use the mic...")

#     with col2:
#         mic_clicked = st.button("🎤 Speak")

#     if mic_clicked:
#         user_text_input = record_and_transcribe()

#     if user_text_input:
#         st.chat_message('user').markdown(user_text_input)
#         st.session_state.messages.append({'role': 'user', 'content': user_text_input})

#         CUSTOM_PROMPT_TEMPLATE = """
#         Use the pieces of information provided in the context to answer user's question.
#         If you don't know the answer, just say that you don't know, don't try to make up an answer. 
#         Don't provide anything out of the given context.

#         Context: {context}
#         Question: {question}

#         Start the answer directly. No small talk please.
#         """

#         HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
#         HF_TOKEN = os.environ.get("HF_TOKEN")

#         try:
#             vectorstore = get_vectorstore()
#             if vectorstore is None:
#                 st.error("Failed to load the vector store")

#             qa_chain = RetrievalQA.from_chain_type(
#                 llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
#                 chain_type="stuff",
#                 retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
#                 return_source_documents=True,
#                 chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
#             )

#             response = qa_chain.invoke({'query': user_text_input})
#             result = response["result"]
#             source_documents = response["source_documents"]

#             result_to_show = result + "\n\n📚 Source Docs:\n" + str(source_documents)
#             st.chat_message('assistant').markdown(result_to_show)
#             st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

#         except Exception as e:
#             st.error(f"❌ Error: {str(e)}")
# with st.sidebar:
#     st.title("💬 About")
#     st.write("""
#     This is an AI-powered chatbot that understands documents.
    
#     🔍 Ask questions about any PDF you've uploaded.
#     🎤 Speak your query too!

#     Example:
#     - "Summarize the key points of the PDF"
#     - "What is emotional support therapy?"
#     """)


# if __name__ == "__main__":
#     main()


import os
import streamlit as st
import whisper
import sounddevice as sd
import soundfile as sf
import tempfile
import numpy as np
import pickle
import pandas as pd
import requests
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

whisper_model = whisper.load_model("base")

movies = pickle.load(open("movies.pkl", "rb"))
movies_dict = pickle.load(open("movies_dict.pkl", "rb"))
movies_df = pd.DataFrame(movies_dict)
similarity = pickle.load(open("similarity.pkl", "rb"))

def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=a1699cc285d421a97e13c0a34d478cd0&language=en-US"
        response = requests.get(url)
        data = response.json()
        poster_path = data.get('poster_path', None)
        if poster_path:
            return f"https://image.tmdb.org/t/p/w500/{poster_path}"
        else:
            return "https://via.placeholder.com/500x750?text=No+Image"
    except:
        return "https://via.placeholder.com/500x750?text=Error"

def recommend_by_genre(title):
    title = title.lower()
    matching_movies = movies_df[movies_df['title'].apply(lambda g: title in g.lower())]
    if matching_movies.empty:
        return [], []
    top_movies = matching_movies.sample(min(5, len(matching_movies)))
    titles = top_movies['title'].tolist()
    posters = [fetch_poster(mid) for mid in top_movies['movie_id']]
    return titles, posters

def recommend_similar(movie_name):
    try:
        index = movies_df[movies_df['title'] == movie_name].index[0]
        distances = similarity[index]
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
        titles = [movies_df.iloc[i[0]].title for i in movie_list]
        posters = [fetch_poster(movies_df.iloc[i[0]].movie_id) for i in movie_list]
        return titles, posters
    except:
        return ["Couldn't find similar movies."], []

def record_and_transcribe(duration=4, samplerate=16000):
    status_placeholder = st.empty()
    status_placeholder.info("🎙️ Listening... Please speak now...")
    try:
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, recording, samplerate)
            audio_path = f.name
        result = whisper_model.transcribe(audio_path, language="en")
        os.remove(audio_path)
        status_placeholder.empty()
        return result["text"]
    except Exception as e:
        status_placeholder.empty()
        return f"❌ Error during transcription: {e}"

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, HF_TOKEN):
    return HuggingFaceEndpoint(
        repo_id=huggingface_repo_id,
        temperature=0.5,
        task="text-generation",
        model_kwargs={"token": HF_TOKEN, "max_length": "512"}
    )

def main():
    st.title("🎥 AI Chatbot + Movie Recommender")

    def set_bg():
        st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://static.vecteezy.com/system/resources/previews/037/756/557/non_2x/ai-generated-abstract-3d-concrete-cube-background-with-neon-lights-generative-ai-free-photo.jpg");
            background-size: cover;
            background-attachment: fixed;
            background-position: center;
            background-repeat: no-repeat;
            color: #f5f5f5 !important;
        }}
        .stMarkdown, .stTextInput, .stButton, .stCaption, .stChatMessage, .stChatMessage p, .stChatInput input {{
            color: #f5f5f5 !important;
        }}
        section[data-testid="stSidebar"] {{
            color: #f5f5f5 !important;
        }}
        .stChatInput input {{
            background-color: #1e1e1e;
            border: 1px solid #f5f5f5;
            color: #ffffff;
        }}
        button[kind="secondary"] {{
            background-color: #333;
            color: #fff;
            border: 1px solid #f5f5f5;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    set_bg()

    st.markdown(
        "<h2 style='color: #f0f0f0;'>Chat or speak your queries. Ask anything... or get movie recommendations!</h2>",
        unsafe_allow_html=True
    )

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    col1, col2 = st.columns([3, 1])
    with col1:
        user_text_input = st.chat_input("Type your prompt or use the mic...")

    with col2:
        mic_clicked = st.button("🎤 Speak")

    if mic_clicked:
        user_text_input = record_and_transcribe()

    if user_text_input:
        st.chat_message('user').markdown(user_text_input)
        st.session_state.messages.append({'role': 'user', 'content': user_text_input})

        if st.session_state.get("expecting_genre", False):
            genre = user_text_input.lower()
            titles, posters = recommend_by_genre(genre)
            st.chat_message('assistant').markdown(f"🎬 Recommending {genre.title()} movies:")
            cols = st.columns(len(titles))
            for idx, col in enumerate(cols):
                with col:
                    st.image(posters[idx])
                    st.caption(titles[idx])
            st.session_state.expecting_genre = False
            return

        if "recommend me movies like" in user_text_input.lower():
            movie_name = user_text_input.lower().replace("recommend me movies like", "").strip().title()
            titles, posters = recommend_similar(movie_name)
            if titles:
                st.chat_message('assistant').markdown(f"🎞️ If you liked **{movie_name}**, try these:")
                cols = st.columns(len(titles))
                for idx, col in enumerate(cols):
                    with col:
                        st.image(posters[idx])
                        st.caption(titles[idx])
            return

        CUSTOM_PROMPT_TEMPLATE = """
        Use the pieces of information provided in the context to answer user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer. 
        Don't provide anything out of the given context.

        Context: {context}
        Question: {question}

        Start the answer directly. No small talk please.
        """

        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            vectorstore = get_vectorstore()
            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': user_text_input})
            result = response["result"]

            if "Let's Watch Movies Together!!" in result.lower():
                st.session_state.expecting_genre = True
                result += "\n🎬 I can suggest movies too! What genre do you like?"
            else:
                st.session_state.expecting_genre = False

            st.chat_message('assistant').markdown(result)
            st.session_state.messages.append({'role': 'assistant', 'content': result})

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

with st.sidebar:
    st.title("💬 About")
    st.write("""
    This is an AI-powered chatbot that understands documents and recommends movies.

    🗂️ Ask questions from your PDFs  
    🎙️ Speak your queries  
    🎬 Get content-based and genre-based movie suggestions!
    """)

if __name__ == "__main__":
    main()
