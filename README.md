# LangChain-based-psychological-wellness-RAG-bot
# 🤖 AI-Powered Multimodal Chatbot with Document Q&A and Movie Recommender System

This project is a **voice + text enabled conversational assistant** that combines the power of **Large Language Models (LLMs)**, **document understanding**, and **content-based movie recommendations** into a single Streamlit web application.

It is built with:
- 🧠 Mistral-7B-Instruct for natural language responses
- 📄 FAISS + HuggingFace Embeddings for vectorstore and semantic search over documents respectively 
- 🎬 TMDB API for movie retrieval based on keywords.
- 🎤 Whisper for voice input transcription
- 💻 Streamlit for a clean, user-friendly interface

---
## Tech Stack

### Programming Languages
- **Python**

### Libraries & Frameworks
- **Langchain**: For handling natural language processing (NLP) tasks and integrating LLMs.
- **Hugging Face**: For using the Mistral-7B-Instruct model and embeddings.
- **FAISS**: For vector storage and retrieval.
- **Streamlit**: For building the web interface and interactive features.
- **Whisper**: For speech-to-text transcription.
- **Sounddevice**: For recording audio from the microphone.
- **Pandas**: For data manipulation and handling movie data.
- **Requests**: For interacting with external APIs.
- **PyPDFLoader** and **DirectoryLoader**: For loading and processing PDF documents.

### Models
- **Mistral-7B-Instruct**: A large language model from Hugging Face used for text generation and answering questions.
- **Whisper**: A speech-to-text model used for transcribing audio input.

### APIs
- **The Movie Database (TMDB) API**: Used for fetching movie posters based on movie IDs.

### Data Storage
- **FAISS**: For storing and retrieving embeddings of documents and movie information.
---
## 📙Dataset Used 
- **The GALE ENCYCLOPEDIA of MEDICINE SECOND EDITION**
- **SELF MADE EMOTIONAL SUPPORT DATASET**
---

## 🧩 Project Structure 

### 📍 Phase 1: Creating memory for LLM 

#### 🔹 Objective:
Enable the chatbot to answer questions based on the content of uploaded documents (PDFs).

#### 🔹 Implementation:
- **Document Loading:** PDFs are loaded using `PyPDFLoader`.
- **Text Splitting:** Documents are split into overlapping chunks using `CharacterTextSplitter` for better semantic continuity.
- **Embedding Generation:** Each chunk is embedded using HuggingFace’s `all-MiniLM-L6-v2`.
- **Vector Indexing:** FAISS is used to index and store the embeddings for efficient retrieval.

#### 🔹 Prompt Template:
Ensures that answers remain based on document context and avoids hallucination by instructing the model to say “I don't know” if unsure.

---

### 📍 Phase 2: Connecting Memory with LLM 

#### 🔹 Objective:
Allows to connect LLM with FAISS and create QA chain

#### 🔹 Implementation:
- **Setup LLM:** Setting LLM (Mistral with HuggingFace)
- **Connect LLM+ VectorStore:** Connection of LLM and FAISS for data retrievel.
- **Loading Database:** Loading database for vectorstore directory.
- **QA Chain creation**
---

### 📍 Phase 2: Voice-to-Text Interaction

#### 🔹 Objective:
Allow users to interact with the chatbot via voice in addition to text.

#### 🔹 Implementation:
- Records audio using `sounddevice`.
- Transcribes audio using OpenAI’s **Whisper** model.
- Transcribed text is automatically sent to the chatbot for processing.

---

### 📍 Phase 4: Streamlit UI & Styling

#### 🔹 Objective:
Deliver a clean, user-friendly interface with multi-modal input support and added user engagement.

#### 🔹 Implementation:
- **Chat Interface:** Supports both text and voice input, and retains multi-turn history via `st.session_state`.
- **File Uploader:** For uploading PDFs used in document Q&A.
- **Custom Styling:** Background and input fields styled using inline CSS.
- **Dynamic Outputs:** Chat responses, transcribed audio, and movie posters rendered in real-time.
- **Integrated Movie Recommendation System:** Seamlessly embedded the movie recommendation model into the interface. If a user mentions feeling unwell or needs a break, the chatbot suggests relevant movies based on **keywords or Title** — enhancing user satisfaction and engagement.

---

### 📸 Project Screenshots

#### 🔹 Streamlit Chatbot Interface
![Streamlit Chat UI](https://github.com/user-attachments/assets/ee0928ee-6df0-4747-b7f1-69e4bf0c95d9)


## 🚀 Getting Started

### 🛠️ Prerequisites
- Python 3.9+
- Install dependencies:
```bash
pip install -r requirements.txt
