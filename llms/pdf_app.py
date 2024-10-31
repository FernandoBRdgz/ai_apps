import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
import tempfile

# Cargar variables de entorno
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Inicializar la página de Streamlit
st.set_page_config(page_title="Chatbot de PDFs", page_icon="📄")
st.header("Asistente Virtual para PDFs")
st.sidebar.title("Opciones")

# Función para inicializar el estado de la sesión
def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "costs" not in st.session_state:
        st.session_state.costs = []

# Función para seleccionar el modelo de lenguaje
def select_model():
    model = st.sidebar.radio("Modelo:", ("gpt-3.5-turbo", "gpt-4"))
    temperature = st.sidebar.slider("Temperatura:", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    return ChatOpenAI(model_name=model, temperature=temperature)

# Función principal
def main():
    init_session_state()
    llm = select_model()

    # Cargar archivos PDF
    uploaded_files = st.sidebar.file_uploader("Sube tus archivos PDF", accept_multiple_files=True, type=['pdf'])
    if uploaded_files:
        all_text = ""
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_path = temp_file.name
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
            for doc in documents:
                all_text += doc.page_content
            os.remove(temp_file_path)

        # Procesar el texto de los PDFs
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_text(all_text)
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_texts(texts, embeddings)
        retriever = vector_store.as_retriever()
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memory)

        # Interfaz de chat
        if user_input := st.chat_input("Haz una pregunta sobre tus PDFs:"):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner("Generando respuesta..."):
                response = qa_chain({"question": user_input, "chat_history": st.session_state.messages})
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})

        # Mostrar historial de conversación
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.chat_message("user", avatar="👤").markdown(message["content"])
            else:
                st.chat_message("assistant", avatar="🤖").markdown(message["content"])

if __name__ == "__main__":
    main()
