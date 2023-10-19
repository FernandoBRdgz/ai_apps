# Hasta la carga de PDF
import os
import streamlit as st

from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.callbacks import get_openai_callback
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

QDRANT_PATH = "./local_qdrant"
COLLECTION_NAME = "my_collection"


def init_page():
    st.set_page_config(
        page_title="Resúmenes de PDFs",
        page_icon="💻"
    )
    st.sidebar.title("Opciones")
    st.session_state.costs = []


def get_pdf_text():
    uploaded_file = st.file_uploader(
        label='Carga tu PDF aquí:',
        type='pdf'
    )
    if uploaded_file:
        pdf_reader = PdfReader(uploaded_file)
        text = '\n\n'.join([page.extract_text() for page in pdf_reader.pages])
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=st.session_state.emb_model_name,
            # The appropriate chunk size needs to be adjusted based on the PDF being queried.
            # If it's too large, it may not be able to reference information from various parts during question answering.
            # On the other hand, if it's too small, one chunk may not contain enough contextual information.
            chunk_size=250,
            chunk_overlap=0,
        )
        return text_splitter.split_text(text)
    else:
        return None


def load_qdrant():
    client = QdrantClient(path=QDRANT_PATH)

    # Get all collection names.
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    # If the collection does not exist, create it.
    if COLLECTION_NAME not in collection_names:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print('Colección creada')

    return Qdrant(
        client=client,
        collection_name=COLLECTION_NAME, 
        embeddings=OpenAIEmbeddings()
    )


def build_vector_store(pdf_text):
    qdrant = load_qdrant()
    qdrant.add_texts(pdf_text)

    # You can also do it like this. In this case, the vector database will be initialized every time.
    # Qdrant.from_texts(
    #     pdf_text,
    #     OpenAIEmbeddings(),
    #     path="./local_qdrant",
    #     collection_name=COLLECTION_NAME,
    # )


def page_pdf_upload_and_build_vector_db():
    st.title("Cargar PDF")
    container = st.container()
    with container:
        pdf_text = get_pdf_text()
        if pdf_text:
            with st.spinner("Cargando PDF ..."):
                build_vector_store(pdf_text)


def page_ask_my_pdf():
    st.title("Consultar PDF")
    st.write('Bajo Construcción')

    # To be implemented later.


def main():
    init_page()

    selection = st.sidebar.radio("Ir a:", ["Cargar PDF", "Consultar PDF"])
    if selection == "Cargar PDF":
        page_pdf_upload_and_build_vector_db()
    elif selection == "Consultar PDF":
        page_ask_my_pdf()

    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Estimación de Costos")
    st.sidebar.markdown(f"Costo total en dólares: ${sum(costs):.5f}")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")

if __name__ == '__main__':
    main()