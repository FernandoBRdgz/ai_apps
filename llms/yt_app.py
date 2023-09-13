import os
import streamlit as st

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import YoutubeLoader
from langchain.chains.summarize import load_summarize_chain


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def init_page():
    st.set_page_config(
        page_title="Resúmenes de Youtube",
        page_icon="💻"
    )
    st.header("Resúmenes de Youtube")
    st.sidebar.title("Opciones")
    st.session_state.costs = []


def select_model():
    model = st.sidebar.radio("Modelo:", ("GPT-3.5", "GPT-4"))
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo"
    else:
        model_name = "gpt-4"

    return ChatOpenAI(temperature=0, model_name=model_name)


def get_url_input():
    url = st.text_input("URL de Youtube: ", key="input")
    return url


def get_document(url):
    with st.spinner("Obteniendo contenido..."):
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=True,
            language=['en', 'es'] # Se obtienen los subtítulos en estos idiomas
        )
        return loader.load()


def summarize(llm, docs):
    prompt_template = """Write a concise summary of the following transcript of Youtube Video in spanish.
===
    
{text}

"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    with get_openai_callback() as cb:
        chain = load_summarize_chain( 
            llm,
            chain_type="stuff",
            verbose=True,
            prompt=PROMPT
        )
        response = chain({"input_documents": docs}, return_only_outputs=True)
        
    return response['output_text'], cb.total_cost


def main():
    init_page()
    llm = select_model()

    container = st.container()
    response_container = st.container()

    with container:
        if url := get_url_input():
            document = get_document(url)
            with st.spinner("El Asistente Virtual está escribiendo..."):
                output_text, cost = summarize(llm, document)
            st.session_state.costs.append(cost)
        else:
            output_text = None

    if output_text:
        with response_container:
            st.markdown("## Resumen")
            st.write(output_text)
            st.markdown("---")
            st.markdown("## Texto Original")
            st.write(document)

    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Estimación de Costos")
    st.sidebar.markdown(f"Costo total en dólares: ${sum(costs):.5f}")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")

if __name__ == '__main__':
    main()