import os
import streamlit as st

from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
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
        st.session_state.model_name = "gpt-3.5-turbo-1106"
    else:
        st.session_state.model_name = "gpt-4-1106-preview"
    
    # 300: La cantidad de tokens para instrucciones fuera del texto principal
    # st.session_state.max_token = OpenAI.modelname_to_contextsize(st.session_state.model_name) - 300
    return ChatOpenAI(temperature=0, model_name=st.session_state.model_name)


def get_url_input():
    url = st.text_input("URL de Youtube: ", key="input")
    return url


def get_document(url):
    with st.spinner("Obteniendo contenido..."):
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=True,  # Es posible recuperar el título del video y el número de vistas
            language=['en', 'es']  # Se obtienen los subtítulos para estos idiomas
        )
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=st.session_state.model_name,
            # chunk_size=st.session_state.max_token,
            chunk_overlap=0,
        )
        return loader.load_and_split(text_splitter=text_splitter)



def summarize(llm, docs):
    prompt_template = """Write a concise summary of the following transcript of Youtube Video in spanish.
===
    
{text}

"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    with get_openai_callback() as cb:
        chain = load_summarize_chain(
            llm,
            chain_type="map_reduce",
            verbose=True,
            map_prompt=PROMPT,
            combine_prompt=PROMPT
        )
        response = chain(
            {
                "input_documents": docs,
                # Si no se especifica token_max, el procesamiento interno se ajustará
                # para adaptarse a los tamaños de modelos habituales como GPT-3.5
                # "token_max": st.session_state.max_token
            },
            return_only_outputs=True
        )
        
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