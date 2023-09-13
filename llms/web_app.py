import os
import requests
import streamlit as st

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from urllib.parse import urlparse

from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.callbacks import get_openai_callback

import langchain
langchain.verbose = False

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def init_page():
    st.set_page_config(
        page_title="Generador de resúmenes",
        page_icon="💻"
    )
    st.header("Generador de resúmenes")
    st.sidebar.title("Opciones")


def init_messages():
    clear_button = st.sidebar.button("Limpiar conversación", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]
        st.session_state.costs = []


def select_model():
    model = st.sidebar.radio("Modelo:", ("GPT-3.5", "GPT-4"))
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo"
    else:
        model_name = "gpt-4"

    return ChatOpenAI(temperature=0, model_name=model_name)


def get_url_input():
    url = st.text_input("URL: ", key="input")
    return url


def validate_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


def get_content(url):
    try:
        with st.spinner("Obteniendo contenido ..."):
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            if soup.main:
                return soup.main.get_text()
            elif soup.article:
                return soup.article.get_text()
            else:
                return soup.body.get_text()
    except:
        st.write('Ocurrió algo mal')
        return None


def build_prompt(content, n_chars=300):
    return f"""Aquí está el contenido de una página web. Proporcione un resumen conciso de alrededor de {n_chars} caracteres.

========

{content[:1000]}

"""


def get_answer(llm, messages):
    with get_openai_callback() as cb:
        answer = llm(messages)
    return answer.content, cb.total_cost


def main():
    init_page()

    llm = select_model()
    init_messages()

    container = st.container()
    response_container = st.container()

    with container:
        url = get_url_input()
        is_valid_url = validate_url(url)
        if not is_valid_url:
            st.write('Por favor ingresa una URL válida')
            answer = None
        else:
            content = get_content(url)
            if content:
                prompt = build_prompt(content)
                st.session_state.messages.append(HumanMessage(content=prompt))
                with st.spinner("El Asistente Virtual está escribiendo..."):
                    answer, cost = get_answer(llm, st.session_state.messages)
                st.session_state.costs.append(cost)
            else:
                answer = None

    if answer:
        with response_container:
            st.markdown("## Resumen")
            st.write(answer)
            st.markdown("---")
            st.markdown("## Texto Original")
            st.write(content)

    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Estimación de Costos")
    st.sidebar.markdown(f"Costo total en dólares: ${sum(costs):.5f}")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")

if __name__ == '__main__':
    main()