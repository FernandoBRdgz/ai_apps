import os
import streamlit as st

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)
from langchain.callbacks import get_openai_callback

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def init_page():
    st.set_page_config(
        page_title="App Básica",
        page_icon="💻"
    )
    st.header("Asistente Virtual")
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
    temperature = st.sidebar.slider("Temperatura:", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    return ChatOpenAI(temperature=temperature, model_name=model_name)


def get_answer(llm, messages):
    with get_openai_callback() as cb:
        answer = llm(messages)
    return answer.content, cb.total_cost


def main():
    init_page()

    llm = select_model()
    init_messages()

    # Monitor user input
    if user_input := st.chat_input("Ingresa tu pregunta aquí:"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("El Asistente Virtual está escribiendo..."):
            answer, cost = get_answer(llm, st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=answer))
        st.session_state.costs.append(cost)

    messages = st.session_state.get('messages', [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message('assistant', avatar="🖥️"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message('user', avatar="👤"):
                st.markdown(message.content)
        else:
            st.write("¿En qué puedo ayudarte?")
            # st.write(st.session_state)

    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Estimación de Costos")
    st.sidebar.markdown(f"Costo total en dólares: ${sum(costs):.5f}")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")

if __name__ == '__main__':
    main()