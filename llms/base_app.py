import os
import streamlit as st

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import (SystemMessage, HumanMessage, AIMessage)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def main():
    llm = ChatOpenAI(model_name="gpt-3.5-turbo-1106", openai_api_key=OPENAI_API_KEY, temperature=0)

    st.set_page_config(
        page_title="App Básica",
        page_icon="💻"
    )
    st.header("Asistente Virtual")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]

    # Monitor user input
    if user_input := st.chat_input("Ingresa tu pregunta aquí:"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("El Asistente Virtual está escribiendo..."):
            response = llm(st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=response.content))

    # Chat history
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

if __name__ == '__main__':
    main()