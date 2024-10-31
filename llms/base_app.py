import os
import streamlit as st

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def main():
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=OPENAI_API_KEY, temperature=0)

    st.set_page_config(
        page_title="App Básica",
        page_icon="💻",
    )
    st.header("Asistente Virtual")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="You are a helpful assistant.")
        ]

    user_input = st.chat_input("Ingresa tu pregunta aquí:")
    if user_input:
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("El Asistente Virtual está escribiendo..."):
            response = llm.invoke(st.session_state.messages)  # Updated method
        st.session_state.messages.append(AIMessage(content=response.content))

    for message in st.session_state.messages:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant", avatar="🖥️"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user", avatar="👤"):
                st.markdown(message.content)
        else:
            st.write("¿En qué puedo ayudarte?")

if __name__ == '__main__':
    main()