import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
print(API_KEY)

st.write("Hello world. Let's learn how to build a AI-based app together.")