import streamlit as st
import os
from Taskbot.minibot import calculate_expression, read_file

st.set_page_config(page_title="Langchain Taskbot", layout="centered")
st.title("Langchain Taskbot with Streamlit")

st.write("""
This is a simple Streamlit interface for your Langchain Taskbot. You can:
- Calculate math expressions
- Read a file from disk
""")

option = st.selectbox("Choose a tool:", ("Calculator", "Read File"))

if option == "Calculator":
    expr = st.text_input("Enter a math expression (e.g., sqrt(16) + log(10)):")
    if st.button("Calculate"):
        if expr:
            result = calculate_expression(expr)
            st.success(result)
        else:
            st.warning("Please enter an expression.")
elif option == "Read File":
    file_path = st.text_input("Enter the full file path:")
    if st.button("Read File"):
        if file_path:
            result = read_file(file_path)
            st.info(result)
        else:
            st.warning("Please enter a file path.")
