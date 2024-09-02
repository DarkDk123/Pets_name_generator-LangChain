"""### Main `Streamlit` file, should be run using `streamlit run main.py`"""
# ________Imports__________
import langchain_helper as lch
import streamlit as st

import os


# Creating APP Info
st.title("Pets Name Generator 🐶")
st.markdown(
    f"""
A simple LangChain app to generate Pet name suggestions with given descriptions

It utilizes HuggingFace [`'{os.environ["HUGGINGFACE_MODEL_ID"]}'`](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) model."""
)

# Sidebar Appearance
st.sidebar.header("Select your pet info ")
pet_type = st.sidebar.selectbox(
    "Pet Type",
    [
        "Dog",
        "Friend",
        "Cat",
        "Cow",
    ],
)

pet_color = st.sidebar.text_input(
    f"What colour is your {pet_type}", placeholder="white like crystal"
)

pet_description = st.sidebar.text_area(
    "Description", placeholder="A small energetic dog!", max_chars=30
)

submit_btn = st.sidebar.button("Submit")  # Submit


# Calling the model to generate names!
if submit_btn:
    if not pet_color:
        st.error("Please Enter pet color! 🐒")
    else:
        st.subheader(f"Here are 5 names for your {pet_type}", divider="green")
        st.markdown(lch.generate_pet_name(pet_type, pet_color, pet_description))
        st.balloons()