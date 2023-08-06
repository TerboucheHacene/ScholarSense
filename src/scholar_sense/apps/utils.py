import base64

import streamlit as st


def add_bg_from_url(url: str) -> None:
    st.markdown(
        f"""
         <style>
         .stApp {{
             background-image: url("{url}");
             background-attachment: fixed;
             background-size: cover;


         }}
         </style>
         """,
        unsafe_allow_html=True,
    )


def add_bg_from_local(image_file: str) -> None:
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
        f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )
