import base64
import pickle

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer


def read_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[["title", "abstract", "pdf_url", "created"]]
    return df


def get_embeddings_data(path: str) -> np.array:
    with open(path, "rb") as f:
        embeddings = pickle.load(f)
    return embeddings


def get_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    model = SentenceTransformer(model_name)
    return model


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
