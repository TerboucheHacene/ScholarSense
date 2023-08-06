"""The module contains the basic app for ScholarSense.

In this first version, the app is very simple.
* The embeddings are precomputed and stored in a pickle file.
* The app loads the embeddings and the dataframe containing the papers' metadata.
* Then, it uses the Transformer model to calculate the similarity between the
query and the papers' abstracts using `util.cos_sim`.
* It then displays the top-k papers that are the most similar to the query using the
`torch.topk` function.
"""
import os
from pathlib import Path

import streamlit as st
import torch
from sentence_transformers import util

from scholar_sense.apps.constants import ABOUT, BACKGROUND_URL_IMAGE, HOW_TO_USE, MADE_BY
from scholar_sense.apps.utils import add_bg_from_url
from scholar_sense.data.indexing import Embedder


def main(
    db_path: str,
    topk: int,
    model_name: str,
) -> None:
    st.set_page_config(
        page_title="ScholarSense",
        page_icon="📚",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    html_temp = """
    <div style="background-color:grey;padding:10px">
    <h2 style="color:white;text-align:center;">ScholarSense</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    add_bg_from_url(BACKGROUND_URL_IMAGE)
    html_temp = """
    <div style="background-color:{};padding:10px;border-radius:10px">
    </div>
    """
    with st.sidebar:
        st.markdown(ABOUT, unsafe_allow_html=True)
        st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"), unsafe_allow_html=True)
        st.markdown(HOW_TO_USE, unsafe_allow_html=True)
        st.markdown(html_temp.format("rgba(55, 53, 47, 0.16)"), unsafe_allow_html=True)
        st.markdown(MADE_BY, unsafe_allow_html=True)

    embedder = Embedder(model_name, "title")
    df = embedder.read_data(os.path.join(db_path, "arxiv.csv"))
    embeddings = embedder.load(os.path.join(db_path, "arxiv_embeddings.pkl"))

    if "paper_recommendations" not in st.session_state:
        st.session_state["paper_recommendations"] = None
    if "order_indices" not in st.session_state:
        st.session_state["order_indices"] = None
    if "search_hits" not in st.session_state:
        st.session_state["search_hits"] = None

    form = st.form(key="my_form")
    query = form.text_input(label="Enter you query here :")
    submit_button = form.form_submit_button(label="Submit")

    if submit_button:
        if query == "" or query == " " or len(query) < 3:
            st.error("Please enter a query")
            st.session_state["paper_recommendations"] = None
            st.session_state["order_indices"] = None
        else:
            query_embedding = embedder.embedding_model.encode_sentences(query)
            cosine_scores = util.cos_sim(embeddings, query_embedding)
            search_hits = torch.topk(cosine_scores, dim=0, k=topk, sorted=True).indices
            search_hits = search_hits.cpu().numpy().squeeze()
            st.session_state["search_hits"] = search_hits
            st.session_state["paper_recommendations"] = df.iloc[search_hits].copy()

    if st.session_state["paper_recommendations"] is not None:
        order = st.checkbox("Order by date", key="order_by_date")
        if order:
            st.session_state["order_indices"] = (
                st.session_state["paper_recommendations"]
                .sort_values(by="created", ascending=False)
                .index
            )
        else:
            st.session_state["order_indices"] = st.session_state[
                "paper_recommendations"
            ].index
        st.subheader("Here are some papers that might interest you:")
        for _, row in (
            st.session_state["paper_recommendations"]
            .loc[st.session_state["order_indices"]]
            .iterrows()
        ):
            with st.expander(row["title"]):
                paper_abstract = row["abstract"]
                paper_url = row["pdf_url"]
                st.markdown(paper_abstract)
                st.markdown("Read the full paper here: " + paper_url)
                st.markdown("Published on: " + row["created"])


if __name__ == "__main__":
    db_path = Path("/home/hacene/Documents/workspace/ScholarSense/artifacts/data/")
    model_name = "roberta-large-nli-stsb-mean-tokens"
    main(db_path=db_path, model_name=model_name, topk=20)
