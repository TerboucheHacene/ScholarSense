import os
from pathlib import Path

import streamlit as st
import torch
from sentence_transformers import util

from scholar_sense.utils import (
    add_bg_from_url,
    get_embeddings_data,
    get_model,
    read_data,
)

BACKGROUND_URL_IMAGE = "https://images.unsplash.com/photo-1531346878377-a5be20888e57?ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D&autot=format"  # noqa
ABOUT = """
# About
ScholarSense is a tool that helps you find relevant papers to read based on your interests.
It uses the [SPECTER](https://github.com/allenai/specter/tree/master) model to calculate the similarity between your query and the papers' abstracts.
The similarity is calculated using the cosine similarity between the embeddings of the query and the abstracts.
"""  # noqa

HOW_TO_USE = """
# How does it work
To use ScholarSense, you need to enter a query in the text box below.
The query can be a sentence or a paragraph.
For example, if you are interested in papers about transformers applied to object detection, you can enter the following query:

" transformers applied to object detection "
"""  # noqa

MADE_BY = """Made by [hacene-terbouche](https://github.com/TerboucheHacene) with ❤️"""


def main(
    db_path: str,
    topk: int,
    model_name: str,
    device: torch.device,
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

    embeddings = get_embeddings_data(os.path.join(db_path, "arxiv_embeddings.pkl"))
    df = read_data(os.path.join(db_path, "arxiv.csv"))
    model = get_model(model_name=model_name)

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
            query_embedding = model.encode(
                query, convert_to_tensor=True, device=device, normalize_embeddings=True
            )
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "all-MiniLM-L6-v2"
    main(db_path=db_path, model_name=model_name, device=device, topk=20)
