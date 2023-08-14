import argparse
import os
from collections import defaultdict
from pathlib import Path

import streamlit as st
from docarray.index import QdrantDocumentIndex
from docarray.index.abstract import BaseDocIndex
from docarray.index.backends.in_memory import InMemoryExactNNIndex

from scholar_sense.apps.constants import ABOUT, BACKGROUND_URL_IMAGE, HOW_TO_USE, MADE_BY
from scholar_sense.apps.utils import add_bg_from_url
from scholar_sense.data.schemas import DocPaper
from scholar_sense.nn.models import EmbeddingModel


def main(doc_index: BaseDocIndex, embedding_model: EmbeddingModel, topK: int):
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

    if "paper_recommendations" not in st.session_state:
        st.session_state["paper_recommendations"] = defaultdict(lambda: [])
    if "order_indices" not in st.session_state:
        st.session_state["order_indices"] = []

    form = st.form(key="my_form")
    query = form.text_input(label="Enter you query here :")
    submit_button = form.form_submit_button(label="Submit")

    if submit_button:
        if query == "" or query == " " or len(query) < 3:
            st.error("Please enter a query")
            st.session_state["paper_recommendations"] = None
            st.session_state["order_indices"] = None
        else:
            # Clear previous results before adding new ones
            st.session_state["paper_recommendations"] = defaultdict(lambda: [])
            st.session_state["order_indices"] = []
            query_embedding = embedding_model.encode_sentences(query)
            items, _ = doc_index.find(
                query_embedding, search_field="embedding", limit=topK
            )
            for i, item in enumerate(items):
                st.session_state["paper_recommendations"]["title"].append(item.title)
                st.session_state["paper_recommendations"]["abstract"].append(
                    item.abstract
                )
                st.session_state["paper_recommendations"]["url"].append(item.pdf_url)
                st.session_state["paper_recommendations"]["created"].append(item.created)
                st.session_state["paper_recommendations"]["id"].append(item.id)
                # order
                st.session_state["order_indices"].append(i)

    if st.session_state["paper_recommendations"].get("title", None) is not None:
        order = st.checkbox("Order by date", key="order_by_date")
        if order:
            # sort by date
            st.session_state["order_indices"] = sorted(
                st.session_state["order_indices"],
                key=lambda x: st.session_state["paper_recommendations"]["created"][x],
                reverse=True,
            )
        else:
            st.session_state["order_indices"] = [*range(topK)]

        st.subheader("Here are some papers that might interest you:")
        for idx in st.session_state["order_indices"]:
            with st.expander(
                label=st.session_state["paper_recommendations"]["title"][idx]
            ):
                st.write(st.session_state["paper_recommendations"]["abstract"][idx])
                st.write(
                    f"Published on {st.session_state['paper_recommendations']['created'][idx]}"  # noqa
                )
                st.write(
                    f"Link to the paper: {st.session_state['paper_recommendations']['url'][idx]}"  # noqa
                )


def main_in_memory(topK: int):
    index_file_path = Path(
        "/home/hacene/Documents/workspace/ScholarSense/artifacts/embeddings/docs.bin"
    )
    model_name = "all-MiniLM-L6-v2"
    doc_index = InMemoryExactNNIndex[DocPaper](index_file_path=index_file_path)
    embedding_model = EmbeddingModel(model_name=model_name)
    main(doc_index=doc_index, embedding_model=embedding_model, topK=topK)


def main_qdrant(topK: int):
    model_name = "all-MiniLM-L6-v2"
    collection_name = "papers"
    embedding_model = EmbeddingModel(model_name=model_name)
    doc_index = QdrantDocumentIndex[DocPaper](
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
        collection_name=collection_name,
        default_column_config={
            "id": {},
            "vector": {"dim": embedding_model.embedding_size},
            "payload": {},
        },
    )
    main(doc_index=doc_index, embedding_model=embedding_model, topK=topK)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=str,
        default="qdrant",
        help="Backend to use for the index. Either `in_memory` or `qdrant`",
    )
    parser.add_argument(
        "--topK",
        type=int,
        default=15,
        help="Number of results to return",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.backend == "in_memory":
        main_in_memory(topK=args.topK)
    elif args.backend == "qdrant":
        main_qdrant(topK=args.topK)
    else:
        raise ValueError("Unknown backend")
