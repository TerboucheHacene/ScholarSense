import argparse
import os
from collections import defaultdict
from typing import Union

import streamlit as st
from docarray.index import QdrantDocumentIndex
from docarray.index.abstract import BaseDocIndex
from docarray.index.backends.in_memory import InMemoryExactNNIndex

from scholar_sense.apps.constants import ABOUT, BACKGROUND_URL_IMAGE, HOW_TO_USE, MADE_BY
from scholar_sense.apps.utils import add_bg_from_url
from scholar_sense.data.enums import (
    EncodingMethod,
    IndexingMethod,
    ModelType,
    OpenAIModel,
    SentenceTransformersModel,
    SimpleEncodingMethod,
)
from scholar_sense.data.indexing import SimpleIndexer
from scholar_sense.data.schemas import DocPaper
from scholar_sense.nn import embedding_models
from scholar_sense.nn.models import EmbeddingModel


def main(
    doc_index: BaseDocIndex,
    embedding_model: EmbeddingModel,
    topK: int,
):
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
            st.session_state["paper_recommendations"] = defaultdict(lambda: [])
            st.session_state["order_indices"] = []
        else:
            # Clear previous results before adding new ones
            st.session_state["paper_recommendations"] = defaultdict(lambda: [])
            st.session_state["order_indices"] = []
            query_embedding = embedding_model.encode_sentence(query)
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


def run_app(
    backend: IndexingMethod,
    model_type: ModelType,
    model_name: Union[OpenAIModel, SentenceTransformersModel],
    encoding_method: Union[EncodingMethod, SimpleEncodingMethod],
    topK: int,
    collection_name: str = None,
    index_file_path: str = None,
    csv_file_path: str = None,
    embedding_file_path: str = None,
):
    embedding_model = embedding_models[model_type](
        model_name=model_name, encoding_method=encoding_method
    )
    if backend == IndexingMethod.IN_MEMORY:
        doc_index = InMemoryExactNNIndex[DocPaper](index_file_path=index_file_path)
    elif backend == IndexingMethod.QDRANT:
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

    elif backend == IndexingMethod.SIMPLE:
        doc_index = SimpleIndexer(
            model_type=model_type, model_name=model_name, encoding_method=encoding_method
        )
        doc_index.df = csv_file_path
        doc_index.embeddings = embedding_file_path
        embedding_model = embedding_models[model_type](
            model_name=model_name, encoding_method=encoding_method
        )

    else:
        raise ValueError(f"Unknown backend {backend}")
    main(doc_index=doc_index, embedding_model=embedding_model, topK=topK)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        type=str,
        default="qdrant",
        help="Backend to use for the index. Either `in_memory` or `qdrant`",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=15,
        help="Number of results to return",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="sentence-transformers",
        help="Type of model to use for the embeddings",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Name of the model to use for the embeddings",
    )
    parser.add_argument(
        "--encoding-method",
        type=str,
        default="title",
        help="Encoding method to use for the embeddings",
    )
    parser.add_argument(
        "--collection-name",
        type=str,
        required=False,
        help="Name of the collection to use for the index",
    )
    parser.add_argument(
        "--index-file-path",
        type=str,
        required=False,
        help="Path to the index file to use for the index",
    )
    parser.add_argument(
        "--csv-file-path",
        type=str,
        required=False,
        help="Path to the csv file contaning the data",
    )
    parser.add_argument(
        "--embeddings-file-path",
        type=str,
        required=False,
        help="Path to the embeddings pickle file",
    )
    args = parser.parse_args()
    return args


def check_args(args: argparse.Namespace) -> argparse.Namespace:
    try:
        args.backend = IndexingMethod[args.backend.upper()]
    except KeyError:
        raise ValueError(f"Unknown backend {args.backend} for the index")
    try:
        args.model_type = ModelType[args.model_type.upper()]
    except KeyError:
        raise ValueError(f"Unknown model type {args.model_type}")
    try:
        args.model_name = args.model_type.value[args.model_name.upper()]
    except KeyError:
        raise ValueError(f"Unknown model name {args.model_name} for {args.model_type}")
    try:
        if args.backend == IndexingMethod.SIMPLE:
            args.encoding_method = SimpleEncodingMethod[args.encoding_method.upper()]
        else:
            args.encoding_method = EncodingMethod[args.encoding_method.upper()]
    except KeyError:
        raise ValueError(f"Unknown encoding method {args.encoding_method}")
    return args


if __name__ == "__main__":
    args = parse_args()
    for arg in vars(args):
        if isinstance(getattr(args, arg), str):
            setattr(args, arg, getattr(args, arg).lower().replace("-", "_"))
    args = check_args(args)

    run_app(
        backend=args.backend,
        topK=args.limit,
        model_type=args.model_type,
        model_name=args.model_name,
        encoding_method=args.encoding_method,
        collection_name=args.collection_name,
        index_file_path=args.index_file_path,
        csv_file_path=args.csv_file_path,
        embedding_file_path=args.embeddings_file_path,
    )
