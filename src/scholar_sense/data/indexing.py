import logging
import os
import pickle
from abc import ABC, abstractmethod
from typing import List

import pandas as pd
import torch
from docarray import DocVec
from docarray.index import QdrantDocumentIndex
from docarray.index.abstract import BaseDocIndex
from docarray.index.backends.in_memory import InMemoryExactNNIndex

from scholar_sense.data.schemas import DocPaper
from scholar_sense.nn.models import (
    OpenAIEmbeddingModel,
    SentenceTransformerEmbeddingModel,
)


class BaseIndexer(ABC):
    def __init__(
        self,
        db_path: str,
        use_openai: bool,
        model_name: str,
        encoding_method: str = None,
        **kwargs,
    ):
        self.db_path = db_path
        if use_openai:
            self.embedding_model = OpenAIEmbeddingModel(model_name=model_name)
        else:
            self.embedding_model = SentenceTransformerEmbeddingModel(
                model_name=model_name, encoding_method=encoding_method, **kwargs
            )

    def run(self, **kwargs) -> BaseDocIndex:
        docs = self.create_doc_vec()
        docs = self.embedding_model.encode(docs)
        docs_index = self.create_index(docs, **kwargs)
        return docs_index

    def create_doc_vec(self) -> DocVec[DocPaper]:
        json_files = [f for f in os.listdir(self.db_path) if f.endswith(".json")]
        if len(json_files) == 0:
            raise ValueError(f"No JSON files found in {self.db_path}")
        logging.info(f"Found {len(json_files)} JSON files in {self.db_path}")
        docs = DocVec[DocPaper](
            docs=[DocPaper.from_json(f"{self.db_path}/{f}") for f in json_files]
        )
        return docs

    @abstractmethod
    def create_index(self, docs: DocVec[DocPaper], **kwargs):
        raise NotImplementedError("This method should be implemented in a child class.")


class InMemoryIndexer(BaseIndexer):
    """Class to index the papers in the database.

    Attributes
    ----------
    db_path : str
        Path to the directory containing the JSON files of the papers.
    model_name : str
        Name of the model to use for embedding the papers.
    encoding_method : str
        Method to use for encoding the papers.
    embedding_model : SentenceTransformerEmbeddingModel
        Instance of the SentenceTransformerEmbeddingModel class.
    """

    def __init__(
        self,
        db_path: str,
        use_openai: bool,
        model_name: str,
        encoding_method: str,
        **kwargs,
    ):
        super().__init__(db_path, use_openai, model_name, encoding_method, **kwargs)

    def create_index(
        self, index_file_path: str, docs: DocVec[DocPaper]
    ) -> InMemoryExactNNIndex:
        if os.path.exists(index_file_path):
            logging.warning(f"Index file {index_file_path} already exists. Deleting...")
            os.remove(index_file_path)
        docs_index = InMemoryExactNNIndex(index_file_path=index_file_path)
        docs_index.index(docs)
        docs_index.persist()
        return docs_index


class QdrantIndexer(BaseIndexer):
    def __init__(
        self,
        db_path: str,
        use_openai: bool,
        model_name: str,
        encoding_method: str,
        **kwargs,
    ):
        super().__init__(db_path, use_openai, model_name, encoding_method, **kwargs)

    def create_index(
        self,
        host: str,
        port: int,
        collection_name: str,
        docs: DocVec[DocPaper],
    ) -> QdrantDocumentIndex:
        qdarnt_config = QdrantDocumentIndex.DBConfig(
            host=host,
            port=port,
            collection_name=collection_name,
            default_column_config={
                "id": {},
                "vector": {"dim": self.embedding_model.embedding_size},
                "payload": {},
            },
        )

        docs_index = QdrantDocumentIndex[DocPaper](qdarnt_config)
        docs_index.index(docs)
        return docs_index


class Embedder:
    COLUMNS = ["title", "abstract"]
    ENCODING_METHODS = ["title", "abstract", "concat"]

    def __init__(self, model_name: str, encoding_method: str, **kwargs):
        self.embedding_model = SentenceTransformerEmbeddingModel(
            model_name=model_name, encoding_method=encoding_method, **kwargs
        )
        self.encoding_method = encoding_method

    def run(self, df_path: str, output_path: str):
        if os.path.exists(output_path):
            logging.warning(f"Output file {output_path} already exists. Deleting...")
            os.remove(output_path)
        df = self.read_data(df_path)
        text_to_encode = self.get_text_to_encode(df)
        embeddings = self.encode(text_to_encode)
        self.save(embeddings, output_path)

    def read_data(self, df_path) -> pd.DataFrame:
        df = pd.read_csv(df_path)
        for col in self.COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Column {col} not found in {df_path}")
        df = df[[col for col in self.COLUMNS]]
        return df

    def get_text_to_encode(self, df: pd.DataFrame) -> List[str]:
        text_to_encode = []
        for _, row in df.iterrows():
            if self.encoding_method == "title":
                text_to_encode.append(row["title"])
            elif self.encoding_method == "abstract":
                text_to_encode.append(row["abstract"])
            elif self.encoding_method == "concat":
                text_to_encode.append(f"{row['title']} [SEP] {row['abstract']}")
        return text_to_encode

    def encode(self, text_to_encode: List[str]) -> torch.Tensor:
        return self.embedding_model.encode_sentence(text_to_encode)

    @staticmethod
    def save(embeddings: torch.Tensor, output_path: str):
        with open(output_path, "wb") as f:
            pickle.dump(embeddings, f)

    @staticmethod
    def load(embeddings_path: str) -> torch.Tensor:
        with open(embeddings_path, "rb") as f:
            embeddings = pickle.load(f)
        return embeddings
