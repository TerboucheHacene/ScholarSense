import logging
import os
import pickle
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import pandas as pd
import torch
from docarray import DocVec
from docarray.index import QdrantDocumentIndex
from docarray.index.abstract import BaseDocIndex
from docarray.index.backends.in_memory import InMemoryExactNNIndex
from sentence_transformers import util

from scholar_sense.data.enums import (
    EncodingMethod,
    ModelType,
    OpenAIModel,
    SentenceTransformersModel,
)
from scholar_sense.data.schemas import DocPaper, SearchResult
from scholar_sense.nn.models import (
    OpenAIEmbeddingModel,
    SentenceTransformerEmbeddingModel,
)


class BaseIndexer(ABC):
    def __init__(
        self,
        db_path: str,
        model_type: ModelType,
        model_name: Union[OpenAIModel, SentenceTransformersModel],
        encoding_method: str = EncodingMethod,
        **kwargs,
    ):
        self.db_path = db_path
        self.model_type = model_type
        self.model_name = model_name
        self.encoding_method = encoding_method
        if model_type == ModelType.OPEN_AI:
            if model_name not in ModelType.OPEN_AI.value:
                raise ValueError(f"Unknown model name {model_name}")
            self.embedding_model = OpenAIEmbeddingModel(model_name=model_name)
        elif model_type == ModelType.SENTENCE_TRANSFORMERS:
            if model_name not in ModelType.SENTENCE_TRANSFORMERS.value:
                raise ValueError(f"Unknown model name {model_name}")
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
    def create_index(self, docs: DocVec[DocPaper], **kwargs) -> BaseDocIndex:
        raise NotImplementedError("This method should be implemented in a child class.")


@BaseIndexer.register
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
        model_type: ModelType,
        model_name: Union[OpenAIModel, SentenceTransformersModel],
        encoding_method: EncodingMethod,
        **kwargs,
    ):
        super().__init__(db_path, model_type, model_name, encoding_method, **kwargs)

    def create_index(
        self, docs: DocVec[DocPaper], index_file_path: str
    ) -> InMemoryExactNNIndex:
        if os.path.exists(index_file_path):
            logging.warning(f"Index file {index_file_path} already exists. Deleting...")
            os.remove(index_file_path)
        docs_index = InMemoryExactNNIndex(index_file_path=index_file_path)
        docs_index.index(docs)
        docs_index.persist()
        return docs_index


@BaseIndexer.register
class QdrantIndexer(BaseIndexer):
    def __init__(
        self,
        db_path: str,
        model_type: ModelType,
        model_name: Union[OpenAIModel, SentenceTransformersModel],
        encoding_method: EncodingMethod,
        **kwargs,
    ):
        super().__init__(db_path, model_type, model_name, encoding_method, **kwargs)

    def create_index(
        self,
        docs: DocVec[DocPaper],
        host: str,
        port: int,
        collection_name: str,
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


class SimpleIndexer:
    COLUMNS = ["id", "title", "abstract", "pdf_url", "created"]
    ENCODING_METHODS = ["title", "abstract", "concat"]

    def __init__(
        self,
        model_type: ModelType,
        model_name: Union[OpenAIModel, SentenceTransformersModel],
        encoding_method: EncodingMethod,
        **kwargs,
    ):
        self.model_type = model_type
        self.model_name = model_name
        self.encoding_method = encoding_method
        self._df = None
        self._embeddings = None

    @property
    def df(self):
        return self._df

    @df.setter
    def df(self, df_path: str):
        self._df = self.read_data(df_path)

    @property
    def embeddings(self):
        return self._embeddings

    @embeddings.setter
    def embeddings(self, emb_path: str):
        self._embeddings = self.load(emb_path)

    def run(self, df_path: str, output_path: str):
        if os.path.exists(output_path):
            logging.warning(f"Output file {output_path} already exists. Deleting...")
            os.remove(output_path)
        df = self.read_data(df_path)
        text_to_encode = self.get_text_to_encode(df)
        embeddings = self.encode(text_to_encode)
        self._embeddings = embeddings
        self.save(embeddings, output_path)

    def read_data(self, df_path) -> pd.DataFrame:
        df = pd.read_csv(df_path)
        for col in self.COLUMNS:
            if col not in df.columns:
                raise ValueError(f"Column {col} not found in {df_path}")
        df = df[[col for col in self.COLUMNS]]
        self._df = df
        return df

    def get_text_to_encode(self, df: pd.DataFrame) -> List[str]:
        text_to_encode = []
        for _, row in df.iterrows():
            if self.encoding_method == EncodingMethod.TITLE:
                text_to_encode.append(row["title"])
            elif self.encoding_method == EncodingMethod.ABSTRACT:
                text_to_encode.append(row["abstract"])
            elif self.encoding_method == EncodingMethod.CONCAT:
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

    def find(
        self, query: torch.Tensor, search_field: str, limit: int
    ) -> Tuple[List[int], List[float]]:
        if search_field != "embedding":
            raise ValueError(f"Search field {search_field} not supported.")
        if self._embeddings is None:
            raise ValueError("Embeddings not loaded.")
        if self._embeddings.device != query.device:
            query = query.to(self._embeddings.device)

        cosine_scores = util.cos_sim(self.embeddings, query)
        search_hits = torch.topk(cosine_scores, dim=0, k=limit, sorted=True)
        values = search_hits.values.cpu().numpy().squeeze().tolist()
        indices = search_hits.indices.cpu().numpy().squeeze().tolist()
        items = []
        for idx in indices:
            paper = SearchResult.from_dict(self.df.iloc[idx].to_dict())
            items.append(paper)

        return items, values
