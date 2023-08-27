import logging
import os
import time
from abc import ABC, abstractmethod, abstractproperty
from typing import List, Union

import numpy as np
import openai
import torch
from docarray import DocVec
from sentence_transformers import SentenceTransformer
from tqdm.autonotebook import trange

from scholar_sense.data.enums import (
    EncodingMethod,
    OpenAIModel,
    SentenceTransformersModel,
)
from scholar_sense.data.schemas import DocPaper


class EmbeddingModel(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractproperty
    def embedding_size(self):
        pass

    @abstractproperty
    def sequence_length(self):
        pass

    @abstractmethod
    def encode(self, docs: DocVec[DocPaper]) -> DocVec[DocPaper]:
        pass

    @abstractmethod
    def encode_sentence(self, text: str) -> torch.Tensor:
        pass


@EmbeddingModel.register
class OpenAIEmbeddingModel(EmbeddingModel):
    EMBEDDING_SIZES = {
        "text-embedding-ada-002": 1536,
        "text-search-davinci-001": 12288,
        "text-search-curie-001": 4096,
        "text-search-babbage-001": 2048,
        "text-search-ada-001": 1024,
    }
    SEQUENCE_LENGTHS = {
        "text-embedding-ada-002": 8191,
        "text-search-davinci-001": 2046,
        "text-search-curie-001": 2046,
        "text-search-babbage-001": 2046,
        "text-search-ada-001": 2046,
    }

    def __init__(
        self,
        model_name: OpenAIModel,
        batch_size: int = 128,
        max_tries: int = 5,
        **kwargs,
    ) -> None:
        self.model_name = model_name
        try:
            openai.api_key = os.getenv("OPENAI_API_KEY")
        except KeyError:
            raise KeyError(
                "Please set OPENAI_API_KEY environment variable to use OpenAI API."
            )
        try:
            self._embedding_size = self.EMBEDDING_SIZES[model_name.value]
        except KeyError:
            raise KeyError(
                f"Embedding size for model {model_name} not found. Please add it to the"
                "EMBEDDING_SIZES dictionary."
            )
        try:
            self._sequence_length = self.SEQUENCE_LENGTHS[model_name.value]
        except KeyError:
            raise KeyError(
                f"Sequence length for model {model_name} not found. Please add it to the"
                "SEQUENCE_LENGTHS dictionary."
            )
        self.batch_size = batch_size
        self.max_tries = max_tries

    @property
    def embedding_size(self):
        return self._embedding_size

    @property
    def sequence_length(self):
        return self._sequence_length

    def encode(self, docs: DocVec[DocPaper]) -> DocVec[DocPaper]:
        docs.embedding = torch.zeros((len(docs), self.embedding_size))
        logging.info("Encoding OpenAI")
        for i in trange(0, len(docs), self.batch_size, desc="Encoding OpenAI"):
            text = [
                *zip(
                    docs.title[i : i + self.batch_size],
                    docs.abstract[i : i + self.batch_size],
                )
            ]
            text = [t[0] + t[1] for t in text]
            for j in range(self.max_tries):
                try:
                    docs.embedding[i : i + self.batch_size] = self.encode_sentence(text)
                    break
                except Exception as e:
                    logging.error(f"Failed to encode batch: {e}")
                    time.sleep(10)
                    continue
            if j == self.max_tries - 1:
                logging.error("Failed to encode batch")
                break

            time.sleep(10)
        return docs

    def encode_sentence(self, text: Union[str, List]) -> torch.Tensor:
        if isinstance(text, str):
            text = [text]
        text = [t.replace("\n", " ") for t in text]
        data = openai.Embedding.create(input=text, model=self.model_name.value)["data"]
        emb = []
        for d in data:
            emb.append(np.array(d["embedding"]))
        emb = np.array(emb)
        emb = torch.from_numpy(emb)
        if emb.shape[0] == 1:
            emb = emb.squeeze(0)
        return emb


@EmbeddingModel.register
class SentenceTransformerEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model_name: SentenceTransformersModel,
        encoding_method: EncodingMethod,
        batch_size: int = 256,
        normalize_embeddings: bool = True,
        words_sequence_length: int = 300,
        **kwargs,
    ) -> None:
        super().__init__()
        self.encoding_method = encoding_method
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name.value)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.words_sequence_length = words_sequence_length
        self._embedding_size = self.model.get_sentence_embedding_dimension()
        self._sequence_length = self.model.get_max_seq_length()

    @property
    def embedding_size(self):
        return self._embedding_size

    @property
    def sequence_length(self):
        return self._sequence_length

    def encode(self, docs: DocVec[DocPaper]) -> DocVec[DocPaper]:
        if self.encoding_method == EncodingMethod.TITLE:
            docs = self.embedding_model.encode_title(docs)
        elif self.encoding_method == EncodingMethod.ABSTRACT:
            docs = self.embedding_model.encode_abstract(docs)
        elif self.encoding_method == EncodingMethod.MEAN:
            docs = self.embedding_model.encode_mean(docs)
        elif self.encoding_method == EncodingMethod.CONCAT:
            docs = self.embedding_model.encode_concat(docs)
        elif self.encoding_method == EncodingMethod.SLIDING_WINDOW_ABSTRACT:
            docs = self.embedding_model.encode_sliding_window_abstract(docs)
        elif self.encoding_method == EncodingMethod.SLIDING_WINDOW_MEAN:
            docs = self.embedding_model.encode_sliding_window_mean(docs)
        return docs

    @torch.no_grad()
    def encode_sentence(self, text: str) -> torch.Tensor:
        return self.model.encode(
            sentences=text,
            batch_size=1,
            show_progress_bar=False,
            convert_to_tensor=True,
            device=self.device,
            normalize_embeddings=self.normalize_embeddings,
        ).cpu()

    @torch.no_grad()
    def encode_mean(self, docs: DocVec[DocPaper]) -> DocVec[DocPaper]:
        docs.embedding = (
            (
                self.model.encode(
                    sentences=docs.title,
                    batch_size=self.batch_size,
                    show_progress_bar=True,
                    convert_to_tensor=True,
                    device=self.device,
                    normalize_embeddings=self.normalize_embeddings,
                )
                + self.model.encode(
                    sentences=docs.abstract,
                    batch_size=self.batch_size,
                    show_progress_bar=True,
                    convert_to_tensor=True,
                    device=self.device,
                    normalize_embeddings=self.normalize_embeddings,
                )
            )
            / 2
        ).cpu()
        return docs

    @torch.no_grad()
    def encode_concat(self, docs: DocVec[DocPaper]) -> DocVec[DocPaper]:
        docs.embedding = self.model.encode(
            sentences=[f"{t}[SEP]{a}" for t, a in zip(docs.title, docs.abstract)],
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=self.device,
            normalize_embeddings=self.normalize_embeddings,
        ).cpu()
        return docs

    @torch.no_grad()
    def encode_title(self, docs: DocVec[DocPaper]) -> DocVec[DocPaper]:
        docs.embedding = self.model.encode(
            sentences=docs.title,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=self.device,
            normalize_embeddings=self.normalize_embeddings,
        ).cpu()
        return docs

    @torch.no_grad()
    def encode_abstract(self, docs: DocVec[DocPaper]) -> DocVec[DocPaper]:
        docs.embedding = self.model.encode(
            sentences=docs.abstract,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=self.device,
            normalize_embeddings=self.normalize_embeddings,
        ).cpu()
        return docs

    @torch.no_grad()
    def encode_abstract_sliding_window(self, docs: DocVec[DocPaper]) -> DocVec[DocPaper]:
        abstract_embeddings = []
        for index in trange(0, len(docs.abstract), 1, desc="Encoding abstracts"):
            abstract = docs.abstract[index]
            abstract_embeddings.append(self.encode_sentence_sliding_window(abstract))
        docs.embedding = torch.stack(abstract_embeddings).cpu()
        return docs

    @torch.no_grad()
    def encode_sliding_window_mean(self, docs: DocVec[DocPaper]) -> DocVec[DocPaper]:
        abstract_embeddings = []
        for index in trange(0, len(docs.abstract), 1, desc="Encoding abstracts"):
            abstract = docs.abstract[index]
            abstract_embeddings.append(self.encode_sentence_sliding_window(abstract))
        abstract_embeddings = torch.stack(abstract_embeddings).cpu()
        title_embeddings = self.model.encode(
            sentences=docs.title,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=self.device,
            normalize_embeddings=self.normalize_embeddings,
        ).cpu()
        docs.embedding = (abstract_embeddings + title_embeddings) / 2
        return docs

    @torch.no_grad()
    def encode_sentence_sliding_window(self, sentences: str) -> torch.Tensor:
        num_words = len(sentences.split())
        chunks = [
            sentences.split()[i : i + self.words_sequence_length]
            for i in range(0, num_words, self.words_sequence_length)
        ]
        chunks = [" ".join(chunk) for chunk in chunks]
        return (
            self.model.encode(
                sentences=chunks,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_tensor=True,
                device=self.device,
                normalize_embeddings=self.normalize_embeddings,
            )
            .mean(dim=0)
            .cpu()
        )
