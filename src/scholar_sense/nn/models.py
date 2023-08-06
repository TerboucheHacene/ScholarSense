from typing import List, Union

import torch
import torch.nn as nn
from docarray import DocVec
from sentence_transformers import SentenceTransformer
from tqdm.autonotebook import trange

from scholar_sense.data.schemas import DocPaper


class EmbeddingModel(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-nli-mean-tokens",
        batch_size: int = 256,
        normalize_embeddings: bool = True,
        words_sequence_length: int = 300,
    ) -> None:
        super().__init__()
        self.model = SentenceTransformer(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.words_sequence_length = words_sequence_length

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
        # concat title and abstract

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
    def encode_sentences(self, sentences: Union[str, List[str]]) -> torch.Tensor:
        return self.model.encode(
            sentences=sentences,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            device=self.device,
            normalize_embeddings=self.normalize_embeddings,
        ).cpu()

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
