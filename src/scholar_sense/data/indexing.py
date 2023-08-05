import logging
import os

from docarray import DocVec
from docarray.index.backends.in_memory import InMemoryExactNNIndex

from scholar_sense.data.schemas import DocPaper
from scholar_sense.nn.models import EmbeddingModel


class Indexer:
    """Class to index the papers in the database.

    Attributes
    ----------
    db_path : str
        Path to the directory containing the JSON files of the papers.
    model_name : str
        Name of the model to use for embedding the papers.
    encoding_method : str
        Method to use for encoding the papers.
    embedding_model : EmbeddingModel
        Instance of the EmbeddingModel class.
    """

    ENCODING_METHODS = [
        "title",
        "abstract",
        "mean",
        "concat",
        "sliding_window_abstract",
        "sliding_window_mean",
    ]
    MODELS = [
        "all-MiniLM-L6-v2",
        "bert-base-nli-mean-tokens",
        "roberta-base-nli-mean-tokens",
        "distilbert-base-nli-mean-tokens",
        "distilbert-base-nli-stsb-mean-tokens",
        "roberta-base-nli-stsb-mean-tokens",
    ]

    def __init__(self, db_path: str, model_name: str, encoding_method: str, **kwargs):
        self.db_path = db_path
        if model_name not in self.MODELS:
            raise ValueError(f"Invalid model name. Must be one of {self.MODELS}")
        else:
            self.model_name = model_name
        if encoding_method not in self.ENCODING_METHODS:
            raise ValueError(
                f"Invalid encoding method. Must be one of {self.ENCODING_METHODS}"
            )
        else:
            self.encoding_method = encoding_method
        self.embedding_model = EmbeddingModel(model_name=self.model_name)

    def run(self, index_file_path: str):
        docs = self.create_doc_vec()
        docs = self.embed_data(docs)
        docs_index = self.create_index(index_file_path, docs)
        return docs_index

    def embed_data(self, docs: DocVec[DocPaper]) -> DocVec[DocPaper]:
        if self.encoding_method == "title":
            docs = self.embedding_model.encode_title(docs)
        elif self.encoding_method == "abstract":
            docs = self.embedding_model.encode_abstract(docs)
        elif self.encoding_method == "mean":
            docs = self.embedding_model.encode_mean(docs)
        elif self.encoding_method == "concat":
            docs = self.embedding_model.encode_concat(docs)
        elif self.encoding_method == "sliding_window_abstract":
            docs = self.embedding_model.encode_sliding_window_abstract(docs)
        elif self.encoding_method == "sliding_window_mean":
            docs = self.embedding_model.encode_sliding_window_mean(docs)
        return docs

    def create_doc_vec(self) -> DocVec[DocPaper]:
        json_files = [f for f in os.listdir(self.db_path) if f.endswith(".json")]
        if len(json_files) == 0:
            raise ValueError(f"No JSON files found in {self.db_path}")
        logging.info(f"Found {len(json_files)} JSON files in {self.db_path}")
        docs = DocVec[DocPaper](
            docs=[DocPaper.from_json(f"{self.db_path}/{f}") for f in json_files]
        )
        return docs

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
