from enum import Enum


class BaseEnum(Enum):
    @classmethod
    def values(cls):
        return [e.value for e in cls]

    @classmethod
    def names(cls):
        return [e.name for e in cls]


class SimpleEncodingMethod(BaseEnum):
    TITLE = "title"
    ABSTRACT = "abstract"
    CONCAT = "concat"


class EncodingMethod(BaseEnum):
    TITLE = "title"
    ABSTRACT = "abstract"
    MEAN = "mean"
    CONCAT = "concat"
    SLIDING_WINDOW_ABSTRACT = "sliding_window_abstract"
    SLIDING_WINDOW_MEAN = "sliding_window_mean"


class OpenAIModel(BaseEnum):
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"
    TEXT_SEARCH_DAVINCI_001 = "text-search-davinci-001"
    TEXT_SEARCH_CURIE_001 = "text-search-curie-001"
    TEXT_SEARCH_BABBAGE_001 = "text-search-babbage-001"
    TEXT_SEARCH_ADA_001 = "text-search-ada-001"


class SentenceTransformersModel(BaseEnum):
    ALL_MINILM_L6_V2 = "all-MiniLM-L6-v2"
    BERT_BASE_NLI_MEAN_TOKENS = "bert-base-nli-mean-tokens"
    ROBERTA_BASE_NLI_MEAN_TOKENS = "roberta-base-nli-mean-tokens"
    DISTILBERT_BASE_NLI_MEAN_TOKENS = "distilbert-base-nli-mean-tokens"
    DISTILBERT_BASE_NLI_STSB_MEAN_TOKENS = "distilbert-base-nli-stsb-mean-tokens"
    ROBERTA_BASE_NLI_STSB_MEAN_TOKENS = "roberta-base-nli-stsb-mean-tokens"
    ROBERTA_LARGE_NLI_STSB_MEAN_TOKENS = "roberta-large-nli-stsb-mean-tokens"


class ModelNameCLI(BaseEnum):
    # OpenAI
    TEXT_EMBEDDING_ADA_002 = "text-embedding-ada-002"
    TEXT_SEARCH_DAVINCI_001 = "text-search-davinci-001"
    TEXT_SEARCH_CURIE_001 = "text-search-curie-001"
    TEXT_SEARCH_BABBAGE_001 = "text-search-babbage-001"
    TEXT_SEARCH_ADA_001 = "text-search-ada-001"
    # SentenceTransformers
    ALL_MINILM_L6_V2 = "all-MiniLM-L6-v2"
    BERT_BASE_NLI_MEAN_TOKENS = "bert-base-nli-mean-tokens"
    ROBERTA_BASE_NLI_MEAN_TOKENS = "roberta-base-nli-mean-tokens"
    DISTILBERT_BASE_NLI_MEAN_TOKENS = "distilbert-base-nli-mean-tokens"
    DISTILBERT_BASE_NLI_STSB_MEAN_TOKENS = "distilbert-base-nli-stsb-mean-tokens"
    ROBERTA_BASE_NLI_STSB_MEAN_TOKENS = "roberta-base-nli-stsb-mean-tokens"
    ROBERTA_LARGE_NLI_STSB_MEAN_TOKENS = "roberta-large-nli-stsb-mean-tokens"


class ModelType(BaseEnum):
    OPEN_AI = OpenAIModel
    SENTENCE_TRANSFORMERS = SentenceTransformersModel


class ModelTypeCLI(BaseEnum):
    OPEN_AI = "open-ai"
    SENTENCE_TRANSFORMERS = "sentence-transformers"


class IndexingMethod(BaseEnum):
    QDRANT = "qdrant"
    IN_MEMORY = "in_memory"
    SIMPLE = "simple"
