from scholar_sense.data.enums import ModelType, ModelTypeCLI
from scholar_sense.nn.models import (
    OpenAIEmbeddingModel,
    SentenceTransformerEmbeddingModel,
)

embedding_models = {
    ModelType.OPEN_AI: OpenAIEmbeddingModel,
    ModelTypeCLI.OPEN_AI: OpenAIEmbeddingModel,
    ModelType.SENTENCE_TRANSFORMERS: SentenceTransformerEmbeddingModel,
    ModelTypeCLI.SENTENCE_TRANSFORMERS: SentenceTransformerEmbeddingModel,
}
