from scholar_sense.nn.models import (
    OpenAIEmbeddingModel,
    SentenceTransformerEmbeddingModel,
)

embedding_models = {
    "sentence-transformers": SentenceTransformerEmbeddingModel,
    "open-ai": OpenAIEmbeddingModel,
}
