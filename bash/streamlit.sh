# streamlit run src/scholar_sense/apps/vectordb_app.py -- \
#     --backend qdrant \
#     --topK  15 \
#     --model-type open-ai \
#     --model-name text-embedding-ada-002 \
#     --collection-name papers_openai \

streamlit run src/scholar_sense/apps/app.py -- \
    --backend simple \
    --topK  15 \
    --model-type sentence-transformers \
    --model-name roberta-large-nli-stsb-mean-tokens \
    --encoding-method title \
    --csv-file-path artifacts/data/csv/arxiv.csv \
    --embeddings-file-path artifacts/embeddings/arxiv_embeddings.pkl
