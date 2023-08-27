streamlit run src/scholar_sense/apps/app.py -- \
    --backend qdrant \
    --limit  15 \
    --model-type open-ai \
    --model-name text-embedding-ada-002 \
    --collection-name papers_openai \

# streamlit run src/scholar_sense/apps/app.py -- \
#     --backend simple \
#     --limit  15 \
#     --model-type sentence_transformers \
#     --model-name roberta_large_nli_stsb_mean_tokens \
#     --encoding-method title \
#     --csv-file-path artifacts/data/csv/arxiv.csv \
#     --embeddings-file-path artifacts/embeddings/arxiv_embeddings.pkl
