poetry run python scripts/indexpy \
    --db_path artifacts/data/json/ \
    --model_type "open-ai" \
    --model_name "text-embedding-ada-002" \
    --encoding "title" \
    --collection_name "papers_openai" \

# poetry run python scripts/create_index.py \
#     --db_path artifacts/data/json/ \
#     --model_name "all-MiniLM-L6-v2" \
#     --encoding "title" \
#     --index_file_path artifacts/data/index/docs.bin \
