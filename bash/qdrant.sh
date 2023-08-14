poetry run python scripts/index_qdrant.py \
    --db_path artifacts/data/json/ \
    --model_name "all-MiniLM-L6-v2" \
    --encoding "title" \
    --collection_name "papers" \
