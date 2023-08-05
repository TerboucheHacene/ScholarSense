poetry run python scripts/create_index.py \
    --db_path artifacts/data/json/ \
    --model_name "all-MiniLM-L6-v2" \
    --encoding "title" \
    --index_file_path artifacts/data/index/docs.bin \
