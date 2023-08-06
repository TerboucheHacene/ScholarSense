poetry run python scripts/embed.py \
    --df_path artifacts/data/csv/df.csv \
    --model_name "all-MiniLM-L6-v2" \
    --encoding "title" \
    --output_path artifacts/data/embeddings/embeddings.pkl
