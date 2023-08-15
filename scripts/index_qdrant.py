import argparse
import logging
import os

from scholar_sense.data.indexing import QdrantIndexer


def main(
    db_path: str,
    use_openai: bool,
    model_name: str,
    encoding_method: str,
    collection_name: str,
):
    indexer = QdrantIndexer(
        db_path=db_path,
        use_openai=use_openai,
        model_name=model_name,
        encoding_method=encoding_method,
    )
    host = os.getenv("QDRANT_HOST", "localhost")
    port = os.getenv("QDRANT_PORT", 6333)
    indexer.run(host=host, port=port, collection_name=collection_name)


def parse_args() -> argparse.Namespace:
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--db_path",
        type=str,
        required=True,
        help="Path to the directory containing the JSON files of the papers",
    )
    args_parser.add_argument(
        "--use_openai",
        action="store_true",
        help="Whether to use OpenAI API for encoding the papers",
    )
    args_parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to use for embedding the papers",
    )
    args_parser.add_argument(
        "--encoding_method",
        type=str,
        required=True,
        help="Method to use for encoding the papers",
    )
    args_parser.add_argument(
        "--collection_name",
        type=str,
        required=True,
        help="Name of the collection to use in Qdrant",
    )
    return args_parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    logging.info(f"Indexing papers from {args.db_path} to Qdrant")
    main(
        db_path=args.db_path,
        use_openai=args.use_openai,
        model_name=args.model_name,
        encoding_method=args.encoding_method,
        collection_name=args.collection_name,
    )
