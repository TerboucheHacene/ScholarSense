import argparse
import logging

from scholar_sense.data import indexers


def main(
    indexing_method: str,
    db_path: str,
    model_type: str,
    model_name: str,
    encoding_method: str,
    index_file_path: str = None,
    collection_name: str = None,
    host: str = "localhost",
    port: int = 6333,
):
    indexer_class = indexers[indexing_method]
    indexer = indexer_class(
        db_path=db_path,
        model_type=model_type,
        model_name=model_name,
        encoding_method=encoding_method,
    )
    if indexing_method == "qdrant":
        logging.info(f"Indexing papers from {args.db_path} to Qdrant")
        indexer.run(host=host, port=port, collection_name=collection_name)
    elif indexing_method == "in_memory":
        logging.info(f"Indexing papers from {args.db_path} to in-memory index")
        indexer.run(index_file_path=index_file_path)


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
    args_parser.add_argument(
        "--index_file_path",
        type=str,
        required=True,
        help="Path to the index file to be created",
    )

    return args_parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    main(
        db_path=args.db_path,
        use_openai=args.use_openai,
        model_name=args.model_name,
        encoding_method=args.encoding_method,
        collection_name=args.collection_name,
        index_file_path=args.index_file_path,
    )
