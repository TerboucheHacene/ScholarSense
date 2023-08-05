import argparse
import os

from scholar_sense.data.indexing import Indexer


def main(db_path: str, model_name: str, encoding_method: str, index_file_path: str):
    indexer = Indexer(db_path, model_name, encoding_method)
    indexer.run(index_file_path)


def parse_args() -> argparse.Namespace:
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--db_path",
        type=str,
        required=True,
        help="Path to the directory containing the JSON files of the papers",
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
        "--index_file_path",
        type=str,
        required=True,
        help="Path to the index file to be created",
    )
    return args_parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if os.path.exists(args.index_file_path):
        raise ValueError(f"Index file {args.index_file_path} already exists")
    main(
        db_path=args.db_path,
        model_name=args.model_name,
        encoding_method=args.encoding_method,
        index_file_path=args.index_file_path,
    )
