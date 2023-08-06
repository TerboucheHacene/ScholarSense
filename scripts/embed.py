import argparse
import os

from scholar_sense.data.indexing import Embedder


def main(df_path: str, model_name: str, encoding_method: str, output_path: str):
    embedder = Embedder(model_name, encoding_method)
    embedder.run(df_path, output_path)


def parse_args() -> argparse.Namespace:
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument(
        "--df_path",
        type=str,
        required=True,
        help="Path to the CSV file containing the papers",
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
        "--output_path",
        type=str,
        required=True,
        help="Path to the file where the embeddings will be saved",
    )
    return args_parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if os.path.exists(args.output_path):
        raise ValueError(f"Output file {args.output_path} already exists")
    main(
        df_path=args.df_path,
        model_name=args.model_name,
        encoding_method=args.encoding_method,
        output_path=args.output_path,
    )
