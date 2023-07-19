import argparse
import os

import pandas as pd

from scholar_sense.data.schemas import Paper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="artifacts/data/json")
    parser.add_argument("--output_path", type=str, default="artifacts/data/arxiv.csv")
    parser.add_argument("--sample_size", type=int, default=-1)
    return parser.parse_args()


def main():
    args = parse_args()
    json_files = os.listdir(args.input_path)
    if args.sample_size > 0:
        json_files = json_files[: args.sample_size]
    json_files = [os.path.join(args.input_path, file) for file in json_files]
    papers = []
    for json_file in json_files:
        paper = Paper.from_json(json_file)
        papers.append(paper.to_dict())
    df = pd.DataFrame(papers)
    df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
