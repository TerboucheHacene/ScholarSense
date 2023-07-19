import argparse

import yaml

from scholar_sense.data.scraping import scrape_arxiv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/configs.yaml")
    parser.add_argument("--output_path", type=str, default="artifacts/data/json")
    parser.add_argument("--max_results", type=int, default=6000)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        configs = yaml.safe_load(f)
    query_keywords = configs["keywords"]
    scrape_arxiv(
        query_keywords, output_path=args.output_path, max_results=args.max_results
    )


if __name__ == "__main__":
    main()
