import datetime
import os
from pathlib import Path
from typing import List

import arxiv
from tqdm import tqdm

from scholar_sense.data.schemas import Paper


def run_one_arxiv_query(
    query: str, output_path: Path, client: arxiv.Client, max_results: float = 1000
):
    """Run one query to arxiv API and return a dataframe with the results."""
    MAIN_CATEGORIES = ["cs.CV", "stat.ML", "cs.LG", "cs.AI"]
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    query_results = client.results(search=search)

    for result in query_results:
        if result.primary_category in MAIN_CATEGORIES:
            id = result.entry_id.rsplit("/", 1)[-1]
            if not os.path.exists(os.path.join(output_path, id + ".json")):
                paper = Paper(
                    id=result.entry_id,
                    title=result.title,
                    abstract=result.summary,
                    authors=[author.name for author in result.authors],
                    categories=result.categories,
                    primary_category=result.primary_category,
                    doi=result.doi,
                    journal_reference=result.journal_ref,
                    pdf_url=result.pdf_url,
                    created=result.published,
                    updated=result.updated,
                    obtained=datetime.date.today(),
                )
                paper.to_json(output_path)


def scrape_arxiv(query_keywords: List, output_path: Path, max_results: float = 1000):
    """Scrape arxiv for papers matching the query keywords."""
    client = arxiv.Client(num_retries=50, page_size=1000)
    for query in tqdm(query_keywords):
        query = '"' + query + '"'
        try:
            run_one_arxiv_query(query, output_path, client, max_results)
        except Exception as e:
            print(f"Query {query} failed with error {e}")
            continue
