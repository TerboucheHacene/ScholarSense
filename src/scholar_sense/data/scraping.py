import arxiv
import pandas as pd
from tqdm import tqdm


def run_one_arxiv_query(query: str, client: arxiv.Client, max_results: int = 1000):
    """Run one query to arxiv API and return a dataframe with the results."""
    MAIN_CATEGORIES = ["cs.CV", "stat.ML", "cs.LG", "cs.AI"]
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    query_results = client.results(search=search)
    list_of_paper_dicts = [
        {
            "id": result.entry_id,
            "title": result.title,
            "abstract": result.summary,
            "authors": [author.name for author in result.authors],
            "categories": result.categories,
            "primary_category": result.primary_category,
            "doi": result.doi,
            "journal_reference": result.journal_ref,
            "pdf_url": result.pdf_url,
            "created": result.published,
            "updated": result.updated,
        }
        for result in query_results
        if result.primary_category in MAIN_CATEGORIES
    ]
    return list_of_paper_dicts


def scrape_arxiv(query_keywords: list, max_results: int = 1000):
    """Scrape arxiv for papers matching the query keywords."""
    client = arxiv.Client()
    list_of_paper_dicts = []
    for query in tqdm(query_keywords):
        list_of_paper_dicts += run_one_arxiv_query(query, client, max_results)
    df = pd.DataFrame(list_of_paper_dicts)
    return df
