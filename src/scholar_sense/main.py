import os
import subprocess
from typing import Optional

import pandas as pd
import typer
import yaml
from typing_extensions import Annotated

from scholar_sense.data import indexers
from scholar_sense.data.enums import (
    EncodingMethod,
    IndexingMethod,
    ModelNameCLI,
    ModelTypeCLI,
)
from scholar_sense.data.indexing import SimpleIndexer
from scholar_sense.data.schemas import Paper
from scholar_sense.data.scraping import scrape_arxiv

app = typer.Typer()


@app.command()
def scrape(
    config: Annotated[str, typer.Argument(help="Path to YAML config file")],
    output_path: Annotated[str, typer.Argument(help="Path to output directory")],
    max_results: Annotated[
        Optional[int], typer.Argument(help="Number of max results per query")
    ] = 1000000,
):
    """Scrape ArXiv using the given config file and saves the results to the given output directory."""  # noqa
    if not os.path.isfile(config):
        typer.echo(f"Config file {config} does not exist.")
        raise typer.Exit(code=1)
    if not os.path.isdir(output_path):
        typer.echo(f"Output path {output_path} does not exist.")
        raise typer.Exit(code=1)
    typer.echo("Scraping...")
    with open(config, "r") as f:
        configs = yaml.safe_load(f)
    query_keywords = [""] + configs["keywords"]
    scrape_arxiv(query_keywords, output_path=output_path, max_results=max_results)


@app.command()
def embed(
    input_path: Annotated[str, typer.Argument(help="Path to directory of JSON files")],
    output_path: Annotated[str, typer.Argument(help="Path to output pickle file")],
    csv_file_path: Annotated[str, typer.Argument(help="Path to CSV file")],
    model_type: Annotated[
        ModelTypeCLI, typer.Argument(help="Model type to use for the embeddings")
    ] = "sentence-transformers",
    model_name: Annotated[
        ModelNameCLI, typer.Argument(help="Model name of the chosen model type")
    ] = "all-MiniLM-L6-v2",
    encoding_method: Annotated[
        EncodingMethod, typer.Argument(help="Method used to encode papers")
    ] = "title",
    sample_size: Annotated[
        Optional[int], typer.Argument(help="Number of JSON files to consider")
    ] = -1,
):
    """Embed the JSON files in the given directory using the given model and saves the results to the given output file."""  # noqa
    if not os.path.exists(input_path):
        typer.echo(f"Input path {input_path} does not exist.")
        raise typer.Exit(code=1)
    if not os.path.isdir(os.path.dirname(output_path)):
        typer.echo(f"Output path {output_path} does not exist.")
        raise typer.Exit(code=1)
    if not os.path.basename(output_path).endswith(".pkl"):
        typer.echo(f"Output path {output_path} must be a pickle file.")
        raise typer.Exit(code=1)
    if not os.path.isdir(os.path.dirname(csv_file_path)):
        typer.echo(f"CSV file path {csv_file_path} does not exist.")
        raise typer.Exit(code=1)
    if not os.path.basename(csv_file_path).endswith(".csv"):
        typer.echo(f"CSV file path {csv_file_path} must be a CSV file.")
        raise typer.Exit(code=1)

    typer.echo("Embedding...")
    json_files = os.listdir(input_path)
    json_files = [file for file in json_files if file.endswith(".json")]
    if len(json_files) == 0:
        typer.echo(f"No JSON files found in {input_path}.")
        raise typer.Exit(code=1)

    if sample_size > 0:
        json_files = json_files[:sample_size]
    json_files = [os.path.join(input_path, file) for file in json_files]
    papers = []
    for json_file in json_files:
        paper = Paper.from_json(json_file)
        papers.append(paper.to_dict())
    df = pd.DataFrame(papers)
    df.to_csv(csv_file_path, index=False)
    embedder = SimpleIndexer(
        model_type=model_type,
        model_name=model_name,
        encoding_method=encoding_method,
    )
    embedder.run(df, output_path)


@app.command()
def index(
    db_path: Annotated[
        str, typer.Argument(help="Path to database file containing json papers")
    ],
    model_type: Annotated[
        ModelTypeCLI, typer.Argument(help="Model type to use for the embeddings")
    ] = "sentence-transformers",
    model_name: Annotated[
        ModelNameCLI, typer.Argument(help="Model name of the chosen model type")
    ] = "all-MiniLM-L6-v2",
    encoding_method: Annotated[
        EncodingMethod, typer.Argument(help="Method used to encode papers")
    ] = "title",
    indexing_method: Annotated[
        IndexingMethod, typer.Argument(help="Indexing method")
    ] = "in-memory",
    host: Annotated[str, typer.Option(help="Name of the host for Qdrant Index")] = None,
    port: Annotated[int, typer.Option(help="Port Number for Qdrant Index")] = None,
    collection_name: Annotated[
        str, typer.Option(help="Collection name for Qrant Index")
    ] = None,
    index_file_path: Annotated[
        str, typer.Option(help="Path to .bin index file for In-memory index")
    ] = None,
):
    """Indexethe papers in the given database using the given model and saves the results to the given output file."""  # noqa
    if not os.path.exists(db_path):
        typer.echo("db_path does not exist")
        raise typer.Exit(code=1)
    typer.echo("Indexing...")
    indexer_class = indexers[indexing_method]
    indexer = indexer_class(
        db_path=db_path,
        model_type=model_type,
        model_name=model_name,
        encoding_method=encoding_method,
    )
    if indexing_method == "qdrant":
        typer.echo(f"Indexing papers from {db_path} to Qdrant index")
        indexer.run(host=host, port=port, collection_name=collection_name)
    elif indexing_method == "in_memory":
        typer.echo(f"Indexing papers from {db_path} to in-memory index")
        indexer.run(index_file_path=index_file_path)


@app.command()
def streamlit(
    backend: Annotated[IndexingMethod, typer.Argument(help="Indexing method")],
    model_type: Annotated[
        ModelTypeCLI, typer.Argument(help="Model type to use for the embeddings")
    ],
    model_name: Annotated[
        ModelNameCLI, typer.Argument(help="Model name of the chosen model type")
    ],
    encoding_method: Annotated[
        EncodingMethod, typer.Argument(help="Method used to encode papers")
    ],
    limit: Annotated[int, typer.Argument(help="Number of results to return")],
    collection_name: Annotated[
        Optional[str], typer.Option(help="Name of the Qdrant collection")
    ] = None,
    index_file_path: Annotated[
        Optional[str], typer.Option(help="In-memory index file path")
    ] = None,
    csv_file_path: Annotated[
        Optional[str], typer.Option(help="Path to csv file containing data")
    ] = None,
    embedding_file_path: Annotated[
        Optional[str], typer.Option(help="Path to a pickled embedding file")
    ] = None,
):
    """Launch the Streamlit app."""
    if backend == IndexingMethod.SIMPLE:
        if not csv_file_path:
            typer.echo("csv_file_path is required for backend SIMPLE.")
            raise typer.Exit(code=1)
        if not os.path.exists(csv_file_path):
            typer.echo("csv_file_path must be a file.")
            raise typer.Exit(code=1)
        if not csv_file_path.endswith(".csv"):
            typer.echo("csv_file_path must be a CSV file.")
            raise typer.Exit(code=1)
        if not embedding_file_path:
            typer.echo("embedding_file_path is required for backend SIMPLE.")
            raise typer.Exit(code=1)
        if not os.path.exists(embedding_file_path):
            typer.echo("embedding_file_path must be a file.")
            raise typer.Exit(code=1)
        if not embedding_file_path.endswith(".pkl"):
            typer.echo("embedding_file_path must be a pickle file.")
            raise typer.Exit(code=1)

    elif backend == IndexingMethod.QDRANT:
        if not collection_name:
            typer.echo("collection_name is required for backend QDRANT.")
            raise typer.Exit(code=1)
    elif backend == IndexingMethod.IN_MEMORY:
        if not index_file_path:
            typer.echo("index_file_path is required for backend IN_MEMORY.")
            raise typer.Exit(code=1)

    typer.echo("Starting Streamlit...")

    # launch streamlit app
    command = [
        "streamlit",
        "run",
        "src/scholar_sense/apps/app.py",
        "--",
        "--backend",
        backend.value,
        "--model-type",
        model_type.value,
        "--model-name",
        model_name.value,
        "--encoding-method",
        encoding_method.value,
        "--limit",
        str(limit),
    ]
    if collection_name:
        command.extend(["--collection-name", collection_name])
    if index_file_path:
        command.extend(["--index-file-path", index_file_path])
    if csv_file_path:
        command.extend(["--csv-file-path", csv_file_path])
    if embedding_file_path:
        command.extend(["--embeddings-file-path", embedding_file_path])
    command = " ".join(command)
    typer.echo(command)
    subprocess.run(command, shell=True)
