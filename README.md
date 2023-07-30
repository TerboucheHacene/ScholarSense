# ScholarSense

![PaperWhiz-logo](./assets/ScholarSense.png)

# Table of contents
- [What is ScholarSense](#what-is-scholarsense)
- [How does it work](#how-does-it-work)
- [How to run it locally](#how-to-run-it-locally)

# What is ScholarSense
ScholarSense is a tool that helps you find relevant papers to read based on your interests.
It uses the [SPECTER](https://github.com/allenai/specter/tree/master) model to calculate the similarity between your query and the papers' abstracts.
The similarity is calculated using the cosine similarity between the embeddings of the query and the abstracts.

# How does it work
To use ScholarSense, you need to enter a query in the text box below.
The query can be a sentence or a paragraph.
For example, if you are interested in papers about transformers applied to object detection, you can enter the following query:
```
transformers applied to object detection
```
Then, you can click on the "Search" button to get the results.

# How to run it locally
To run ScholarSense locally, you need to install the virtual environment as well as all the dependencies using [poetry](https://python-poetry.org/) python package manager.
```
poetry install
```
To activate the virtual environment, you can run the following command:
```
poetry shell
```

Then, you can run the app using the following command:
```
poetry run streamlit run scripts/app.py
```

# Pipeline

- [x] Scarp scientific papers from Arxiv using [Arxiv API](https://info.arxiv.org/help/api/basics.html)
- [x] Exploratory data analysis is done to understand the data distribution, of both the titles and the abstracts
- [x] A [Sentence Transformer](https://www.sbert.net/index.html) Model is used to embed the papers by concatenating the title and the abstract into one sentence.
- [x] The embeddings are saved in a pickle file to be used later for the search
- [x] The search is done using the cosine similarity between the query and the embeddings of the papers
- [x] The results are displayed in a web app using [Streamlit](https://streamlit.io/)

# References
- [SPECTER](https://github.com/allenai/specter/tree/master)
- [Sentence Transformer](https://www.sbert.net/index.html)
- [Arxiv API](https://info.arxiv.org/help/api/basics.html)
- [Streamlit](https://streamlit.io/)
