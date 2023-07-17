from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def plot_length_distribution(df, column_name) -> None:
    """Plot the distribution of the length of a specified column of a DataFrame.

    The function calculates the mean and standard deviation of the column, and plots the
    distribution of the column values. It also plots the mean, and one standard deviation
    above and below the mean.

    Parameters
    ----------
        df (pd.DataFrame): The DataFrame containing the text data.
        column_name (str): The column of the DataFrame to analyze.

    Returns
    -------
        None. The function shows a plot.
    """
    # Calculate the mean and standard deviation of the column
    mean = df[column_name].mean()
    std = df[column_name].std()

    # Plot the distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x=column_name, color="cornflowerblue")

    # Plot the mean, and one standard deviation above and below the mean
    plt.axvline(mean, color="red", linestyle="dashed", linewidth=2)
    plt.axvline(mean + std, color="red", linestyle="dashed", linewidth=2)
    plt.axvline(mean - std, color="red", linestyle="dashed", linewidth=2)

    # Remove top and right spines
    sns.despine()

    plt.title(f"Distribution of {column_name} Length")
    plt.show()


def plot_top_words(df, column_name) -> None:
    """Plot the top 10 words in a specified column of a DataFrame.

    The function tokenizes the text in the specified column, converts the tokens to lower
    case, removes non-alphabetic tokens, removes stop words, and stems the words. It then
    counts the frequency of each word, and plots the top 10 words.

    Parameters
    ----------
        df (pd.DataFrame): The DataFrame containing the text data.
        column_name (str): The column of the DataFrame to analyze.

    Returns
    -------
        None. The function shows a plot.
    """
    # Tokenize the text
    tokens = df[column_name].apply(word_tokenize)

    # Convert to lower case
    tokens = tokens.apply(lambda x: [token.lower() for token in x])

    # Remove non-alphabetic tokens
    tokens = tokens.apply(lambda x: [token for token in x if token.isalpha()])

    # Remove stop words
    stop_words = stopwords.words("english")
    tokens = tokens.apply(lambda x: [token for token in x if token not in stop_words])

    # stem words
    stemmer = PorterStemmer()
    tokens = tokens.apply(lambda x: [stemmer.stem(token) for token in x])

    # Count the frequency of each word
    word_counts = Counter([token for sublist in tokens.tolist() for token in sublist])

    # Create dataframe
    df_word_counts = pd.DataFrame.from_dict(word_counts, orient="index").reset_index()
    df_word_counts.columns = ["Word", "Count"]

    # Sort by count and take the top 10
    df_word_counts_top10 = df_word_counts.sort_values("Count", ascending=False).head(10)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Count", y="Word", data=df_word_counts_top10, color="cornflowerblue")

    # Remove top and right spines
    sns.despine()

    plt.title(f"Top 10 Words in {column_name}")
    plt.show()
