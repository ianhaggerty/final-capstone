import textwrap
from textblob import TextBlob
from tabulate import tabulate


def preprocess(doc):
    """Preprocess the given spaCy Doc by lemmatizing tokens, removing stopwords and punctuation.

    Parameters:
        doc (spacy.tokens.Doc): The spaCy Doc object to preprocess.

    Returns:
        str: The preprocessed document text.
    """
    return ' '.join(
        [
            token.lemma_.lower() for token in doc
            if not token.is_stop and not token.is_punct
        ]
    )


def sentiment_analysis(review):
    """
    Performs sentiment analysis on the given review using TextBlob.

    Parameters:
        review (str): The review text to analyse.

    Returns:
        sentiment: A namedtuple containing the polarity and subjectivity scores.
    """
    blob = TextBlob(review)
    return blob.sentiment


def print_review(review):
    """Prints the review title, polarity, subjectivity 
    and text to the console in a formatted table.
    """
    table = [["Title",          review['reviews.title']],
             ["Polarity",       review['polarity']],
             ["Subjectivity",   review['subjectivity']],
             ["Text",  '\n'.join(textwrap.wrap(review['reviews.text'], width=70))]]
    print(tabulate(table))