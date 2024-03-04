########################
### CAPSTONE PROJECT ###
########################

"""
1.  Implement a sentiment analysis model using spaCy: Load the
    en_core_web_sm spaCy model to enable natural language processing
    tasks. This model will help you analyse and classify the sentiment of the
    product reviews.
2.  Preprocess the text data: Remove stopwords, and perform any
    necessary text cleaning to prepare the reviews for analysis.
    2.1.    To select the 'review.text' column from the dataset and retrieve
            its data, you can simply use the square brackets notation. Here
            is the basic syntax:
            reviews_data = dataframe['review.text']
            This column, 'review.text,' represents the feature variable
            containing the product reviews we will use for sentiment
            analysis.
    2.2.    To remove all missing values from this column, you can simply
            use the dropna() function from Pandas using the following
            code:
            clean_data = dataframe.dropna(subset=['reviews.text'])
3.  Create a function for sentiment analysis: Define a function that takes
    a product review as input and predicts its sentiment.
4.  Test your model on sample product reviews: Test the sentiment
    analysis function on a few sample product reviews to verify its accuracy
    in predicting sentiment.
5.  Write a brief report or summary in a PDF file:
    sentiment_analysis_report.pdf that must include:
    5.1.    A description of the dataset used.
    5.2.    Details of the preprocessing steps.
    5.3.    Evaluation of results.
    5.4.    Insights into the model's strengths and limitations.
Additional Instructions:
●   Some helpful guidelines on cleaning text:
    ○   To remove stopwords, you can utilise the .is_stop attribute in spaCy.
        This attribute helps identify whether a word in a text qualifies as a
        stop word or not. Stopwords are common words that do not add
        much meaning to a sentence, such as "the", "is", and "of".
        Subsequently, you can then employ the filtered list of tokens or
        words(words with no stop words) for conducting sentiment analysis.
    ○   You can also make use of the lower(), strip() and str() methods to
        perform some basic text cleaning.
●   You can use the spaCy model and the .sentiment attribute to analyse the
    review and determine whether it expresses a positive, negative, or neutral
    sentiment. To use the .polarity attribute, you will need to install the
    TextBlob library. You can do this with the following commands:
■   # Install spacytextblob
■   pip install spacytextblob
    ○   Textblob requires additional data before getting started, download the data
        using the following code:
■   python -m textblob.download_corpora
    ○   Once you have installed TextBlob, you can use the .sentiment and
        .polarity attribute to analyse the review and determine whether it
        expresses a positive, negative, or neutral sentiment. You can also
        incorporate this code to get yourself started:
■   # Using the polarity attribute
■   polarity = doc._.blob.polarity
■   # Using the sentiment attribute
■   sentiment = doc._.blob.sentiment
    FYI: The underscore in the code just above is a Python convention for naming
    private attributes. Private attributes are not meant to be accessed directly by the
    user, but can be accessed through public methods.
●   You can use the .polarity attribute to measure the strength of the
    sentiment in a product review. A polarity score of 1 indicates a very positive
    sentiment, while a polarity score of -1 indicates a very negative sentiment. A
    polarity score of 0 indicates a neutral sentiment.
●   You can also use the similarity() function to compare the similarity of two
    product reviews. A similarity score of 1 indicates that the two reviews are
    more similar, while a similarity score of 0 indicates that the two reviews are
    not similar.
    ○   Choose two product reviews from the 'review.text' column and
        compare their similarity. To select a specific review from this column,
        simply use indexing, as shown in the code below:
        my_review_of_choice = data['reviews.text'][0]
    ○   The above code retrieves a review from the 'review.text' column at
        index 0. You can select two reviews of your choice using indexing.
"""

### Dependencies ###

# External.
import pandas as pd
import spacy
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import namedtuple
from wordcloud import WordCloud

# Internal.
from lib.util import preprocess, sentiment_analysis, print_review

# Import CSV file and parse into a pandas dataframe.
# NOTE It's tempting to use `id` column as the index here. However, this
#      is actually the id of an individual *user*. A *user* has many
#      product reviews, so their `id` is not unique within this dataset.
#      Moreover, the `reviews.id` is mostly NA, so instead, we use the
#      default rangeindex here.
reviews = pd.read_csv(Path(".", "data", "amazon_product_reviews.csv"))[0:10]

# Load the *small* spaCy NLP model.
nlp = spacy.load("en_core_web_sm")

# Remove rows where the review is absent.
# NOTE There doesn't appear to be any missing reviews for this dataset.
#      Nevertheless, this was suggested for this task.
reviews.dropna(subset=["reviews.text"], inplace=True)

# Perform NLP analysis on the 'reviews.text' column.
# Here, we store in it's own pandas series, so as to avoid recomputing.
review_docs = reviews["reviews.text"].map(lambda review: nlp(review))

# Process the review documents and store in a new series.
review_docs_processed = review_docs.map(
    lambda review_doc: preprocess(review_doc))

# Perform sentiment analysis and store as two new columns
# ["polarity", "subjectivity"] in the original dataframe.
# NOTE Here, `sentiments` is a series of namedtuples. We "unpack" the
#      namedtuples into their own series', before assigning back to the
#      original dataframe.
sentiments = review_docs_processed.map(
    lambda review: sentiment_analysis(review))
reviews["polarity"]     = sentiments.map(lambda s: s.polarity)
reviews["subjectivity"] = sentiments.map(lambda s: s.subjectivity)

# Create a similarity matrix for the first `similarity_dim` product reviews.
# NOTE Using the full array (5000 products) requires
#      4999(5000)/2 = 12,497,500 similarity comparisons.
#      This took ~ 40 minutes on my machine, though GPU optimisation
#      might be available. https://tqdm.github.io/
#      https://stackoverflow.com/a/75355418/1030067
similarity_dim = 10
similarity_df = pd.DataFrame(np.empty((similarity_dim, similarity_dim)))
for i in range(0, similarity_dim):
    similarity_df.iloc[i,i] = 1
    for j in range(i + 1, similarity_dim):
        similarity = review_docs.iloc[i].similarity(review_docs.iloc[j])
        similarity_df.iloc[i, j] = similarity
        similarity_df.iloc[j, i] = similarity

# Let's select the most dissimilar and similar reviews,
# amongst those that have been compared.
max = 0
min = 1
Indices = namedtuple("Indices", ["min", "max"])

for i in range(0, similarity_dim):
    for j in range(i + 1, similarity_dim):
        if (similarity_df.iloc[i,j]) < min:
            min = similarity_df.iloc[i,j]
            index_min = (i,j)
        if (similarity_df.iloc[i,j]) > max:
            max = similarity_df.iloc[i,j]
            index_max = (i,j)

indices = Indices(index_min, index_max)

# Print details of the most dissimilar and similar reviews,
# amongst those that have been compared.
print()
print(f"!! The Two Least Similar Reviews ({round(min, 2)}) !!")
print()
print_review(reviews.iloc[indices.min[0]])
print_review(reviews.iloc[indices.min[1]])

print()
print(f"!! The Two Most Similar Reviews ({round(max, 2)}) !!")
print()
print_review(reviews.iloc[indices.max[0]])
print_review(reviews.iloc[indices.max[1]])


# Create a wordcloud plot for the most dissimilar and similar reviews.
WC_config = WordCloud(background_color ='white')

for i in range(2):
    plt.figure(figsize=(20, 7))
    plt.axis("off")

    for j in range(2):
        words = review_docs_processed.iloc[indices[i][j]]
        wordcloud = WC_config.generate(words)

        plt.subplot(1, 2, j + 1)
        plt.imshow(wordcloud, interpolation="bilinear")
        
        review          = reviews.iloc[indices[i][j]]
        title           = review["reviews.title"]
        polarity        = review["polarity"]
        subjectivity    = review["subjectivity"]

        plt.title(f"\"{title}\"\n"
                  f"(polarity: {round(polarity, 2)},"
                  f" subjectivity: {round(subjectivity, 2)})")

    similarity = similarity_df.iloc[indices[i]]

    plt.suptitle(f"Most {'Diss' if i == 0 else 'S'}imilar Reviews\n"
                 f"(similarity: {round(similarity, 2)})",
                 fontsize=16)
    plt.tight_layout(pad=5)
    plt.show()

# Create a heatmap of the similarity matrix.
# Here, we create tick labels from the review titles.
titles = similarity_df.index.map(
    lambda i: preprocess(nlp(reviews.iloc[i]["reviews.title"])))
titles = titles.map(lambda t: "\n".join(t.split(" ")))

# Plot the heatmap using seaborn and matplotlib.
# NOTE The library `plotly` https://plotly.com/python/heatmaps/ might be a
#      better choice, allowing for interactivity through mouse hover.
plt.figure(figsize=(12, 10))
sns.heatmap(similarity_df, annot=True, cmap='viridis',
            xticklabels=titles, yticklabels=titles)
plt.title("Review Similarity Matrix", fontsize=16)
plt.show()
