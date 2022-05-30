# Importing necessary libraries
import inline as inline
import pandas as pd
import numpy as np

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
import re
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import gensim.downloader as api

df = pd.read_csv("data.csv")


def _removeNonAscii(s):
    return "".join(i for i in s if ord(i) < 128)


def make_lower_case(text):
    return text.lower()


def remove_stop_words(text):
    text = text.split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]
    text = " ".join(text)
    return text


def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)


def remove_punctuation(text):
    tokenizer = RegexpTokenizer(r'\w+')
    text = tokenizer.tokenize(text)
    text = " ".join(text)
    return text


df['cleaned'] = df['Desc'].apply(_removeNonAscii)

df['cleaned'] = df.cleaned.apply(func=make_lower_case)
df['cleaned'] = df.cleaned.apply(func=remove_stop_words)
df['cleaned'] = df.cleaned.apply(func=remove_punctuation)
df['cleaned'] = df.cleaned.apply(func=remove_html)

corpus = []
for words in df['cleaned']:
    corpus.append(words.split())

# Downloading the Google pretrained Word2Vec Model
path = api.load("word2vec-google-news-300", return_path=True)

EMBEDDING_FILE = path
google_word2vec = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

# Training our corpus with Google Pretrained Model

google_model = Word2Vec(vector_size=300, window=5, min_count=2, workers=-1)
google_model.build_vocab(corpus)

google_model.wv.vectors_lockf = np.ones(len(google_model.wv))

google_model.wv.intersect_word2vec_format(EMBEDDING_FILE, binary=True)

google_model.train(corpus, total_examples=google_model.corpus_count, epochs=5)

# Building TFIDF model and calculate TFIDF score

tfidf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=5, stop_words='english')
tfidf.fit(df['cleaned'])

# Getting the words from the TF-IDF model

tfidf_list = dict(zip(tfidf.get_feature_names(), list(tfidf.idf_)))
tfidf_feature = tfidf.get_feature_names()  # tfidf words/col-names

# Storing the TFIDF Word2Vec embeddings
tfidf_vectors = [];
line = 0;
# for each book description
for desc in corpus:
    # Word vectors are of zero length (Used 300 dimensions)
    sent_vec = np.zeros(300)
    # num of words with a valid vector in the book description
    weight_sum = 0;
    # for each word in the book description
    for word in desc:
        if word in google_model.wv and word in tfidf_feature:
            vec = google_model.wv[word]
            tf_idf = tfidf_list[word] * (desc.count(word) / len(desc))
            sent_vec += (vec * tf_idf)
            weight_sum += tf_idf
    if weight_sum != 0:
        sent_vec /= weight_sum
    tfidf_vectors.append(sent_vec)
    line += 1


def recommendations(title):
    # finding cosine similarity for the vectors

    cosine_similarities = cosine_similarity(tfidf_vectors, tfidf_vectors)

    # taking the title and book image link and store in new data frame called books
    books = df[['title', 'image_link']]
    # Reverse mapping of the index
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()

    idx = indices[title]
    sim_scores = list(enumerate(cosine_similarities[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    book_indices = [i[0] for i in sim_scores]
    recommend = books.iloc[book_indices]
    for index, row in recommend.iterrows():
        response = requests.get(row['image_link'])
        img = Image.open(BytesIO(response.content))
        plt.figure()
        plt.imshow(img)
        plt.title(row['title'])

recommendations("The One Minute Manager Meets the Monkey")
plt.show()
