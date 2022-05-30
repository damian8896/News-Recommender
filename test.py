# Importing necessary libraries
import inline as inline
import pandas as pd
import numpy as np
import nltk
import ssl

from nltk.corpus import stopwords
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
import re
import string
import random
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from matplotlib import pyplot
from gensim.models import KeyedVectors
import gensim.downloader as api


# Downloading the Google pretrained Word2Vec Model
path = api.load("word2vec-google-news-300", return_path=True)

EMBEDDING_FILE = path
model = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)

result = model.most_similar(positive=['king', 'women'], negative=['man'], topn=1)
print(result)