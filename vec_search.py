from __future__ import print_function

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re, time
from nltk.corpus import stopwords, wordnet
from nltk.stem import SnowballStemmer
from gensim import corpora, models, similarities
from textblob import TextBlob
import requests
import scipy as sp
import urllib
import json

def translate(doc, to_language = 'en'):
    url = 'http://translate.google.com/translate_a/t'
    params = {
        "text": doc,
        "sl": TextBlob(doc).detect_language(),
        "tl": to_language,
        "client": "p"
    }
    return requests.get(url, params=params).content

'''
def translate(text, src = '', to = 'en'):
  parameters = ({'langpair': '{0}|{1}'.format(src, to), 'v': '1.0' })
  translated = ''

  for text in (text[index:index + 4500] for index in range(0, len(text), 4500)):
    parameters['q'] = text
    response = json.loads(urllib.request.urlopen('http://ajax.googleapis.com/ajax/services/language/translate',
                                                 data=urllib.parse.urlencode(parameters).encode('utf-8')).read().decode('utf-8'))
    try:
      translated += response['responseData']['translatedText']
    except:
      pass
  return translated '''

def prepare_request(request, synonyms = False):
    #request = translate(request)
    request = re.sub(r"(\n)", " ", request.lower())
    request = re.sub(r"(-\n)", "", request)
    request = re.split("[^a-z0-9]", request)
    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer('english')
    if synonyms == True:
        request = add_synonyms([word for word in request if word not in stop_words])
    request = [stemmer.stem(word) for word in request if (word not in stop_words) & (len(word) > 1) & (len(word) < 20)]
    return ' '.join(request)


def dist_raw(v1, v2):
    delta = v1 - v2
    return sp.linalg.norm(delta.toarray())

def cos_raw(v1, v2):
    return sp.spatial.distance.cosine(v1.toarray(), v2.toarray())

def range_search(vec_request, corpus):
    distances = []
    for i, doc in enumerate(corpus):
        vec = corpus.getrow(i)
        distances.append((cos_raw(vec, vec_request), i))
    return distances

def lda_search(request, dictionary_path, model_number):
    model_path = "./models/lda" + str(model_number) + ".model"
    index_path = "./models/lda_MatrixSimilarity" + str(model_number) + ".index"

    dictionary = corpora.Dictionary.load_from_text(dictionary_path)
    model = models.LdaModel.load(model_path)
    index = similarities.MatrixSimilarity.load(index_path)

    vec = dictionary.doc2bow(re.split(' ', request))
    topics = model[vec]
    key_words = set()
    [key_words.update([key[0] for key in model.show_topic(topic_index, topn=20)])
     for topic_index in [topic[0] for topic in topics]]

    sims = index[topics]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])

    return sims, key_words


def print_highlighted_doc(doc, key_words):
    highlighted_doc = ""
    str_len = 0
    for word in re.split(' ', doc):
        if word in re.split(' ', key_words):
            highlighted_doc += "\033[44;38m" + word + "\033[m "
        else:
            highlighted_doc += word + " "
        str_len += len(word)
        if str_len > 100:
                str_len = 0
                highlighted_doc += "\n"
    print(highlighted_doc + "\n")

def print_n_docs(files, distances, key_words, num_sims = 1):
    for i in range(num_sims):
        print("Distance: ", distances[i][0])
        print_highlighted_doc(files[distances[i][1]], key_words)


def get_ngrams(doc, n):
    blob = TextBlob(doc)
    ngrams = blob.ngrams(n = n)
    return ngrams

def add_synonyms(request):
    extended_request = set()
    for word in request:
        synonyms = wordnet.synsets(word)
        for syn in synonyms:
            extended_request.update(syn.lemma_names())
    return list(extended_request)

def vec_search(vectorizer, corpus, request):

    request = prepare_request(request)
    key_words = request
    request = vectorizer.transform([request])
    distances = range_search(request, corpus)
    distances = sorted(distances, key=lambda item: item[0])
    return distances, key_words


def create_vectorizer(processed_files_path):
    processed_files = np.load(processed_files_path)
    vectorizer = TfidfVectorizer(min_df=1)
    corpus = vectorizer.fit_transform(processed_files)
    return vectorizer, corpus, processed_files


if __name__ == '__main__':
    processed_files_path = "./processed_files.npy"
    request = "apple"

    vectorizer, corpus, processed_files = create_vectorizer(processed_files_path)
    distances, key_words = vec_search(vectorizer, corpus, request)
    print_n_docs(processed_files, distances, key_words, num_sims=5)

    sims, key_words = lda_search(processed_files[distances[0][1]], './models/dictionary.dict', 9)

