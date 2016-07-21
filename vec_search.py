from __future__ import print_function

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import re, time
from nltk.corpus import stopwords, wordnet
from nltk.stem import SnowballStemmer
from gensim import corpora, models, similarities
from textblob import TextBlob
import scipy as sp

def prepare_doc(doc):
    blob = TextBlob(doc)
    if blob.detect_language() != "en":
        doc = blob.translate(to="en")
    doc = re.sub(r"(\n)", " ", doc.string.lower())
    doc = re.sub(r"(-\n)", "", doc)
    doc = re.split("[^a-z0-9]", doc)
    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer('english')
    doc = [stemmer.stem(word) for word in doc if word not in stop_words & len(word) > 1 & len(word) < 20]
    return ' '.join(doc)


def dist_raw(v1, v2):
    delta = v1 - v2
    return sp.linalg.norm(delta.toarray())

def cos_raw(v1, v2):
    return sp.spatial.distance.cosine(v1.toarray(), v2.toarray())

def prepare_request(request):
    blob = TextBlob(request)
    if blob.detect_language() != "en":
        request = blob.translate(to="en").string
    request = re.split("[^a-z0-9]", request.lower())
    stop_words = stopwords.words('english')
    stemmer = SnowballStemmer('english')
    #request = add_synonyms([word for word in request if word not in stop_words])
    request = [stemmer.stem(word) for word in request if word not in stop_words]

    return ' '.join(request)


def range_search(request, corpus):
    distances = []
    for i, doc in enumerate(corpus):
        vec = corpus.getrow(i)
        distances.append((cos_raw(vec, request), i))
    return distances

def find_similar(file, processed_files, request_key_words, n):
    model = models.LdaModel.load('./models/lda4.model')
    index = similarities.MatrixSimilarity.load('./models/lda_MatrixSimilarity4.index')
    dictionary = corpora.Dictionary.load_from_text('./models/dictionary.dict')
    vec = dictionary.doc2bow(re.split(' ', file))
    topics = model[vec]
    #print(topics)
    #print(model.show_topic(46, topn = 30))
    key_words = set()
    [key_words.update([key[0] for key in model.show_topic(topic_index, topn=20)]) for topic_index in [topic[0] for topic in topics]]

    sims = index[topics]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    min_weight = 0.1
    for i in range(n):
        if sims[i][1] > min_weight:
            print("Weight: ", sims[i][1])
            print_highlighted_doc(processed_files[sims[i][0]], ' '.join(list(key_words)))
    return sims

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

def print_n_docs(files, distances, keywords, n):
    for i in range(n):
        print("Distance: ", distances[i][0])
        print_highlighted_doc(files[distances[i][1]], key_words)


def get_ngrams(doc, n, amount = 0):
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



#Main
start_time = cur_time = time.time()


processed_files = np.load("./processed_files.npy")
vectorizer = TfidfVectorizer(min_df=1)

corpus = vectorizer.fit_transform(processed_files)
num_samples, num_features = corpus.shape

print("Number of features: ", num_features)
print("Number of samples: ", num_samples)


request = "apple music"
print("Request: ", request)
request = prepare_request(request)
key_words = request
print("Prepared request: ", request, "\n")
request = vectorizer.transform([request])


distances = range_search(request, corpus)
distances = sorted(distances, key=lambda item: item[0])
num_vec_sim = 5
print_n_docs(processed_files, distances, key_words, num_vec_sim)

num_vec_sim = 5
sim_docs = find_similar(processed_files[distances[0][1]], processed_files, key_words, num_vec_sim)

print("Total processing time: ", time.time() - start_time)


