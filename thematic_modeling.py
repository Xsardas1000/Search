from __future__ import print_function
import numpy as np
from gensim import corpora, models, similarities
import re, time

class Model:
    def __init__(self):
        self.model_name = None
        self.num_topics = None
        self.alpha = None
        self.model = None
    def make_lda_model(self, name, corpus, dictionary, num_topics, alpha):
        self.model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, alpha=alpha)
        self.num_topics = num_topics
        self.alpha = alpha
        self.model_name = name

#Main
start_time = cur_time = time.time()

processed_files = np.load("./processed_files.npy")
processed_files = [re.split(' ', file) for file in processed_files]

dictionary = corpora.Dictionary(processed_files)
corpus = [dictionary.doc2bow(file) for file in processed_files]
dictionary.save_as_text("./models/dictionary.dict")
corpora.MmCorpus.serialize("./models/corpus.mm", corpus)



NUM_MODELS = 5
num_topics = [30, 50, 70, 100, 200]
alpha_parameters = [0.05, 0.1, 0.2, 0.5, 1.0]
for i in range(NUM_MODELS):
    lda_model = models.LdaModel(corpus, num_topics=num_topics[i], id2word=dictionary, alpha=1/num_topics[i])
    lda_model.save("./models/lda" + str(i) + ".model")
    index = similarities.MatrixSimilarity(lda_model[corpus])
    index.save("./models/lda_MatrixSimilarity" + str(i) + ".index")

print("Total processing time: ", time.time() - start_time)
