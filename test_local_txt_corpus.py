from local_txt_corpus import LocalTxtCorpus
import numpy as np
import re
from gensim import models, similarities, corpora
import vec_search

class TestLocalTxtCorpus:
    def test_iter(self):
        corpus = LocalTxtCorpus('test_data/local_txt_corpus')
        expected = [
            'Lorem ipsum dolor sit amet, consectetur adipiscing elit,\n'
            'sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.\n',
            'Hello World!\n',
        ]
        for (filename, text), expected_text in zip(corpus, expected):
            assert text == expected_text

    def test_get_document(self):
        corpus = LocalTxtCorpus('test_data/local_txt_corpus')
        assert corpus.get_document('2.txt') == 'Hello World!\n'

    def test_shards(self):
        corpus = LocalTxtCorpus('test_data/local_txt_corpus', shard=1, n_shards=2)
        for filename, text in corpus:
            assert filename == '2.txt'
            assert text == 'Hello World!\n'

    def test_vec_search(self, processed_files_path):
        '''
        Trying to find the most closest doc for the doc from the corpus using vectorized search we expect
        exactly this doc as a result, because each doc from the corpus is the most similar to itself (vectors which represent
        these docs are collinear)
        '''

        vectorizer, corpus, processed_files = vec_search.create_vectorizer(processed_files_path)
        num_samples = len(processed_files)
        expected = np.array(range(num_samples))
        result = np.array([0] * num_samples)
        for i, file in enumerate(processed_files):
            distances, key_words = vec_search.vec_search(vectorizer, corpus, file)
            result[i] = distances[0][1]
        score = np.sum(expected == result) / num_samples
        print("Testing vec model, score: ", score)
        return score


    def test_lda_model(self, processed_files_path, dictionary_path, model_number, n = 1):
        '''
        In case of thematic modeling, the model can choose the set of topics, which can correspond to another document.
        To check the quality of the model we can run the search script on each example of the corpus to determine how many
        documents have been matched correctly

        We get '1' if our test doc is in N the most similar docs which were found and '0' otherwise
        '''

        processed_files = np.load(processed_files_path)
        num_samples = len(processed_files)
        expected = np.array(range(num_samples))
        result = np.array([0] * num_samples)

        for i, file in enumerate(processed_files):
            sims, key_words = vec_search.lda_search(file, dictionary_path, model_number)
            indices = map(lambda x: x[0], sims[:n])
            if i in indices:
                result[i] = i
            else:
                result[i] = i + 1
        score = np.sum(expected == result) / num_samples
        print("Testing lda model number ", model_number, "Score: ", score)
        return score

    def test_part_doc_vec_search(self, processed_files_path, percent):
        '''
        We want to know how many correct searches will be in case of taking only part of the input doc
        '''
        vectorizer, corpus, processed_files = vec_search.create_vectorizer(processed_files_path)
        num_samples = len(processed_files)
        expected = np.array(range(num_samples))
        result = np.array([0] * num_samples)
        for i, file in enumerate(processed_files):
            if percent >= 50:
                miss = 100 // (100 - percent)
                file = [word for i, word in enumerate(re.split(' ', file)) if i % miss > 0]
            else:
                miss = 100 // percent
                file = [word for i, word in enumerate(re.split(' ', file)) if i % miss == 0]

            distances, key_words = vec_search.vec_search(vectorizer, corpus, ' '.join(file))
            result[i] = distances[0][1]
        score = np.sum(expected == result) / num_samples
        print("Testing vec model, score: ", score)
        return score


test = TestLocalTxtCorpus()
NUM_MODELS = 10
scores = [0] * NUM_MODELS
#for i in range(NUM_MODELS):
#   scores.append(test.test_lda_model('./processed_files.npy', './models/dictionary.dict', i, 5))


#test.test_vec_search("./processed_files.npy")
test.test_part_doc_vec_search('./processed_files.npy', 10)