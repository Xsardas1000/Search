from __future__ import print_function
import re, os, fnmatch
from nltk.corpus import stopwords
from collections import defaultdict
from nltk.stem import SnowballStemmer
from textblob import TextBlob
import time
import numpy as np


source_path = "./lenta_articles"
format_mask = "*.txt"


class Data:
    def __init__(self):
        self.source_paths = []
        self.source_files = []
        self.translated_files = []
        self.prepared_files = []
        self.stemmed_files = []
        self.processed_files = []

        self.dictionary = defaultdict(int)
        self.stop_words = stopwords.words('english')
        self.stemmer = SnowballStemmer('english')
        self.languages = defaultdict(list)

    def find_paths(self, source_path, mask):
        source_paths = []
        for elem in os.listdir(source_path):
            if fnmatch.fnmatch(elem, mask):
             name = os.path.join(source_path, elem)
             source_paths.append(name)
        return source_paths

    def get_files(self, source_paths):
        source_files = []
        for path in source_paths:
            with open(path, 'r') as file:
                source_files.append(file.read())
        return source_files

    def prepare_files(self, files):
        prepared_files = []
        for doc in files:
            raw_text = re.sub(r"(-\n)", "", doc.lower())
            raw_text = re.sub(r"(\n)", " ", raw_text)
            raw_text = re.split("[^a-z0-9]", raw_text)
            prepared_files.append([word for word in raw_text if (len(word) > 2) & (len(word) < 20)])
        return prepared_files

    def stemming(self, files):
        stemmed_files = []
        for doc in files:
            stemmed_file = [self.stemmer.stem(word) for word in doc if word not in self.stop_words]
            stemmed_files.append(stemmed_file)
            for word in stemmed_file:
                self.dictionary[word]+=1
        return stemmed_files

    def recognize_languages(self, files):
        for i in range(len(files)):
            blob = TextBlob(files[i])
            self.languages[blob.detect_language()]+=[i]


    def translation(self, files):
        translated_files = []
        for doc in files:
            if TextBlob(doc).detect_language() != "en":
                translated_files.append(TextBlob(doc).translate(to="en").string)
            else:
                translated_files.append(doc)
        return translated_files

    def delete_rear_words(self, files):
        new_dictionary = defaultdict(int)
        processed_files = [[word for word in doc if self.dictionary[word] > 2] for doc in files]
        for (word, freq) in self.dictionary.items():
            if freq > 2:
                new_dictionary[word] = freq
        self.dictionary = new_dictionary
        return processed_files



def process_files():
    ##Main

    data = Data()

    #working with input data
    start_time = cur_time = time.time()
    print("Started working with data: ", cur_time)

    data.source_paths = data.find_paths(source_path, format_mask)
    data.source_files = data.get_files(data.source_paths)

    print("Finished getting files: ", time.time() - cur_time)
    cur_time = time.time()

    data.translated_files = data.translation(data.source_files)

    print("Finished translating files: ", time.time() - cur_time)
    cur_time = time.time()

    data.prepared_files = data.prepare_files(data.translated_files)

    print("Finished to prepare files: ", time.time() - cur_time)
    cur_time = time.time()

    #data.recognize_languages(data.source_files)

    print("Finished recognizing languages: ", time.time() - cur_time)
    cur_time = time.time()

    data.stemmed_files = data.stemming(data.prepared_files)

    print("Finished stemming files: ", time.time() - cur_time)
    cur_time = time.time()

    data.processed_files = data.delete_rear_words(data.stemmed_files)

    print("Finished processing files: ", time.time() - cur_time)

    print("Total processing time: ", time.time() - start_time)

    processed_files = np.array([' '.join(file) for file in data.processed_files])
    np.save("processed_files", processed_files)

    return data

if __name__ == '__main__':
    process_files()





