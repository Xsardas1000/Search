import os
import os.path

class LocalTxtCorpus:
    def __init__(self, directory, shard=0, n_shards=1):
        assert n_shards > 0
        assert 0 <= shard < n_shards
        self.directory = directory
        self.shard = shard
        self.n_shards = n_shards

    def __iter__(self):
        filenames = [name for name in os.listdir(self.directory)
                if os.path.splitext(name)[-1] == '.txt']
        filenames.sort()
        for filename in filenames[self.shard::self.n_shards]:
            yield filename, self.get_document(filename)

    def get_document(self, filename):
        with open(os.path.join(self.directory, filename), 'r') as document:
            return document.read()
