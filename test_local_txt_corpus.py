from local_txt_corpus import LocalTxtCorpus

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
    def test_lda_model(self, directory, model_path):
        corpus = LocalTxtCorpus('lenta_articles')
        model = models.LdaModel.load('./models/lda4.model')
        index = similarities.MatrixSimilarity.load('./models/lda_MatrixSimilarity4.index')


if __name__ == '__main__':
