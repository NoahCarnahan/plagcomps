import unittest
from plagcomps.intrinsic.featureextraction import FeatureExtractor

class FeatureExtractionTestCase(unittest.TestCase):

    def setUp(self):
        '''
        TODO Find a way to run this only once rather than before EACH test
        '''
        self.one_paragraph = 'The brown fox ate. Believe it. I go to the school.'
        self.multi_paragraph = 'The brown fox ate. Believe it. I go to the school? \n\n' + \
                               'Here is a second paragraph.'
    
        self.one_paragraph_extractor = FeatureExtractor(self.one_paragraph)
        self.multi_paragraph_extractor = FeatureExtractor(self.multi_paragraph)

    def test_get_spans(self):
        pass

    def test_get_passages(self):
        pass

    def test_get_feature_vectors(self):
        pass

    # All the below use <get_feature_vectors> for a given feature

    def test_average_word_length(self):
        actual = self.one_paragraph_extractor.get_feature_vectors(['average_word_length'], 'sentence')
        expected = [(3.75,), (5.0,), (3.0,)]

        for a, e in zip(actual, expected):
            # Only need to look at first element, since using just one feature
            self.assertAlmostEqual(a[0], e[0])

    def test_average_sentence_length(self):
        #actual = self.multi_paragraph_extractor.get_feature_vectors(['average_sentence_length'], 'paragraph')
        pass

    def test_stopword_percentage(self):
        actual = self.one_paragraph_extractor.get_feature_vectors(['stopword_percentage'], 'sentence')
        expected = [(0.25,), (0.5,), (0.6,)]

        for a, e in zip(actual, expected):
            # Only need to look at first element, since using just one feature
            self.assertAlmostEqual(a[0], e[0])

    def test_punctuation_percentage(self):
        actual = self.one_paragraph_extractor.get_feature_vectors(['punctuation_percentage'], 'sentence')
        expected = [(0.05555555,), (0.09090909,), (0.05263157,)]

        for a, e in zip(actual, expected):
            # Only need to look at first element, since using just one feature
            # Note that we compare to 7 decimal points
            self.assertAlmostEqual(a[0], e[0], 7)

    def test_syntactic_complexity(self):
        pass

    def test_syntactic_complexity_average(self):
        pass

    def test_avg_internal_word_freq_class(self):
        pass

    def test_avg_external_word_freq_class(self):
        pass

if __name__ == '__main__':
    unittest.main()
