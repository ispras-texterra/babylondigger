import unittest

from babylondigger.datamodel import Document, Token, Pos
from babylondigger.tagger import TaggerInterface, TaggerTrainerInterface

from babylondigger.evaluation.evaluation import DataSetTaggerTester, ParametersEstimator, score_comparator

class _Tagger_Mock(TaggerInterface):
    def __init__(self, pos):
        self.__pos = pos

    def tag(self, documents):
        return map(self._tag, documents)

    def _tag(self, document: Document):
        tokens = [Token.with_morphology(token, Pos(self.__pos), token.text) for token in document.tokens]
        return Document.from_document(document, tokens)


_gold_data = [Document("Hello, world!", [
            Token(0, 5, Pos("INTJ"), "hello"), Token(5, 6, Pos("PUNCT"), ","),
            Token(7, 12, Pos("N", feats={"Number": "Singular"}), "world"), Token(12, 13, Pos("PUNCT"), "!")])]

_tester = DataSetTaggerTester(_gold_data)

class TaggerEvaluatorTest(unittest.TestCase):

    def test_right_answer(self):
        class Tagger_Mock(TaggerInterface):
            def tag(self, documents):
                return iter(_gold_data)

        scores = _tester.test(Tagger_Mock())

        for score in scores.values():
            self._check_score(score, 1.0, 1.0, 1.0)

    def test_half_of_answers(self):


        scores = _tester.test(_Tagger_Mock("PUNCT"))

        self._check_score(scores["UPOS"], 0.5, 0.5, 0.5)
        self._check_score(scores["Feats"], 0.75, 0.75, 0.75)
        self._check_score(scores["Lemmas"], 0.75, 0.75, 0.75)


    def _check_score(self, score, precision, recall, f1):
        self.assertAlmostEqual(score.precision, precision)
        self.assertAlmostEqual(score.recall, recall)
        self.assertAlmostEqual(score.f1, f1)


class ScoreComparatorTest(unittest.TestCase):

    class Score(object):
        def __init__(self, f1):
            self.f1 = f1

    _scores1 = {"metric1": Score(1.0), "metric2": Score(0.5), "metric3": Score(0.7)}
    _scores2 = {"metric1": Score(0.9), "metric2": Score(0.6), "metric3": Score(0.5)}
    _scores3 = {"metric1": Score(1.0), "metric2": Score(0.6), "metric3": Score(0.7)}

    def test_single(self):
        comparator = score_comparator(["metric1"], lambda score: score.f1)
        self.assertTrue(comparator(ScoreComparatorTest._scores1, ScoreComparatorTest._scores2))
        self.assertFalse(comparator(ScoreComparatorTest._scores1, ScoreComparatorTest._scores3))

    def test_several(self):
        comparator = score_comparator(["metric1", "metric2"], lambda score: score.f1)
        self.assertFalse(comparator(ScoreComparatorTest._scores1, ScoreComparatorTest._scores2))
        self.assertTrue(comparator(ScoreComparatorTest._scores3, ScoreComparatorTest._scores1))


class ParameterEstimatorTest(unittest.TestCase):
    def test_best_params(self):
        class TrainerMock(TaggerTrainerInterface):
            def train(self, documents, **kwargs):
                return [_Tagger_Mock(kwargs['param'])]

        hyperparams = [{'param': 'N'}, {'param': 'PUNCT'}, {'param': 'INTJ'}]

        estimator = ParametersEstimator(score_comparator(["UPOS"], lambda score: score.f1), _gold_data, _tester)
        params, _, _ = estimator.params_search(TrainerMock(), iter(hyperparams))

        self.assertEqual(params, hyperparams[1])

