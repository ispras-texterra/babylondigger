from typing import Iterable, Iterator, Dict, Any, Callable, Tuple, List

from io import StringIO
import argparse
import json
import os
import shutil
from collections import namedtuple, defaultdict
from babylondigger.tagger import TaggerInterface, TaggerTrainerInterface
from babylondigger.datamodel import Document, Token
from babylondigger.evaluation.conllu_io import CoNLLUWriter
from babylondigger.evaluation.dataset import CoNLLUDataSet

from babylondigger.evaluation.trainers import get_trainer

import babylondigger.evaluation.conll17_ud_eval as conll17

_SCORES = Dict[str, 'Score']
_PARAMS = Dict[str, Any]

###################### tester ######################

_writer = CoNLLUWriter()


class TaggerTester(object):

    def __init__(self, results_path: str = None):
        self.__results_path = results_path

    def _gold_data_internal(self):
        pass

    def _gold_data(self) -> Iterator[Document]:
        pass

    def test(self, tagger: TaggerInterface, file_name: str) -> _SCORES:
        test_data = self.__prepare_test_data(self._gold_data())
        predicted = tagger.tag(test_data)
        internal = self._save_and_read(file_name, predicted) if self.__results_path \
            else TaggerTester._convert_to_internal(predicted)
        gold_data = self._gold_data_internal()
        return conll17.evaluate(gold_data, internal)

    def _save_and_read(self, file_name: str, documents: Iterator[Document]):
        path = os.path.join(self.__results_path, file_name)
        dir_name = os.path.dirname(path)
        os.makedirs(dir_name, exist_ok=True)
        _writer.write_to_file(path, *documents)
        print("Test results have been written to '{}'".format(path))
        with open(path, encoding='utf8') as file:
            return conll17.load_conllu(file)

    @classmethod
    def _convert_to_internal(cls, documents: Iterator[Document]):
        conll = _writer.write_to_str(*documents)
        with StringIO(conll) as file:
            return conll17.load_conllu(file)

    def __prepare_test_data(self, documents: Iterable[Document]) -> Iterator[Document]:
        def __clear_document(document: Document):
            tokens = [Token.with_syntax(Token.with_morphology(token)) for token in document.tokens]
            return Document.from_document(document, tokens)

        return map(__clear_document, documents)


class DataSetTaggerTester(TaggerTester):
    def __init__(self, gold_data: Iterable[Document]):
        TaggerTester.__init__(self)
        self.__gold_data = gold_data

    def _gold_data_internal(self):
        return TaggerTester._convert_to_internal(self._gold_data())

    def _gold_data(self) -> Iterator[Document]:
        return iter(self.__gold_data)


class FileTaggerTester(TaggerTester):
    def __init__(self, gold_data_file, results_path: str = None):
        TaggerTester.__init__(self, results_path)
        self.__gold_data_file = gold_data_file

    def _gold_data_internal(self):
        with open(self.__gold_data_file, encoding='utf8') as file:
            return conll17.load_conllu(file)

    def _gold_data(self) -> Iterator[Document]:
        return iter(CoNLLUDataSet(self.__gold_data_file))


###################### estimator ######################

def _is_better(is_better_func, scores, best_scores):
    return best_scores is None or is_better_func(scores, best_scores)


class Evaluator(object):
    def evaluate(self, trainer: TaggerTrainerInterface, params: _PARAMS, model_path: str = '') -> \
            Tuple[Callable[[TaggerTester, str], _SCORES], _SCORES]:
        pass


class SingleSeedEvaluator(Evaluator):
    def __init__(self, model_base_dir, seed, is_better_func: Callable[[_SCORES, _SCORES], bool],
                 train_data: Iterable[Document], dev_tester: TaggerTester, metrics_printer, disposable_callback):
        self._model_base_dir = model_base_dir
        self._seed = seed
        self._is_better_func = is_better_func
        self._train_data = train_data
        self._dev_tester = dev_tester
        self._metrics_printer = metrics_printer
        self._disposable_callback = disposable_callback

    def evaluate(self, trainer: TaggerTrainerInterface, params: _PARAMS, model_path: str = '') -> \
            Tuple[Callable[[TaggerTester, str], _SCORES], _SCORES]:
        model_path = os.path.join(model_path, str(self._seed))
        model_dir = os.path.join(self._model_base_dir, model_path)
        model_name_format = "epoch-{}"
        epoch_path = os.path.join(model_path, model_name_format)
        dev_pred_path = os.path.join(model_path, "epoch-{}-dev.conll")

        epoch = 0
        best_tagger, best_scores, best_path = None, None, None
        for scores, tagger in trainer.train(self._train_data,
                                            dev_tester=lambda tagger: self._dev_tester.test(tagger, dev_pred_path.format(epoch)),
                                            seed=self._seed, model_dir=model_dir, model_name_format=model_name_format,
                                            **params):
            print("Evaluation for {}...".format(tagger))
            print("Dev scores:")
            self._metrics_printer(scores)
            if _is_better(self._is_better_func, scores, best_scores):
                if best_path is not None:
                    self._disposable_callback(best_path)
                best_tagger = tagger
                best_path = epoch_path.format(epoch)
                best_scores = scores
            else:
                self._disposable_callback(epoch_path.format(epoch))
            epoch += 1

        print("Quality evaluation final results:\n")
        print("Hyperparams: {}\n".format(params))
        print("Seed: {}\n".format(self._seed))
        print("Best model: {}\n".format(best_tagger))
        print("Dev scores:")
        self._metrics_printer(best_scores)
        return lambda tester, name: tester.test(best_tagger,
                                                os.path.join(model_path, name)), best_scores


class AverageEvaluator(Evaluator):
    def __init__(self, evaluators: List[Evaluator], metrics_averager, metrics_printer):
        self._evaluators = evaluators
        self._metrics_averager = metrics_averager
        self._metrics_printer = metrics_printer

    def evaluate(self, trainer: TaggerTrainerInterface, params: _PARAMS, model_path: str = '') -> \
            Tuple[Callable[[TaggerTester, str], _SCORES], _SCORES]:
        best_taggers = []
        best_scores = []

        for evaluator in self._evaluators:
            tagger_func, scores = evaluator.evaluate(trainer, params, model_path)
            best_taggers.append(tagger_func)
            best_scores.append(scores)

        average_scores = self._metrics_averager(best_scores)

        print("Hyperparams: {}\n".format(params))
        print("Average best scores:")
        self._metrics_printer(average_scores)

        return lambda tester, name: self._metrics_averager(
            [tagger_func(tester, name) for i, tagger_func in enumerate(best_taggers)]), average_scores


_Best = namedtuple('BestModel', ['params', 'model', 'scores', 'path'])


class ParametersEstimator(object):
    def __init__(self, evaluator: Evaluator, is_better_func: Callable[[_SCORES, _SCORES], bool], disposable_callback):
        self._evaluator = evaluator
        self._is_better_func = is_better_func
        self._disposable_callback = disposable_callback

    def params_search(self, trainer: TaggerTrainerInterface, params_iterator: Iterator[_PARAMS]) -> _Best:
        best = _Best(None, None, None, None)
        for index, params in enumerate(params_iterator):
            print("Test #{}".format(index))
            print("Testing {} params...".format(params))
            path = "params-{}".format(index)
            model, scores = self._evaluator.evaluate(trainer, params, path)
            if _is_better(self._is_better_func, scores, best.scores):
                if best.path is not None:
                    self._disposable_callback(best.path)
                best = _Best(params, model, scores, path)
            else:
                self._disposable_callback(path)
        return best


def score_comparator(fields: List[str], value_extractor: Callable[['Score'], float]) -> \
        Callable[[_SCORES, _SCORES], bool]:
    def __sum(scores: _SCORES) -> float:
        return sum(value_extractor(scores.get(field)) for field in fields)

    def __compare(new: _SCORES, old: _SCORES) -> bool:
        return __sum(new) > __sum(old)

    return __compare


def get_metrics_printer(metrics: List[str]):
    def print_metrics(scores):
        print("Metrics    | Precision |    Recall |  F1 Score | AligndAcc")
        print("-----------+-----------+-----------+-----------+-----------")
        for metric in metrics:
            print("{:11}|{:10.2f} |{:10.2f} |{:10.2f} |{}".format(
                metric,
                100 * scores[metric].precision,
                100 * scores[metric].recall,
                100 * scores[metric].f1,
                "{:10.2f}".format(100 * scores[metric].aligned_accuracy) if scores[metric].aligned_accuracy is not None else ""
            ))

    return print_metrics


def get_metrics_averager(metrics: List[str]):
    _Scores = namedtuple('Scores', ['precision', 'recall', 'f1', 'aligned_accuracy'])

    def calculate_average_scores(scores):
        sums = {metric: defaultdict(int) for metric in metrics}
        for score in scores:
            for metric in metrics:
                sums[metric]['precision'] += score[metric].precision
                sums[metric]['recall'] += score[metric].recall
                sums[metric]['f1'] += score[metric].f1
                if score[metric].aligned_accuracy is not None:
                    sums[metric]['aligned_accuracy'] += score[metric].aligned_accuracy

        return {metric: _Scores(**{key: value / len(scores) for key, value in sums[metric].items()}) for metric in
                metrics}

    return calculate_average_scores


###################### main ######################


def main(args):
    if args.delete_after and os.path.exists(args.settings['model_dir']) and os.listdir(args.settings['model_dir']):
        raise ValueError("{} directory is not empty! Specify an empty directory for saving tagger models."
                         .format(args.settings['model_dir']))

    metrics_printer = get_metrics_printer(args.settings["metric_printer"]["fields"])

    trainer = get_trainer(args.method)

    metrics_averager = get_metrics_averager(args.settings["metric_printer"]["fields"])
    estimator = __build_estimator(args, metrics_printer, metrics_averager)
    hyperparams = args.hyperparams
    if not isinstance(hyperparams, list):
        hyperparams = [hyperparams]

    params, tagger_func, dev_scores, path = estimator.params_search(trainer, iter(hyperparams))

    print("Quality evaluation final results:\n")
    print("Hyperparams: {}\n".format(params))

    print("Dev scores:")
    metrics_printer(dev_scores)

    if "test_set" in args.settings:
        tester = __build_tester(args.settings["test_set"])
        test_scores = tagger_func(tester, "best-test.conll")

        print("\nTest scores:")
        metrics_printer(test_scores)

    if args.delete_after:
        shutil.rmtree(args.settings['model_dir'], ignore_errors=True)


def __build_estimator(args, metrics_printer, metrics_averager):
    settings = args.settings
    fields = settings["cmp_metric"]["fields"]
    metric = settings["cmp_metric"]["metric"]
    is_better_func = score_comparator(fields, lambda score: getattr(score, metric))
    train = __build_dataset(settings["train_set"])

    def remove_model(path):
        shutil.rmtree(os.path.join(args.settings['model_dir'], path), ignore_errors=True)

    disposable_callback = remove_model if args.delete_after else lambda _: None

    dev_tester = __build_tester(settings["dev_set"])
    model_path = settings['model_dir']
    if type(settings['random_seed']) is list:
        evaluators = [SingleSeedEvaluator(model_path, seed, is_better_func, train, dev_tester, metrics_printer, disposable_callback)
                      for seed in settings['random_seed']]
        evaluator = AverageEvaluator(evaluators, metrics_averager, metrics_printer)
    else:
        evaluator = SingleSeedEvaluator(model_path, settings['random_seed'], is_better_func, train, dev_tester,
                                        metrics_printer, disposable_callback)

    return ParametersEstimator(evaluator, is_better_func, disposable_callback)


def __build_tester(tester_settings):
    path = tester_settings["path"]
    results_path = tester_settings.get("results_path", None)
    return FileTaggerTester(path, results_path)


def __build_dataset(dataset_settings):
    path = dataset_settings["path"]
    dataset = CoNLLUDataSet(path)
    if "shuffler" in dataset_settings:
        from babylondigger.evaluation.dataset import ShuffledDataset
        dataset = ShuffledDataset(dataset, **dataset_settings["shuffler"])
    return dataset


class _Arguments(object):
    def __init__(self, settings, method, hyperparams, delete_after):
        self.settings = settings
        self.method = method
        self.hyperparams = hyperparams
        self.delete_after = delete_after


def _build_parser():
    parser = argparse.ArgumentParser(description='Quality evaluator')
    parser.add_argument('--settings', dest='settings', type=str,
                            help='path to json configuration file')
    parser.add_argument('--method', dest='method', type=str, help='method to be tested')
    parser.add_argument('--hyperparams', dest='hypers', type=str, help='path to hyperparams file')
    parser.add_argument('--delete_after', dest='delete_after', action='store_true',
                            help='indicates whether models should be deleted after the experiment is over')
    return parser


def _parse_arguments(parser):
    args = parser.parse_args()
    with open(args.settings) as f:
        settings = json.load(f)
    with open(args.hypers) as f:
        hyperparams = json.load(f)
    return _Arguments(settings, args.method, hyperparams, args.delete_after)


if __name__ == "__main__":
    args = _parse_arguments(_build_parser())
    main(args)
