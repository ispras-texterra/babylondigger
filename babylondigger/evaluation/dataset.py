from typing import Iterable, Iterator

from random import Random
import itertools

from babylondigger.datamodel import Document
from babylondigger.evaluation.conllu_io import CoNLLUReader


class CoNLLUDataSet(object):
    def __init__(self, file_name: str, reader: CoNLLUReader = CoNLLUReader(1)):
        self._reader = reader
        self._file_name = file_name

    def __iter__(self) -> Iterator[Document]:
        return self._reader.read_from_file(self._file_name)


class ShuffledDataset(object):
    def __init__(self, dataset: Iterable[Document], seed: int = None, max_batch_size: int = 50):
        self._dataset = dataset
        self._batch_size = max_batch_size
        if seed is not None:
            _seed = seed
        else:
            import time
            _seed = int(time.time())
        self._random = Random(_seed)

    def __iter__(self):
        def __split():
            i = iter(self._dataset)
            rnd = Random(self._random.randrange(1000000))

            def __get_batch():
                batch_size = rnd.randrange(int(self._batch_size/2) + 1, self._batch_size + 1)
                return list(itertools.islice(i, batch_size))

            batch = __get_batch()
            while batch:
                rnd.shuffle(batch)
                yield batch
                batch = __get_batch()

        return itertools.chain(*__split())
