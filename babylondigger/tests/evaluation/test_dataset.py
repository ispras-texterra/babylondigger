import unittest

from babylondigger.evaluation.dataset import ShuffledDataset

class ShuffledDatasetTest(unittest.TestCase):

    @property
    def _base_dataset(self):
        return list(range(300))

    def test_empty(self):
        dataset = ShuffledDataset([])
        self.assertEqual(list(dataset), [])

    def test_always_same_output(self):
        dataset1 = ShuffledDataset(self._base_dataset, 1)
        dataset2 = ShuffledDataset(self._base_dataset, 1)
        for i in range(10):
            self.assertEqual(list(dataset1), list(dataset2))  # each time same output

    def test_different_iterators(self):
        dataset = ShuffledDataset(self._base_dataset, 1)
        self.assertNotEqual(list(dataset), list(dataset))

    def test_skip_iterator(self):
        dataset1 = ShuffledDataset(self._base_dataset, 1)
        dataset2 = ShuffledDataset(self._base_dataset, 1)

        i = iter(dataset1)
        list21 = list(dataset2)
        list12 = list(dataset1)
        list22 = list(dataset2)
        list11 = list(i)

        self.assertEqual(list11, list21)
        self.assertEqual(list12, list22)
