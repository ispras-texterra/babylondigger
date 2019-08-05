from typing import Optional, Tuple, Iterator, Iterable, List
from functools import total_ordering

######################## Lemma Transformation ########################


class LemmaTransformation(object):

    def is_applicable(self, word: str) -> bool:
        pass

    def transform(self, word: str) -> str:
        pass


@total_ordering
class LemmaSuffixTransformation(LemmaTransformation):

    def __init__(self, word_suffix: str, lemma_suffix: str, exception: bool = False):
        self.__word_suffix = word_suffix
        self.__lemma_suffix = lemma_suffix
        self.__exception = exception

    @property
    def is_regular(self) -> bool:
        return not self.__exception

    @property
    def word_suffix(self) -> str:
        return self.__word_suffix

    @property
    def lemma_suffix(self):
        return self.__lemma_suffix

    def is_applicable(self, word: str) -> bool:
        return word.endswith(self.__word_suffix)

    def transform(self, word: str) -> str:
        if not self.is_applicable(word):
            raise ValueError()
        return word[:len(word) - len(self.__word_suffix)] + self.__lemma_suffix

    def __repr__(self):
        return "{{'word_suffix': {}, 'lemma_suffix': {}, 'whole_word': {}}}".format(
            repr(self.__word_suffix), repr(self.__lemma_suffix), self.__exception)

    def __str__(self):
        line = "~{} -> ~{}"
        if self.__exception:
            line = "{} -> {}"
        return line.format(self.__word_suffix, self.__lemma_suffix)

    def __eq__(self, other):
        if isinstance(other, LemmaSuffixTransformation):
            return self.__word_suffix == other.__word_suffix and \
                   self.__lemma_suffix == other.__lemma_suffix and \
                   self.__exception == other.__exception
        return NotImplemented

    def __lt__(self, other):
        if not isinstance(other, LemmaSuffixTransformation):
            return NotImplemented
        return self.__tuple_view() < other.__tuple_view()

    def __hash__(self):
       return hash(self.__tuple_view())

    def __tuple_view(self):
        return self.__word_suffix, self.__lemma_suffix, self.__exception

    @classmethod
    def get_rule(cls, word, lemma):
        def common_start():
            length = min(len(word), len(lemma))
            for i in range(length):
                if not word[i] == lemma[i]:
                    return i
            return length

        start = common_start()
        return LemmaSuffixTransformation(word[start:], lemma[start:], start == 0)


########################## Lemma Suffix Trie #########################


class LemmaTransformationTrie(object):
    def __init__(self):
        self.__children = None
        self.__regular = None
        self.__exceptions = None

    def _get_child(self, ch: str) -> Optional['LemmaTransformationTrie']:
        if self.__children is not None:
            return self.__children.get(ch, None)
        return None

    def __get_or_add_child(self, ch: str) -> 'LemmaTransformationTrie':
        if self.__children is None:
            self.__children = {}
        if ch not in self.__children:
            self.__children[ch] = LemmaTransformationTrie()
        return self.__children[ch]

    def __add_rule(self, rule: Tuple[LemmaSuffixTransformation, int]):
        if rule[0].is_regular:
            self.__add_regular(rule)
        else:
            self.__add_exception(rule)

    def __add_regular(self, rule: Tuple[LemmaSuffixTransformation, int]):
        if self.__regular is None:
            self.__regular = []
        self.__regular.append(rule)

    def __add_exception(self, rule: Tuple[LemmaSuffixTransformation, int]):
        if self.__exceptions is None:
            self.__exceptions = []
        self.__exceptions.append(rule)

    def get_applicable_rules(self, word: str) -> List[Tuple[LemmaSuffixTransformation, int]]:
        result = []
        current = self
        for ch in reversed(word):
            _extend(result, current.__regular)
            child = current._get_child(ch)
            if child is None:
                break
            current = child

        _extend(result, current.__exceptions)
        return result

    @classmethod
    def build(cls, rules: Iterator[Tuple[LemmaSuffixTransformation, int]]) -> 'LemmaTransformationTrie':
        root = LemmaTransformationTrie()

        def __put_rule(rule: Tuple[LemmaSuffixTransformation, int]):
            current = root
            for ch in reversed(rule[0].word_suffix):
                current = current.__get_or_add_child(ch)
            current.__add_rule(rule)

        for rule in rules:
            __put_rule(rule)

        return root


def _extend(l: List, elements: Iterable):
    if elements is not None:
        l.extend(elements)
