from typing import Iterable, List, Iterator

from itertools import chain
from collections import Counter

from babylondigger.datamodel import Document, Token, Pos
from babylondigger.processor import DocumentProcessorInteface


class TaggerInterface(DocumentProcessorInteface):

    def process(self, documents: Iterator[Document]) -> Iterator[Document]:
        return self.tag(documents)

    def tag(self, documents: Iterator[Document]) -> Iterator[Document]:
        pass


class TaggerTrainerInterface(object):

    def train(self, documents: Iterable[Document], **kwargs) -> List[TaggerInterface]:
        pass


class StubTagger(TaggerInterface):

    def __init__(self, pos):
        self.__pos = pos

    def tag(self, documents: Iterator[Document]) -> Iterator[Document]:
        return map(self._tag, documents)

    def _tag(self, document: Document) -> Document:
        tokens = [Token.with_morphology(token, self.__pos, token.text) for token in document.tokens]
        return Document.from_document(document, tokens)


class StubTaggerTrainer(TaggerTrainerInterface):

    def train(self, documents: Iterable[Document], **kwargs):
        pos = self.get_most_frequent_pos(documents)
        return [StubTagger(pos)]

    def get_most_frequent_pos(self, documents: Iterable[Document]) -> Pos:
        tokens = map(lambda document: document.tokens, documents)
        poss = map(lambda token: token.pos, chain(*tokens))
        return Counter(poss).most_common(1)[0][0]
