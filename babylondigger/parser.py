from typing import Iterator

from babylondigger.datamodel import Document
from babylondigger.processor import DocumentProcessorInteface


class ParserInterface(DocumentProcessorInteface):

    def process(self, documents: Iterator[Document]) -> Iterator[Document]:
        return self.parse(documents)

    def parse(self, documents: Iterator[Document]) -> Iterator[Document]:
        pass


class StubParser(ParserInterface):

    def __init__(self, deprel: str = None):
        self.__deprel = deprel

    def parse(self, documents: Iterator[Document]) -> Iterator[Document]:
        return map(self._parse, documents)

    def _parse(self, document: Document) -> Document:
        tokens = []
        for sentence in document.sentences:
            for i, token in enumerate(sentence):
                tokens.append(token.with_syntax(token, self.__deprel, head_sentence_index=i-1))
        return document.from_document(document, tokens)
