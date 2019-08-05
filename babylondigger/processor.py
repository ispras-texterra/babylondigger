from typing import Iterator, Iterable

from babylondigger.datamodel import Document


class DocumentProcessorInteface(object):
    def process(self, documents: Iterator[Document]) -> Iterator[Document]:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class DocumentComplexProcessor(DocumentProcessorInteface):
    def __init__(self, processors: Iterable[DocumentProcessorInteface]):
        self.__processors = processors

    def process(self, documents: Iterator[Document]) -> Iterator[Document]:
        result = documents
        for processor in self.__processors:
            result = processor.process(result)
        return result

    def __enter__(self):
        for processor in self.__processors:
            processor.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for processor in self.__processors:
            processor.__exit__(exc_type, exc_val, exc_tb)
