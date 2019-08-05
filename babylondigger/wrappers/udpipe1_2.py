from typing import Iterator, List, Callable

import ufal.udpipe as ud

from babylondigger.datamodel import Document, NavigableSentence, Token, Pos
from babylondigger.tagger import TaggerInterface
from babylondigger.parser import ParserInterface
from babylondigger.processor import DocumentProcessorInteface


class _AbstractUDPipeProcessor(DocumentProcessorInteface):

    def __init__(self, model: ud.Model, options: str = ''):
        self.__ud_pipe = self.ud_pipe_processor(model, options)

    def process(self, documents: Iterator[Document]) -> Iterator[Document]:
        return map(self._process, documents)

    def _process(self, document: Document) -> Document:
        result_tokens = []
        for sentence in document.sentences:
            ud_sent = convert_to_ud(sentence)
            self.__ud_pipe(ud_sent)
            result_tokens.extend(convert_from_ud(sentence, ud_sent))
        return document.from_document(document, result_tokens)

    @classmethod
    def ud_pipe_processor(cls, model: ud.Model, options: str = '') -> Callable[[ud.Sentence], None]:
        pass

    @classmethod
    def load_udpipe(cls, path, options: str = ''):
        model = ud.Model.load(path)
        return cls(model, options)


class UDPipeTagger(_AbstractUDPipeProcessor, TaggerInterface):

    def __init__(self, model: ud.Model, options: str = ''):
        _AbstractUDPipeProcessor.__init__(self, model, options)
        self.tag = self.process

    @classmethod
    def ud_pipe_processor(cls, model: ud.Model, options: str = ''):
        return lambda ud_sent: model.tag(ud_sent, options)


class UDPipeParser(_AbstractUDPipeProcessor, ParserInterface):

    def __init__(self, model: ud.Model, options: str = ''):
        _AbstractUDPipeProcessor.__init__(self, model, options)
        self.parse = self.process

    @classmethod
    def ud_pipe_processor(cls, model: ud.Model, options: str = '') -> Callable[[ud.Sentence], None]:
        return lambda ud_sent: model.parse(ud_sent, options)


class UDPipeTaggerParser(_AbstractUDPipeProcessor):

    def __init__(self, model: ud.Model, options: str = ''):
        _AbstractUDPipeProcessor.__init__(self, model, options)
        self.parse = self.process

    @classmethod
    def ud_pipe_processor(cls, model: ud.Model, options: str = '') -> Callable[[ud.Sentence], None]:
        def process(ud_sent: ud.Sentence):
            model.tag(ud_sent, options)
            model.parse(ud_sent, options)
        return process


def convert_to_ud(sentence: NavigableSentence) -> ud.Sentence:
    result = ud.Sentence()
    for token in sentence:
        word = result.addWord(token.text)  # type: ud.Word
        if token.lemma:
            word.lemma = token.lemma
        if token.pos:
            if token.pos.upos:
                word.upostag = token.pos.upos
            if token.pos.xpos:
                word.xpostag = token.pos.xpos
            if token.pos.feats:
                word.feats = "|".join(sorted("{}={}".format(*item) for item in token.pos.feats.items()))
    return result


def convert_from_ud(sentence: NavigableSentence, ud_sent: ud.Sentence) -> List[Token]:

    def non_empty(value):
        return value if value else None

    result = []
    for token, word in zip(sentence, ud_sent.words[1:]):  # skip <root> in ud_sent
        feats = None
        if word.feats:
            feats = {entry[0]: entry[1] for entry in map(lambda s: s.split("="), word.feats.split("|"))}
        pos = Pos(non_empty(word.upostag), non_empty(word.xpostag), feats) if word.upostag or word.xpostag or feats else None
        head = None
        if word.head is not None and word.head >= 0:
            head = word.head - 1
        new_token = Token.with_morphology(token, pos, non_empty(word.lemma))
        new_token = Token.with_syntax(new_token, non_empty(word.deprel), head_sentence_index=head)
        result.append(new_token)
    return result
