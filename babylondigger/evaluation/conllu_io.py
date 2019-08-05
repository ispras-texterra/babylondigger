from typing import TextIO, Generator, Dict, List, Iterable, Set, Optional, Callable

from io import StringIO
from babylondigger.datamodel import Document, Sentence, Token, Pos, NavigableToken
from babylondigger.evaluation.conll17_ud_eval import ID, FORM, LEMMA, UPOS, XPOS, FEATS, MISC, HEAD, DEPREL


########## reader ##########

class _AbstractReaderState(object):
    def __init__(self, tokens: List[Token], sentences: List[Sentence], line_parser: Callable[[List[str]], 'InternalToken']):
        self._tokens = tokens
        self._sentences = sentences
        self._line_parser = line_parser

    @property
    def document(self) -> Document:
        raise NotImplementedError()

    @property
    def is_empty(self) -> bool:
        raise NotImplementedError()

    def add_token(self, line: str) -> '_AbstractReaderState':
        splitted_line = line.split('\t')
        parsed_token = self._line_parser(splitted_line)
        return self._process_token(parsed_token)

    def _process_token(self, parsed_token: 'InternalToken') -> '_AbstractReaderState':
        pass

    def end_of_sentence(self) -> int:
        start = self._sentences[-1].end if len(self._sentences) > 0 else 0
        end = len(self._tokens)
        self._sentences.append(Sentence(start, end))
        return len(self._sentences)

    @classmethod
    def _compute_head(cls, parsed_head):
        if parsed_head is None:
            return None
        if parsed_head == 0:
            return -1
        return parsed_head - 1


class _ReaderState(_AbstractReaderState):
    def __init__(self, line_parser: Callable[[List[str]], 'InternalToken']):
        _AbstractReaderState.__init__(self, [], [], line_parser)
        self._text = ""

    @property
    def document(self) -> Document:
        return Document(self._text, self._tokens, self._sentences, validate_syntax=True)

    @property
    def is_empty(self) -> bool:
        return len(self._text) == 0 and len(self._tokens) == 0

    def _process_token(self, parsed_token: 'InternalToken') -> _AbstractReaderState:
        if '.' in parsed_token.id:  # empty token
            return self
        start = len(self._text)
        self._text += parsed_token.text
        end = len(self._text)
        if parsed_token.space_after:
            self._text += " "

        if '-' in parsed_token.id:  # multiword token
            indexes = parsed_token.id.split('-')
            assert len(indexes) == 2
            tokens_count = int(indexes[1]) - int(indexes[0]) + 1
            return _MultiWordReaderState(self, tokens_count, start, end)
        self._tokens.append(Token(start=start, end=end, pos=parsed_token.pos, lemma=parsed_token.lemma,
                                  head_sentence_index=self._compute_head(parsed_token.head), deprel=parsed_token.deprel))
        return self


class _MultiWordReaderState(_AbstractReaderState):
    def __init__(self, state: _AbstractReaderState, tokens_count: int, token_start: int, token_end: int):
        _AbstractReaderState.__init__(self, state._tokens, state._sentences, state._line_parser)
        self._parent_state = state
        self._tokens_count = tokens_count
        self.__tokens_added = 0
        self._token_start = token_start
        self._token_end = token_end

    def _process_token(self, parsed_token: 'InternalToken') -> _AbstractReaderState:
        self._tokens.append(Token(start=self._token_start, end=self._token_end,
                                  pos=parsed_token.pos, lemma=parsed_token.lemma,
                                  head_sentence_index=self._compute_head(parsed_token.head), deprel=parsed_token.deprel,
                                  sub_index=self.__tokens_added, text_replacement=parsed_token.text))
        self.__tokens_added += 1
        if self._tokens_count == self.__tokens_added:
            return self._parent_state
        return self

    def end_of_sentence(self) -> int:
        raise RuntimeError('End of sentence while processing multi-word token')


class CoNLLUReader(object):
    def __init__(self, doc_sentences_limit: int = 50, columns: Iterable[int] = (LEMMA, UPOS, XPOS, FEATS, DEPREL)):
        assert doc_sentences_limit > 0
        self.__limit = doc_sentences_limit
        _columns = set(columns)
        if DEPREL in _columns:
            _columns.add(HEAD)
        self.__line_parser = _line_parser(_columns)

    def read_from_file(self, file_name: str) -> Generator[Document, None, None]:
        return self._read(open(file_name, encoding="utf-8"))

    def read_from_str(self, text: str) -> Generator[Document, None, None]:
        return self._read(StringIO(text))

    def _read(self, file: TextIO) -> Generator[Document, None, None]:
        try:
            state = _ReaderState(self.__line_parser)
            for line in file:
                line = line.strip("\r\n")
                if line.startswith("#"):
                    continue
                if len(line) == 0: # end of sentence
                    length = state.end_of_sentence()
                    if length >= self.__limit:
                        yield state.document
                        state = _ReaderState(self.__line_parser)
                    continue
                state = state.add_token(line)
            if not state.is_empty:
                yield state.document
        finally:
            file.close()


def _line_parser(columns: Set[int]):
    class InternalToken(object):
        def __init__(self, index, text, pos, lemma, head, deprel, space_after):
            self.id = index
            self.text = text
            self.pos = pos
            self.lemma = lemma
            self.head = int(head) if head is not None else None
            self.deprel = deprel
            self.space_after = space_after

    def _column_extractor(column):
        return (lambda x: x[column]) if column in columns else (lambda x: None)

    if UPOS in columns or XPOS in columns or FEATS in columns:
        upos_extractor = _column_extractor(UPOS)
        xpos_extractor = _column_extractor(XPOS)
        feats_extractor = _column_extractor(FEATS)
        pos_extractor = lambda x: _parse_pos(upos_extractor(x), xpos_extractor(x), feats_extractor(x))
    else:
        pos_extractor = lambda x: None

    lemma_extractor = _column_extractor(LEMMA)

    head_extractor = _column_extractor(HEAD)
    deprel_extractor = _column_extractor(DEPREL)

    def _parse_line(splitted_line: List[str]) -> 'InternalToken':

        token_repr = [value if not value == "_" else None for value in splitted_line]
        return InternalToken(
            index=token_repr[ID],
            text=token_repr[FORM] if token_repr[FORM] is not None else '_',  # looks like '_' is valid token text
            pos=pos_extractor(token_repr),
            lemma=lemma_extractor(token_repr),
            head=head_extractor(token_repr),
            deprel=deprel_extractor(token_repr),
            space_after=_parse_to_dict(token_repr[MISC]).get("SpaceAfter") is None)
    return _parse_line


default_line_parser = _line_parser({LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL})


def _parse_pos(upos: Optional[str], xpos: Optional[str], feats: Optional[str]) -> Pos:
    return Pos(upos, xpos, _parse_to_dict(feats))


def _parse_to_dict(feats: Optional[str]) -> Dict[str, str]:
    if feats is None:
        return {}
    return {entry[0]: entry[1] for entry in map(lambda s: s.split("="), feats.split("|"))}


########## writer ##########


class _AbstractWriterState(object):

    def write_token(self, token: NavigableToken) -> '_AbstractWriterState':
        pass

    def write_end_of_sentence(self):
        pass


class _RegularTokenWriterState(_AbstractWriterState):

    def __init__(self, file: TextIO, column_extractors):
        _AbstractWriterState.__init__(self)
        self.__column_extractors = column_extractors
        self.__index = 0
        self.__file = file

    def write_token(self, token: NavigableToken) -> _AbstractWriterState:
        if token.is_multiword:
            return _MultiwordTokenWriterState(self, token, self.__index)
        self.write_regular(token)
        return self

    def write_regular(self, token: NavigableToken, write_space_after=True):
        self.write_to_file(self._to_conll_str(self.__index, token, write_space_after))
        self.__index += 1

    def write_end_of_sentence(self):
        super().write_end_of_sentence()
        self.__file.write("\n")
        self.__index = 0

    def write_to_file(self, line):
        self.__file.write(line)

    def _to_conll_str(self, index: int, token: NavigableToken, write_space_after: bool = True) -> str:
        line_args = to_conll(self.__column_extractors, index, token, write_space_after)
        return _format_line(line_args)


class _MultiwordTokenWriterState(_AbstractWriterState):

    def __init__(self, parent_state: _RegularTokenWriterState, first_token: NavigableToken, current_index: int):
        _AbstractWriterState.__init__(self)
        self.__parent_state = parent_state
        self.__start_index = current_index + 1
        self.__token = first_token
        self.__tokens = [first_token]

    def write_token(self, token: NavigableToken) -> _AbstractWriterState:
        if self.__token.coincides(token):
            self.__tokens.append(token)
            return self
        self._write_multiword_token()
        return self.__parent_state.write_token(token)

    def write_end_of_sentence(self):
        self._write_multiword_token()
        return self.__parent_state.write_end_of_sentence()

    def _write_multiword_token(self):
        line_args = ["_"] * 10
        line_args[ID] = "{}-{}".format(self.__start_index, self.__start_index + len(self.__tokens) - 1)
        line_args[FORM] = self.__token.doc_text
        if not self.__token.space_after:
            line_args[MISC] = "SpaceAfter=No"
        self.__parent_state.write_to_file(_format_line(line_args))  # write header token (start-end  text    _ ...)
        for token in self.__tokens:
            self.__parent_state.write_regular(token, False)


def _column_extractors(columns: Set[int]):
    def _extractor(f, column):
        return f if column in columns else lambda x: None

    extractors = [None] * 10
    extractors[LEMMA] = _extractor(lambda x: x.lemma, LEMMA)
    extractors[UPOS] = _extractor(lambda x: x.pos.upos, UPOS)
    extractors[XPOS] = _extractor(lambda x: x.pos.xpos, XPOS)
    extractors[FEATS] = _extractor(lambda x: x.pos.feats, FEATS)
    extractors[HEAD] = _extractor(lambda x: x.head_index, HEAD)
    extractors[DEPREL] = _extractor(lambda x: x.deprel, DEPREL)
    return extractors


default_column_extractors = _column_extractors({LEMMA, UPOS, XPOS, FEATS, HEAD, DEPREL})


class CoNLLUWriter(object):
    def __init__(self, columns: Iterable[int] = (LEMMA, UPOS, XPOS, FEATS, DEPREL)):
        _columns = set(columns)
        if DEPREL in _columns:
            _columns.add(HEAD)
        self._column_extractors = _column_extractors(_columns)

    def write_to_file(self, file_name: str, *args: Document) -> None:
        with open(file_name, 'w', encoding="utf-8") as file:
            self._write(file, *args)

    def write_to_str(self, *args: Document) -> str:
        with StringIO() as file:
            self._write(file, *args)
            return file.getvalue()

    def _write(self, file: TextIO, *args: Document) -> None:
        for doc in args:
            self.__write_CoNLL_U(file, doc)

    def __write_CoNLL_U(self, file: TextIO, doc: Document):
        state = _RegularTokenWriterState(file, self._column_extractors)
        for sentence in doc.sentences:
            for token in sentence:
                state = state.write_token(token)
            state.write_end_of_sentence()


def to_conll(column_extractors, index: int, token: NavigableToken, write_space_after: bool = True, fake_heads: bool = True) -> List[str]:
    def __shield_none(s: str) -> str:
        return s if s is not None else "_"

    def __feats_to_string(feats: Dict) -> str:
        if len(feats) == 0:
            return "_"
        return "|".join(sorted("{}={}".format(*item) for item in feats.items()))

    def __normalize_index(index: int):
        if index == -1:
            return '0'
        return index + 1

    line_args = ["_"] * 10
    line_args[ID] = str(__normalize_index(index))
    line_args[FORM] = token.text
    line_args[LEMMA] = __shield_none(column_extractors[LEMMA](token))
    if token.pos is not None:
        line_args[UPOS] = __shield_none(column_extractors[UPOS](token))
        line_args[XPOS] = __shield_none(column_extractors[XPOS](token))
        line_args[FEATS] = __feats_to_string(column_extractors[FEATS](token))

    head_value = column_extractors[HEAD](token)

    if head_value is not None:
        line_args[HEAD] = str(__normalize_index(head_value))
        line_args[DEPREL] = __shield_none(column_extractors[DEPREL](token))
    elif fake_heads:
        # hack for evaluation script that doesn't
        # support '_' constants in HEAD column
        line_args[HEAD] = str(index)

    if write_space_after and not token.space_after:
        line_args[MISC] = "SpaceAfter=No"

    return line_args


__connl_line = "\t".join(["{}"] * 10) + "\n"


def _format_line(field_values: List[str]):
    return __connl_line.format(*field_values)
