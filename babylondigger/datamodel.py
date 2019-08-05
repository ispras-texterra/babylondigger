from typing import List, Tuple, Optional, Iterator, Iterable, Callable, Generic, TypeVar, Dict
from operator import attrgetter
from copy import deepcopy

from functools import total_ordering


@total_ordering
class Pos(object):
    def __init__(self, upos: str, xpos: str = None, feats: Dict[str, str] = None):
        self.__upos = upos
        self.__xpos = xpos
        self.__feats = feats.copy() if feats is not None else {}

    @property
    def upos(self) -> str:
        return self.__upos

    @property
    def xpos(self) -> str:
        return self.__xpos

    @property
    def feats(self) -> Dict[str, str]:
        return self.__feats.copy()

    def feature(self, key: str) -> str:
        return self.__feats.get(key, None)

    def __repr__(self):
        return "{{'upos': {}, 'xpos': {} 'feats': {}}}"\
            .format(self.__upos.__repr__(), self.__xpos.__repr__(), self.__feats.__repr__())

    def __eq__(self, other):
        if isinstance(other, Pos):
            return self.__upos == other.__upos and \
                   self.__xpos == other.xpos and \
                   self.__feats == other.__feats
        return NotImplemented

    def __lt__(self, other):
        if not isinstance(other, Pos):
            return NotImplemented
        return self.__tuple_view() < other.__tuple_view()

    def __hash__(self):
        return hash(self.__tuple_view())

    def __tuple_view(self):
        return self.__upos, self.__xpos, tuple(sorted(self.__feats.items()))


class TokenExtras(object):
    def __init__(self, extras: dict):
        self.__extras = deepcopy(extras)

    def __getitem__(self, item):
        return deepcopy(self.__extras[item])

    def __contains__(self, o):
        return self.__extras.__contains__(o)

    def __iter__(self):
        return self.__extras.__iter__()

    def keys(self):
        return self.__extras.keys()

    def __eq__(self, other):
        if isinstance(other, TokenExtras):
            return self.__extras == other.__extras
        raise NotImplemented

    def as_dict(self):
        return deepcopy(self.__extras)

    def update(self, updates: dict) -> 'TokenExtras':
        if not updates:
            return self
        result = self.__extras.copy()
        result.update(updates)
        return TokenExtras(result)


class Token(object):

    def __init__(self, start: int, end: int, pos: Pos = None, lemma: str = None,
                 deprel: str = None,
                 sub_index: int = 0, text_replacement: str = None, *,
                 head_document_index: int = None, head_sentence_index: int = None,
                 extras: TokenExtras = None):
        assert start >= 0, 'token start should be greater or equals to 0 (start: {}, end: {})'.format(start, end)
        assert start < end, 'token start should be less than token end (start: {}, end: {})'.format(start, end)
        if text_replacement is None:
            assert sub_index == 0, 'multiword token should contain text (start: {}, end: {})'.format(start, end)
        self.__start = start
        self.__end = end

        self.__pos = pos
        self.__lemma = lemma

        self.__head_document_index = head_document_index
        self.__head_sentence_index = head_sentence_index
        self.__deprel = deprel

        self.__sub_index = sub_index
        self.__text_replacement = text_replacement

        self.__extras = extras if extras is not None else TokenExtras({})

    @property
    def start(self) -> int:
        return self.__start

    @property
    def end(self) -> int:
        return self.__end

    @property
    def extras(self) -> TokenExtras:
        return self.__extras

    @property
    def pos(self) -> Pos:
        return self.__pos

    @property
    def lemma(self) -> str:
        return self.__lemma

    @property
    def has_head(self) -> bool:
        return self.__head_document_index is not None or self.__head_sentence_index is not None

    @property
    def _head_document_index(self) -> int:
        return self.__head_document_index

    @property
    def _head_sentence_index(self) -> int:
        return self.__head_sentence_index

    @property
    def is_root(self) -> bool:
        return self.__head_document_index == -1 or self.__head_sentence_index == -1

    @property
    def deprel(self) -> str:
        return self.__deprel

    @property
    def is_multiword(self) -> bool:
        return self.__text_replacement is not None

    @property
    def sub_index(self) -> int:
        return self.__sub_index

    @property
    def text_replacement(self) -> Optional[str]:
        return self.__text_replacement

    def coincides(self, other: 'Token') -> bool:
        if not isinstance(other, Token):
            raise TypeError()
        return self.__start == other.__start and self.__end == other.__end

    def intersects(self, other: 'Token') -> bool:
        if not isinstance(other, Token):
            raise TypeError()
        if self.__start >= other.__end or other.__start >= self.__end:
            return False  # not intersects
        if self.__start == other.start and self.__end == other.__end:
            return self.sub_index == other.sub_index  # at the same place
        return True

    def __eq__(self, other):
        if isinstance(other, Token):
            return self.__start == other.__start and self.__end == other.__end and \
                   self.pos == other.pos and self.lemma == other.lemma and \
                   self._head_document_index == other._head_document_index and \
                   self._head_sentence_index == self._head_sentence_index and self.deprel == other.deprel and \
                   self.sub_index == other.sub_index and self.text_replacement == other.text_replacement and \
                   self.extras == other.extras
        return NotImplemented

    def __repr__(self):
        if self.is_multiword:
            return "{{'start': {}, 'end': {}, 'sub_index': {}, 'text': {} 'pos': {}, 'lemma': {}, 'head_doc': {}, 'head_sent': {}, 'deprel': {}}}".format(
                self.__start, self.__end, self.__sub_index,
                self.__text_replacement.__repr__(), self.pos.__repr__(), self.lemma.__repr__(),
                self.__head_document_index, self._head_sentence_index, self.deprel.__repr__())
        return "{{'start': {}, 'end': {}, 'pos': {}, 'lemma': {}, 'head_doc': {}, 'head_sent': {}, 'deprel': {}}}".format(
            self.__start, self.__end, self.pos.__repr__(), self.lemma.__repr__(),
            self.__head_document_index, self.__head_sentence_index, self.deprel.__repr__())

    @classmethod
    def with_morphology(cls, token: 'Token', pos: Pos = None, lemma: str = None) -> 'Token':
        return Token(start=token.__start, end=token.__end,
                     sub_index=token.__sub_index, text_replacement=token.__text_replacement,
                     deprel=token.__deprel, head_document_index=token.__head_document_index,
                     head_sentence_index=token.__head_sentence_index,
                     pos=pos, lemma=lemma, extras=token.__extras)

    @classmethod
    def with_syntax(cls, token: 'Token', deprel: str = None, *, head_document_index: int = None, head_sentence_index: int = None) -> 'Token':
        return Token(start=token.__start, end=token.__end,
                     sub_index=token.__sub_index, text_replacement=token.__text_replacement,
                     pos=token.__pos, lemma=token.__lemma,
                     deprel=deprel, head_document_index=head_document_index, head_sentence_index=head_sentence_index,
                     extras=token.__extras)

    @classmethod
    def with_extras(cls, token: 'Token', extras: TokenExtras = None) -> 'Token':
        return Token(start=token.__start, end=token.__end,
                     sub_index=token.__sub_index, text_replacement=token.__text_replacement,
                     pos=token.__pos, lemma=token.__lemma,
                     deprel=token.__deprel, head_document_index=token.__head_document_index,
                     head_sentence_index=token.__head_sentence_index,
                     extras=extras)

    @classmethod
    def update_extras(cls, token: 'Token', updates: dict = None) -> 'Token':
        if updates is None:
            return token
        return Token.with_extras(token, token.__extras.update(updates))

    @classmethod
    def fill_heads(cls, token: 'Token', sentence_start: int):
        if token.__head_sentence_index is None and token.__head_document_index is None:
            return token
        if token.__head_sentence_index == -1 or token.__head_document_index == -1:
            return token.with_syntax(token, token.__deprel, head_document_index=-1, head_sentence_index=-1)
        head_document_index = token.__head_document_index
        head_sentence_index = token.__head_sentence_index
        if token.__head_document_index is None:
            head_document_index = token.__head_sentence_index + sentence_start
        elif token.__head_sentence_index is None:
            head_sentence_index = token.__head_document_index - sentence_start
        elif head_document_index != head_sentence_index + sentence_start:
            raise ValueError('Broken token {}. Mismatch syntactic head (document view: {}, sentence view: {}, sentence start: {})'
                             .format(token, head_document_index, head_sentence_index, sentence_start))
        return token.with_syntax(token, token.__deprel, head_document_index=head_document_index, head_sentence_index=head_sentence_index)


class Sentence(object):

    def __init__(self, start_token: int, end_token: int):
        assert start_token >= 0, 'sentence start should be greater or equals to 0 (start: {}, end: {})'.format(
            start_token, end_token)
        assert start_token < end_token, 'sentence start should be less than sentence end (start: {}, end: {})'.format(
            start_token, end_token)
        self.__start_token = start_token
        self.__end_token = end_token

    @property
    def start(self) -> int:
        return self.__start_token

    @property
    def end(self) -> int:
        return self.__end_token

    @property
    def len(self) -> int:
        return self.__end_token - self.__start_token

    def __eq__(self, other):
        if isinstance(other, Sentence):
            return self.__start_token == other.__start_token and self.__end_token == other.__end_token
        return NotImplemented

    def __repr__(self):
        return "{{'start_token': {}, 'end_token': {}}}".format(self.__start_token, self.__end_token)


_T = TypeVar('T')


class _NavigableItem(Generic[_T]):
    def __init__(self, getter: Callable[[int], _T], count: int, index: int):
        if index >= count:
            raise IndexError()
        self.__getter = getter
        self.__count = count
        self.__index = index % count

    @property
    def is_first(self) -> bool:
        return self.__index == 0

    @property
    def is_last(self) -> bool:
        return self.__index == self.__count - 1

    @property
    def next(self) -> Optional[_T]:
        return self.__getter(self.__index + 1) if not self.is_last else None

    @property
    def previous(self) -> Optional[_T]:
        return self.__getter(self.__index - 1) if not self.is_first else None


class _WordList(object):
    def __init__(self, text: str, tokens: List[Token], head_index_func: Callable[[Token], int], *,
                 sorted_sentences: List[Sentence] = None):
        sorted_tokens = sorted(tokens, key=attrgetter('start', 'end', 'sub_index'))
        non_intersects, index = _check_for_sequence(sorted_tokens, lambda token1, token2: token1.intersects(token2))
        if not non_intersects:
            raise ValueError(
                'tokens should not intersect ({}, {})'.format(sorted_tokens[index], sorted_tokens[index + 1]))

        no_skipped, index = _check_for_sequence(sorted_tokens, _skipped_token)
        if not no_skipped:
            raise ValueError(
                'multiword token is skipped ({}, {})'.format(sorted_tokens[index], sorted_tokens[index+1])
            )
        if sorted_sentences:
            sorted_tokens = _WordList.__fill_token_heads(sorted_tokens, sorted_sentences)
        self.__text = text
        self._tokens = sorted_tokens
        self._head_index_func = head_index_func

    def __getitem__(self, item: int) -> 'NavigableToken':
        return self.get_token(item)

    @property
    def document_text(self) -> str:
        return self.__text

    @property
    def tokens_count(self) -> int:
        return len(self._tokens)

    def get_token(self, index) -> 'NavigableToken':
        return NavigableToken(self, self._tokens[index], self._head_index_func, index)

    @property
    def tokens(self) -> Iterator['NavigableToken']:
        return map(self.get_token, range(self.tokens_count))

    @classmethod
    def __fill_token_heads(cls, sorted_tokens: List[Token], sorted_sentences: List[Sentence]) -> List[Token]:
        filled = []
        for sentence in sorted_sentences:
            for token in sorted_tokens[sentence.start:sentence.end]:
                filled.append(token.fill_heads(token, sentence.start))
        return filled

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__text == other.__text and self._tokens == other._tokens
        return NotImplemented


class DocumentToken(Token):
    def __init__(self, document_text: str, token: Token):
        Token.__init__(self, token.start, token.end,
                       pos=token.pos, lemma=token.lemma,
                       deprel=token.deprel,
                       sub_index=token.sub_index, text_replacement=token.text_replacement,
                       head_document_index=token._head_document_index, head_sentence_index=token._head_sentence_index,
                       extras=token.extras)
        self.__document_text = document_text

    @property
    def text(self) -> str:
        return self.text_replacement if self.is_multiword else self.doc_text

    @property
    def doc_text(self) -> str:
        return self.__document_text[self.start:self.end]

    @property
    def space_after(self) -> bool:
        return self.end == len(self.__document_text) or \
               self.__document_text[self.end:self.end + 1].isspace()

    @classmethod
    def with_morphology(cls, token: 'DocumentToken', pos: Pos = None, lemma: str = None) -> 'DocumentToken':
        return DocumentToken(token.__document_text, Token.with_morphology(token, pos, lemma))

    @classmethod
    def with_syntax(cls, token: 'DocumentToken', deprel: str = None, *,
                    head_document_index: int = None, head_sentence_index: int = None) -> 'DocumentToken':
        return DocumentToken(token.__document_text,
                             Token.with_syntax(token, deprel, head_document_index=head_document_index,
                                               head_sentence_index=head_sentence_index))

    @classmethod
    def with_extras(cls, token: 'DocumentToken', extras: TokenExtras = None) -> 'DocumentToken':
        return DocumentToken(token.__document_text, Token.with_extras(token, extras))

    @classmethod
    def update_extras(cls, token: 'DocumentToken', updates: dict = None) -> 'DocumentToken':
        return DocumentToken(token.__document_text, Token.update_extras(token, updates))


class NavigableToken(DocumentToken, _NavigableItem['NavigableToken']):
    def __init__(self, container: _WordList, token: Token, head_index_func: Callable[[Token], int], index: int):
        DocumentToken.__init__(self, container.document_text, token)
        _NavigableItem.__init__(self, container.get_token, container.tokens_count, index)
        self.__token_getter = container.get_token
        self.__head_index_func = head_index_func

    @property
    def head(self) -> Optional['NavigableToken']:
        if self.is_root:
            return None
        return self.__token_getter(self.head_index)

    @property
    def head_index(self):
        return self.__head_index_func(self)


class NavigableSentence(_WordList, _NavigableItem['NavigableSentence']):
    def __init__(self, document: 'Document', tokens: List[Token], index: int):
        _WordList.__init__(self, document.document_text, tokens, lambda token: token._head_sentence_index)
        _NavigableItem.__init__(self, document.get_sentence, document.sentences_count, index)

    @property
    def text(self) -> str:
        return self.document_text[self._tokens[0].start:self._tokens[-1].end]


class Document(_WordList):
    def __init__(self, text: str, tokens: List[Token], sentences: Iterable[Sentence] = None, validate_syntax: bool = False):

        if tokens:
            if sentences is None:
                sentences = [Sentence(0, len(tokens))]
            sentences = sorted(sentences, key=attrgetter('start', 'end'))

        _WordList.__init__(self, text, tokens, lambda token: token._head_document_index, sorted_sentences=sentences)

        if len(tokens) == 0:
            self.__sentences = []
            return

        if sentences[-1].end > len(tokens):
            raise ValueError('sentence refers to nonexistent token (sentence: {}, tokens count: {})'
                             .format(sentences[-1], len(tokens)))

        if sentences[0].start > 0:
            raise ValueError('there are some tokens outside of the sentence (tokens: {})'
                             .format(self._tokens[:sentences[0].start]))
        if sentences[-1].end < len(tokens):
            raise ValueError('there are some tokens outside of the sentence (tokens: {})'
                             .format(self._tokens[sentences[-1].end:]))

        alongside, index = _check_for_sequence(sentences, lambda x, y: x.end != y.start)
        if not alongside:
            raise ValueError('there are some tokens outside of the sentence (tokens: {})'
                             .format(self._tokens[sentences[index].end:sentences[index + 1].start]))

        non_intersects, index = _check_for_sequence(sentences, lambda x, y: x.end > y.start)
        if not non_intersects:
            raise ValueError('sentences should not intersect ({}, {})'
                             .format(sentences[index], sentences[index + 1]))
        self.__sentences = tuple(sentences)

        if validate_syntax:
            self.__validate_syntax()

    def __validate_syntax(self):
        def raise_error(msg: str, sentence: Sentence):
            raise ValueError(msg + ' in {} (tokens: {})'.format(
                        sentence, self._tokens[sentence.start:sentence.end]))

        def validate_no_syntax(sentence: Sentence):
            for token in self._tokens[sentence.start+1:sentence.end]:
                if token.has_head:
                    raise_error('syntax tree is partly specified', sentence)

        def validate_syntax(sentence: Sentence):
            root = None
            visited = [None] * sentence.len
            for i in range(sentence.len):
                token = self._tokens[sentence.start + i]
                if not token.has_head:
                    raise_error('syntax tree is partly specified', sentence)
                if token.is_root:
                    if root is not None:
                        raise_error('multiple roots ({}, {})'.format(root, token), sentence)
                    root = token
                    if visited[i] is None:
                        visited[i] = i
                    continue
                if visited[i] is not None:
                    continue
                visited[i] = i
                while token.has_head and not token.is_root:
                    head = token._head_sentence_index
                    if head < 0 or head >= sentence.len:
                        raise_error('token {} head refers outside of the sentence'.format(token), sentence)
                    if visited[head] is None:
                        visited[head] = i
                    elif visited[head] == i:
                        raise_error('loop in syntax tree', sentence)
                    else:
                        break
                    token = self._tokens[token._head_document_index]

        for sentence in self.__sentences:
            if not self._tokens[sentence.start].has_head:
                validate_no_syntax(sentence)
            else:
                validate_syntax(sentence)

    @property
    def text(self) -> str:
        return self.document_text

    @property
    def sentences_count(self) -> int:
        return len(self.__sentences)

    def get_sentence(self, index) -> NavigableSentence:
        sentence = self.__sentences[index]
        return NavigableSentence(self, self._tokens[sentence.start:sentence.end], index)

    @property
    def sentences(self) -> Iterator[NavigableSentence]:
        return map(lambda index: self.get_sentence(index), range(self.sentences_count))

    @property
    def sentences_boundaries(self) -> Iterable[Sentence]:
        return self.__sentences

    def __str__(self):
        result = self.document_text[0:self._tokens[0].start]
        pointer = self._tokens[0].start
        for sentence in self.__sentences:
            result += self.document_text[pointer:self._tokens[sentence.start].start] + "["
            pointer = self._tokens[sentence.start].start
            for i in range(sentence.start, sentence.end):
                token = self._tokens[i]
                result += self.document_text[pointer:token.start]
                result += "[" + self.document_text[token.start:token.end]
                if token.pos is not None:
                    result += "|'pos':{}".format(token.pos)
                if token.lemma is not None:
                    result += "|'lemma':'{}'".format(token.lemma)
                result += "]"
                pointer = self._tokens[i].end
            result += "]"
        result += self.document_text[self._tokens[-1].end:]
        return result

    def __repr__(self):
        return "{{'text':{}, 'tokens':{}, 'sentences':{}".format(
            self.document_text.__repr__(), self._tokens.__repr__(), self.__sentences.__repr__())

    def __eq__(self, other):
        if isinstance(other, Document):
            return super(Document, self).__eq__(other) and self.__sentences == other.__sentences
        return NotImplemented

    @classmethod
    def from_document(cls, document: 'Document', tokens: List[Token]):
        sorted_tokens = sorted(tokens, key=attrgetter('start', 'end', 'sub_index'))
        if any(not old.coincides(new) for (old, new) in zip(document._tokens, sorted_tokens)):
            raise ValueError("new tokens should coincide old ones")
        return cls(document.document_text, sorted_tokens, document.__sentences)


_O = TypeVar("O")


def _check_for_sequence(sorted_list: List[_O], fail_condition: Callable[[_O, _O], bool]) -> Tuple[bool, Optional[int]]:
    for i in range(len(sorted_list) - 1):
        if fail_condition(sorted_list[i], sorted_list[i+1]):
            return False, i
    return True, None


def _skipped_token(token1: Token, token2: Token):
    if token2.is_multiword:
        if token1.is_multiword and token2.coincides(token1):
            return token2.sub_index != token1.sub_index + 1
        return token2.sub_index != 0
    return False
