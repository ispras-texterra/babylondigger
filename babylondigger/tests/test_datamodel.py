import unittest

from operator import attrgetter

from babylondigger.datamodel import *


class TokenTest(unittest.TestCase):

    def test_negative_start(self):
        with self.assertRaises(AssertionError):
            Token(-1, 1)

    def test_empty_token(self):
        with self.assertRaises(AssertionError):
            Token(1, 1)

    def test_incorrect_span(self):
        with self.assertRaises(AssertionError):
            Token(3, 1)

    def test_multiword(self):
        token = Token(0, 1, text_replacement="replacement")
        self.assertTrue(token.is_multiword)
        self.assertEqual(token.text_replacement, "replacement")
        self.assertEqual(token.sub_index, 0)

    def test_copy(self):
        token = Token(0, 1)
        new_token = Token.with_morphology(token, Pos("N"), "lemma")
        self.assertEqual(new_token, Token(0, 1, Pos("N"), "lemma"))
        new_token = Token.with_syntax(token, deprel='deprel', head_sentence_index=1)
        self.assertEqual(new_token, Token(0, 1, deprel='deprel', head_sentence_index=1))
        new_token = Token.with_syntax(new_token, deprel='deprel', head_document_index=1)
        self.assertEqual(new_token, Token(0, 1, deprel='deprel', head_document_index=1))

    def test_multiword_token(self):
        token = Token(0, 1, text_replacement="replacement", sub_index=1)
        new_token = Token.with_morphology(token)
        self.assertEqual(token, new_token)
        new_token = Token.with_syntax(token)
        self.assertEqual(token, new_token)


class SentenceTest(unittest.TestCase):

    def test_negative_start(self):
        with self.assertRaises(AssertionError):
            Sentence(-1, 1)

    def test_empty_sentence(self):
        with self.assertRaises(AssertionError):
            Sentence(1, 1)

    def test_incorrect_span(self):
        with self.assertRaises(AssertionError):
            Sentence(3, 2)


class DocumentTest(unittest.TestCase):

    _text = "Hello, world! This is babylondigger!"
    _tokens = [Token(0, 5), Token(5, 6), Token(7, 12), Token(12, 13), Token(14, 18), Token(19, 21), Token(22, 35), Token(35, 36)]
    _multiword_tokens = [Token(0, 5), Token(5, 6), Token(7, 12), Token(12, 13), Token(14, 18), Token(19, 21), Token(35, 36),
              Token(22, 35, sub_index=1, text_replacement="digger"), Token(22, 35, text_replacement="babylon")]
    _syntax_tokens = [Token(0, 5, head_document_index=2), Token(5, 6, head_document_index=2), Token(7, 12, head_document_index=-1), Token(12, 13, head_document_index=2),
                      Token(14, 18, head_document_index=6), Token(19, 21, head_document_index=6), Token(22, 35, head_document_index=-1), Token(35, 36, head_document_index=6)]
    _syntax_sentence_tokens = [Token(0, 5, head_sentence_index=2), Token(5, 6, head_sentence_index=2),
                      Token(7, 12, head_sentence_index=-1), Token(12, 13, head_sentence_index=2),
                      Token(14, 18, head_sentence_index=2), Token(19, 21, head_sentence_index=2),
                      Token(22, 35, head_sentence_index=-1), Token(35, 36, head_sentence_index=2)]
    _sentences = [Sentence(0, 4), Sentence(4, 8)]

    def test_empty_document(self):
        doc = Document("", [], [])
        self.assertEqual(doc.text, "")
        self.assertEqual(doc.tokens_count, 0)
        self.assertEqual(doc.sentences_count, 0)

    def test_normal(self):
        doc = Document(DocumentTest._text, DocumentTest._tokens, DocumentTest._sentences)
        self.assertEqual(doc.text, DocumentTest._text)
        self.assertEqual(doc.tokens_count, len(DocumentTest._tokens))
        self.assertEqual(list(doc.tokens), DocumentTest._tokens)
        self.assertEqual(doc.sentences_count, len(DocumentTest._sentences))

    def test_multiword(self):
        doc = Document(DocumentTest._text, DocumentTest._multiword_tokens)
        self.assertEqual(doc.text, DocumentTest._text)
        self.assertEqual(doc.tokens_count, len(DocumentTest._multiword_tokens))
        sorted_tokens = sorted(DocumentTest._multiword_tokens, key=attrgetter('start', 'end', 'sub_index'))
        self.assertEqual(list(doc.tokens), sorted_tokens)

    def test_skipped_multiword_token(self):
        with self.assertRaises(ValueError):
            Document(DocumentTest._text, DocumentTest._multiword_tokens[:-1])

    def test_single_sentence(self):
        doc = Document(DocumentTest._text, DocumentTest._tokens)
        self.assertEqual(doc.text, DocumentTest._text)
        self.assertEqual(doc.sentences_count, 1)
        self.assertEqual(list(doc.get_sentence(0).tokens), DocumentTest._tokens)

    def test_excess_token(self):
        with self.assertRaises(ValueError):
            Document(DocumentTest._text, DocumentTest._tokens + [Token(37, 38)], DocumentTest._sentences)

    def test_tokens_intersection(self):
        with self.assertRaises(ValueError):
            Document(DocumentTest._text, DocumentTest._tokens + [Token(21, 23)], DocumentTest._sentences)

    def test_excess_sentence(self):
        with self.assertRaises(ValueError):
            Document(DocumentTest._text, DocumentTest._tokens, DocumentTest._sentences + [Sentence(8, 9)])

    def test_sentences_intersection(self):
        with self.assertRaises(ValueError):
            Document(DocumentTest._text, DocumentTest._tokens, DocumentTest._sentences + [Sentence(2, 5)])

    def test_out_of_sentence_tokens(self):
        with self.assertRaises(ValueError):
            Document(DocumentTest._text, DocumentTest._tokens, DocumentTest._sentences[1:])
        with self.assertRaises(ValueError):
            Document(DocumentTest._text, DocumentTest._tokens, DocumentTest._sentences[:1])
        with self.assertRaises(ValueError):
            Document(DocumentTest._text, DocumentTest._tokens, [Sentence(0, 4), Sentence(5, 8)])

    def test_syntax_normal(self):
        doc = Document(DocumentTest._text, DocumentTest._syntax_tokens, DocumentTest._sentences, validate_syntax=True)
        self.assertEqual(doc.text, DocumentTest._text)
        self.assertEqual(doc.tokens_count, len(DocumentTest._syntax_tokens))
        self.assertEqual(list(doc.tokens), DocumentTest._syntax_tokens)
        self.assertEqual(doc.sentences_count, len(DocumentTest._sentences))

    def test_syntax_multiple_roots(self):
        with self.assertRaises(ValueError):
            Document(DocumentTest._text, DocumentTest._syntax_tokens, validate_syntax=True)

    def test_syntax_out_of_sentence(self):
        with self.assertRaises(ValueError):
            Document(DocumentTest._text, DocumentTest._syntax_tokens[:-1] + [Token(35, 36, head_document_index=3)], DocumentTest._sentences, validate_syntax=True)

    def test_syntax_loop(self):
        with self.assertRaises(ValueError):
            Document(DocumentTest._text, DocumentTest._syntax_tokens[:-2] + [Token(22, 35, head_document_index=7), Token(35, 36, head_document_index=4)],
                     DocumentTest._sentences, validate_syntax=True)
            Document(DocumentTest._text, DocumentTest._syntax_tokens[:-1] + [Token(35, 36, head_document_index=7)],
                     DocumentTest._sentences, validate_syntax=True)

    def test_partly_specified_syntax(self):
        with self.assertRaises(ValueError):
            Document(DocumentTest._text, DocumentTest._syntax_tokens[:-1] + [Token(35, 36)], DocumentTest._sentences, validate_syntax=True)
        with self.assertRaises(ValueError):
            Document(DocumentTest._text, [Token(0, 5)] + DocumentTest._syntax_tokens[1:], DocumentTest._sentences, validate_syntax=True)

    def test_no_syntax_sentence(self):
        tokens = DocumentTest._syntax_tokens[:DocumentTest._sentences[0].end] + DocumentTest._tokens[DocumentTest._sentences[1].start:]
        doc = Document(DocumentTest._text, tokens, DocumentTest._sentences, validate_syntax=True)
        self.assertEqual(doc.text, DocumentTest._text)
        self.assertEqual(doc.tokens_count, len(tokens))
        self.assertEqual(list(doc.tokens), tokens)
        self.assertEqual(doc.sentences_count, len(DocumentTest._sentences))

    def test_sentence_syntax_notation(self):
        doc = Document(DocumentTest._text, DocumentTest._syntax_tokens, DocumentTest._sentences)
        sentence_doc = Document(DocumentTest._text, DocumentTest._syntax_sentence_tokens, DocumentTest._sentences)
        self.assertEqual(doc, sentence_doc)

    def test_mixed_syntax_notation(self):
        doc = Document(DocumentTest._text, DocumentTest._syntax_tokens, DocumentTest._sentences)
        for i in range(1, len(DocumentTest._tokens) - 1):
            tokens = DocumentTest._syntax_tokens[:i] + DocumentTest._syntax_sentence_tokens[i:]
            mixed_doc = Document(DocumentTest._text, tokens, DocumentTest._sentences)
            self.assertEqual(doc, mixed_doc)

    def test_overspecified_syntax_tokens(self):
        doc = Document(DocumentTest._text, DocumentTest._syntax_tokens, DocumentTest._sentences)
        tokens = DocumentTest._syntax_tokens[:-1] + [Token(35, 36, head_document_index=6, head_sentence_index=2)]
        overspecified = Document(DocumentTest._text, tokens, DocumentTest._sentences)
        self.assertEqual(doc, overspecified)
        tokens = DocumentTest._syntax_tokens[:-1] + [Token(35, 36, head_document_index=6, head_sentence_index=1)]
        with self.assertRaises(ValueError):
            Document(DocumentTest._text, tokens, DocumentTest._sentences)

    def test_update_tokens(self):
        doc = Document(DocumentTest._text, DocumentTest._tokens, DocumentTest._sentences)
        new_tokens = DocumentTest._tokens[:-1]
        with self.assertRaises(ValueError):
            Document.from_document(doc, new_tokens)

        with self.assertRaises(ValueError):
            Document.from_document(doc, new_tokens + [Token(6, 7)])

        new_tokens = DocumentTest._tokens[1:] + [Token(0, 5, Pos("N"), "lemma")]
        updated_doc = Document.from_document(doc, new_tokens)

        self.assertEqual(updated_doc.get_token(0), Token(0, 5, Pos("N"), "lemma"))
        self.assertEqual(list(updated_doc.tokens)[1:], list(doc.tokens)[1:])


class NavigableTokenTest(unittest.TestCase):

    _text = "Hello, world! This is babylondigger!"
    _tokens = [Token(0, 5), Token(5, 6), Token(7, 12), Token(12, 13), Token(14, 18), Token(19, 21), Token(22, 35),
               Token(35, 36)]
    _multiword_tokens = [Token(0, 5), Token(5, 6), Token(7, 12), Token(12, 13), Token(14, 18), Token(19, 21), Token(35, 36),
                         Token(22, 35, sub_index=1, text_replacement="digger"), Token(22, 35, text_replacement="babylon")]

    _sentences = [Sentence(0, 4), Sentence(4, 8)]
    _doc = Document(_text, _tokens, _sentences)

    def test_navigation(self):
        first_token = NavigableTokenTest._doc.get_token(0)
        token = NavigableTokenTest._doc.get_token(2)
        last_token = NavigableTokenTest._doc.get_token(-1)

        self.assertTrue(first_token.is_first)
        self.assertFalse(first_token.is_last)
        self.assertIsNone(first_token.previous)
        self.assertIsNotNone(first_token.next)

        self.assertFalse(last_token.is_first)
        self.assertTrue(last_token.is_last)
        self.assertIsNotNone(last_token.previous)
        self.assertIsNone(last_token.next)

        self.assertFalse(token.is_first)
        self.assertFalse(token.is_last)
        self.assertEqual(token.previous.next, token)
        self.assertEqual(token.next.previous, token)

    def test_token_text(self):
        token_texts = [token.text for token in NavigableTokenTest._doc.tokens]
        self.assertEqual(token_texts, ["Hello", ",", "world", "!", "This", "is", "babylondigger", "!"])

    def test_multiword_token_text(self):
        doc = Document(NavigableTokenTest._text, NavigableTokenTest._multiword_tokens)
        token_texts = list(map(lambda token: token.text, doc.tokens))
        self.assertEqual(token_texts, ["Hello", ",", "world", "!", "This", "is", "babylon", "digger", "!"])


class NavigableSentenceTest(unittest.TestCase):

    _text = "Hello, world! This is babylondigger!"
    _tokens = [Token(0, 5), Token(5, 6), Token(7, 12), Token(12, 13), Token(14, 18), Token(19, 21), Token(22, 35),
               Token(35, 36)]
    _sentences = [Sentence(0, 4), Sentence(4, 8)]
    _doc = Document(_text, _tokens, _sentences)

    def test_navigation(self):
        first_sentence = NavigableSentenceTest._doc.get_sentence(0)
        last_sentence = NavigableSentenceTest._doc.get_sentence(-1)

        self.assertTrue(first_sentence.is_first)
        self.assertFalse(first_sentence.is_last)
        self.assertIsNone(first_sentence.previous)
        self.assertEqual(first_sentence.next, last_sentence)

        self.assertFalse(last_sentence.is_first)
        self.assertTrue(last_sentence.is_last)
        self.assertEqual(last_sentence.previous, first_sentence)
        self.assertIsNone(last_sentence.next)

    def test_text(self):
        sentences = [sentence.text for sentence in NavigableSentenceTest._doc.sentences]
        self.assertEqual(sentences, ["Hello, world!", "This is babylondigger!"])

    def test_sentence_tokens(self):
        sentence = NavigableSentenceTest._doc.get_sentence(1)
        self.assertEqual(sentence.tokens_count, 4)

        sentence_tokens = list(sentence.tokens)
        self.assertEqual(sentence_tokens, NavigableSentenceTest._tokens[4:8])

    def test_sentence_token_text(self):
        sentence = NavigableSentenceTest._doc.get_sentence(1)
        tokens_text = [token.text for token in sentence.tokens]
        self.assertEqual(tokens_text, ["This", "is", "babylondigger", "!"])

    def test_sentence_boundary_token(self):
        sentence = NavigableSentenceTest._doc.get_sentence(0)
        last_token = sentence.get_token(-1)
        self.assertTrue(last_token.is_last)

        sentence = NavigableSentenceTest._doc.get_sentence(1)
        first_token = sentence.get_token(0)
        self.assertTrue(first_token.is_first)
