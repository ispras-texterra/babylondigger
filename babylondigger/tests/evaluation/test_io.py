import unittest
from io import StringIO

from babylondigger.evaluation.conllu_io import CoNLLUReader, CoNLLUWriter
import babylondigger.tests.evaluation.test_data as data
from babylondigger.datamodel import Document


class CoNLLReaderTest(unittest.TestCase):
    def test_empty_reader(self):
        with self.assertRaises(AssertionError):
            CoNLLUReader(-1)
        with self.assertRaises(AssertionError):
            CoNLLUReader(0)

    def test_normal(self):
        reader = CoNLLUReader(1)
        docs = list(reader.read_from_str(data.regular_document["CoNLL-U"]))
        self.assertEqual(len(docs), 1)
        doc = docs[0]
        self.assertEqual(doc, data.regular_document["document"])

    def test_empty_tokens(self):
        reader = CoNLLUReader(1)
        docs = list(reader.read_from_str(data.empty_token_document["CoNLL-U"]))
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].text, data.empty_token_document["text"])
        self.assertEqual(docs[0].sentences_count, 1)
        self.assertEqual(docs[0].tokens_count, data.empty_token_document["tokens count"])

    def test_multiword_token(self):
        reader = CoNLLUReader(1)
        docs = list(reader.read_from_str(data.multiword_token_document["CoNLL-U"]))
        self.assertEqual(len(docs), 1)
        doc = docs[0]
        self.assertEqual(doc, data.multiword_token_document["document"])

    def test_multiple_multiword_token(self):
        reader = CoNLLUReader(1)
        docs = list(reader.read_from_str(data.multiword_token_document2["CoNLL-U"]))
        self.assertEqual(len(docs), 1)
        doc = docs[0]
        self.assertEqual(doc, data.multiword_token_document2["document"])

    def test_several_docs(self):
        reader = CoNLLUReader(1)
        docs = list(reader.read_from_str(data.regular_document["CoNLL-U"] + data.empty_token_document["CoNLL-U"]))
        self.assertEqual(len(docs), 2)
        self.assertEqual(docs[0], data.regular_document["document"])
        self.assertEqual(docs[1].text, data.empty_token_document["text"])
        self.assertEqual(docs[1].sentences_count, 1)

    def test_several_sentences(self):
        self.several_sentences(2)
        self.several_sentences(3)

    def several_sentences(self, sentences_limit):
        reader = CoNLLUReader(sentences_limit)
        with StringIO(data.regular_document["CoNLL-U"] + data.empty_token_document["CoNLL-U"]) as f:
            docs = list(reader._read(f))
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].text, data.regular_document["text"] + data.empty_token_document["text"])
        self.assertEqual(docs[0].sentences_count, 2)
        self.assertEqual(docs[0].tokens_count, data.regular_document["document"].tokens_count + data.empty_token_document["tokens count"])


class CoNLLUWriterTest(unittest.TestCase):

    _writer = CoNLLUWriter()

    def test_empty_doc(self):
        self.assertEqual(CoNLLUWriterTest._writer.write_to_str(Document("", [])), "")

    def test_normal(self):
        self.assertEqual(CoNLLUWriterTest._writer.write_to_str(data.regular_document["document"]), _filter_comments(data.regular_document["CoNLL-U"]))

    def test_several_docs(self):
        doc = data.regular_document["document"]
        self.assertEqual(CoNLLUWriterTest._writer.write_to_str(doc, doc, doc), _filter_comments(data.regular_document["CoNLL-U"]) * 3)

    def test_several_sentences(self):
        reader = CoNLLUReader(5)
        docs = list(reader.read_from_str(data.regular_document["CoNLL-U"] * 3))[0]
        no_comments = _filter_comments(data.regular_document["CoNLL-U"])
        self.assertEqual(CoNLLUWriterTest._writer.write_to_str(docs), no_comments * 3)

    def test_multiword_token(self):
        no_comments = _filter_comments(data.multiword_token_document["CoNLL-U"])
        self.assertEqual(CoNLLUWriterTest._writer.write_to_str(data.multiword_token_document["document"]), no_comments)

    def test_multiple_multiword_token(self):
        no_comments = _filter_comments(data.multiword_token_document2["CoNLL-U"])
        self.assertEqual(CoNLLUWriterTest._writer.write_to_str(data.multiword_token_document2["document"]), no_comments)


def _filter_comments(conll: str):
    return '\n'.join(filter(lambda x: not x.startswith('#'), conll.split('\n')))
