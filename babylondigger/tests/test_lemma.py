import unittest

from babylondigger.lemmas.transformation_rules import LemmaSuffixTransformation
from babylondigger.lemmas.transformation_rules import LemmaTransformationTrie


suffix1 = 'ого'
suffix2 = ''
suffix3 = 'гого'

lemma1 = 'хліб'
lemma2 = 'другий'
lemma3 = '.'
lemma4 = 'його'
lemma5 = 'бути'
lemma6 = 'сцена'


class LemmaSuffixTransformationTest(unittest.TestCase):

    def test_applicable(self):
        lms1 = LemmaSuffixTransformation(suffix1, '')
        lms2 = LemmaSuffixTransformation(suffix2, '')
        lms3 = LemmaSuffixTransformation(suffix3, '')
        self.assertTrue(lms1.is_applicable('столичного'))  # real suffix
        self.assertTrue(lms2.is_applicable(','))  # no suffix for PUNC
        self.assertTrue(lms2.is_applicable('столичного'))  # no suffix
        self.assertFalse(lms1.is_applicable(','))  # wrong suffix
        self.assertFalse(lms3.is_applicable(','))  # wrong suffix
        self.assertFalse(lms3.is_applicable('столичного'))  # wrong suffix similar to the right one

    def test_rule(self):
        lms = LemmaSuffixTransformation(suffix1, '')
        b = lms.get_rule('хлібові', lemma1)  # smth -> ~
        self.assertEqual(b.word_suffix, 'ові')
        self.assertEqual(b.lemma_suffix, '')
        b = lms.get_rule('другого', lemma2)  # smth -> smth
        self.assertEqual(b.word_suffix, 'ого')
        self.assertEqual(b.lemma_suffix, 'ий')
        b = lms.get_rule('.', lemma3)  # ~ -> ~ PUNC
        self.assertEqual(b.word_suffix, '')
        self.assertEqual(b.lemma_suffix, '')
        b = lms.get_rule('його', lemma4)  # ~ -> ~
        self.assertEqual(b.word_suffix, '')
        self.assertEqual(b.lemma_suffix, '')
        b = lms.get_rule('єсть', lemma5)  # whole_word
        self.assertEqual(b.word_suffix, 'єсть')
        self.assertEqual(b.lemma_suffix, 'бути')
        b = lms.get_rule('сцен', lemma6)  # ~ -> smth
        self.assertEqual(b.word_suffix, '')
        self.assertEqual(b.lemma_suffix, 'а')


class LemmaTransformationTrieTest(unittest.TestCase):

    def test_applicable_rules(self):
        lms = LemmaSuffixTransformation(suffix1, '')
        ltt = LemmaTransformationTrie()
        it = (lms.get_rule('хлібові', lemma1), 0), (lms.get_rule('другого', lemma2), 1), (lms.get_rule('.', lemma3), 2),\
             (lms.get_rule('єсть', lemma5), 3), (lms.get_rule('сцен', lemma6), 4)
        build = ltt.build(it)
        check1 = [(lms.get_rule('.', lemma3), 2), (lms.get_rule('сцен', lemma6), 4), (lms.get_rule('другого', lemma2), 1)]
        check2 = [(lms.get_rule('.', lemma3), 2), (lms.get_rule('сцен', lemma6), 4), (lms.get_rule('єсть', lemma5), 3)]
        check3 = [(lms.get_rule('.', lemma3), 2), (lms.get_rule('сцен', lemma6), 4)]
        self.assertEqual(build.get_applicable_rules('його'), check1)  # applicable suffixes for word with no suffix
        self.assertEqual(build.get_applicable_rules('єсть'), check2)  # whole word
        self.assertEqual(build.get_applicable_rules('атласі'), check3)  # applicable suffixes for word with suffix
        self.assertEqual(build.get_applicable_rules('.'), check3)  # applicable suffixes for PUNC


