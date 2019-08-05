from babylondigger.datamodel import Document, Token, Sentence, Pos


regular_document = {
    "CoNLL-U": '''# sent_id = train-s8
# text = Закончил Харьковский, а затем и Петроградский университет.
1	Закончил	ЗАКОНЧИТЬ	VERB	VBC	Aspect=Perf|Gender=Masc|Mood=Ind|Number=Sing|Tense=Past|VerbForm=Fin	0	root	_	_
2	Харьковский	ХАРЬКОВСКИЙ	ADJ	JJL	Animacy=Inan|Case=Acc|Gender=Masc|Number=Sing	8	amod	_	SpaceAfter=No
3	,	,	PUNCT	,	_	7	punct	_	_
4	а	А	CCONJ	CC	_	7	cc	_	_
5	затем	ЗАТЕМ	ADV	RB	_	7	advmod	_	_
6	и	И	PART	UH	_	7	discourse	_	_
7	Петроградский	ПЕТРОГРАДСКИЙ	ADJ	JJL	Animacy=Inan|Case=Acc|Gender=Masc|Number=Sing	2	conj	_	_
8	университет	УНИВЕРСИТЕТ	NOUN	NN	Animacy=Inan|Case=Acc|Gender=Masc|Number=Sing	1	obj	_	SpaceAfter=No
9	.	.	PUNCT	.	_	1	punct	_	_

''',
    "text": 'Закончил Харьковский, а затем и Петроградский университет. ',
    "document": Document('Закончил Харьковский, а затем и Петроградский университет. ', [
    Token(0, 8, Pos('VERB', 'VBC', {'Aspect': 'Perf', 'Gender': 'Masc', 'Mood': 'Ind', 'Number': 'Sing', 'Tense': 'Past', 'VerbForm': 'Fin'}),
          'ЗАКОНЧИТЬ', 'root', head_sentence_index=-1),
    Token(9, 20, Pos('ADJ', 'JJL', {'Animacy': 'Inan', 'Case': 'Acc', 'Gender': 'Masc', 'Number': 'Sing'}),
          'ХАРЬКОВСКИЙ', 'amod', head_sentence_index=7),
    Token(20, 21, Pos('PUNCT', ','), ',', 'punct', head_sentence_index=6),
    Token(22, 23, Pos('CCONJ', 'CC'), 'А', 'cc', head_sentence_index=6),
    Token(24, 29, Pos('ADV', 'RB'), 'ЗАТЕМ', 'advmod', head_sentence_index=6),
    Token(30, 31, Pos('PART', 'UH'), 'И', 'discourse', head_sentence_index=6),
    Token(32, 45, Pos('ADJ', 'JJL', {'Animacy': 'Inan', 'Case': 'Acc', 'Gender': 'Masc', 'Number': 'Sing'}),
          'ПЕТРОГРАДСКИЙ', 'conj', head_sentence_index=1),
    Token(46, 57, Pos('NOUN', 'NN', {'Animacy': 'Inan', 'Case': 'Acc', 'Gender': 'Masc', 'Number': 'Sing'}),
          'УНИВЕРСИТЕТ', 'obj', head_sentence_index=0),
    Token(57, 58, Pos('PUNCT', '.'), '.', 'punct', head_sentence_index=0)], [Sentence(0, 9)])
}

empty_token_document = {
    "CoNLL-U": '''# source = 2011Alpinizm.xml 29
# text = До сих пор идут споры о том, достигли они вершины или нет
# sent_id = 191
1	До	до	ADP	_	_	4	advmod	4:advmod	_
2	сих	сей	DET	_	Case=Gen|Number=Plur	1	fixed	1:fixed	_
3	пор	пора	NOUN	_	Animacy=Inan|Case=Gen|Gender=Fem|Number=Plur	2	fixed	2:fixed	_
4	идут	идти	VERB	_	Aspect=Imp|Mood=Ind|Number=Plur|Person=3|Tense=Pres|VerbForm=Fin|Voice=Act	0	root	0:root	_
5	споры	спор	NOUN	_	Animacy=Inan|Case=Nom|Gender=Masc|Number=Plur	4	nsubj	4:nsubj	_
6	о	о	ADP	_	_	7	case	7:case	_
7	том	то	PRON	_	Animacy=Inan|Case=Loc|Gender=Neut|Number=Sing	5	nmod	5:nmod	SpaceAfter=No
8	,	,	PUNCT	_	_	7	punct	7:punct	_
9	достигли	достигать	VERB	_	Aspect=Perf|Mood=Ind|Number=Plur|Tense=Past|VerbForm=Fin|Voice=Act	7	acl	7:acl	_
10	они	они	PRON	_	Case=Nom|Number=Plur|Person=3	9	nsubj	9:nsubj	_
11	вершины	вершина	NOUN	_	Animacy=Inan|Case=Gen|Gender=Fem|Number=Sing	9	obl	9:obl	_
12	или	или	CCONJ	_	_	9	orphan	13.1:cc	_
13	нет	нет	PART	_	_	9	orphan	13.1:advmod	SpaceAfter=No
13.1	_	_	_	_	_	_	_	9:conj	_

''',
    "text": 'До сих пор идут споры о том, достигли они вершины или нет',
    "tokens count": 13
}

################ multiword token document ######################

multiword_token_document = {
    "CoNLL-U": '''# sent_id = es-dev-001-s72
# text = También comenzó a emitirse publicidad comercial y programas estadounidenses.
1	También	también	ADV	_	_	6	advmod	_	_
2	comenzó	comenzar	AUX	_	Mood=Ind|Number=Sing|Person=3|Tense=Past|VerbForm=Fin	6	aux	_	_
3	a	a	ADP	_	_	4	mark	_	_
4-5	emitirse	_	_	_	_	_	_	_	_
4	emitir	emitir	VERB	_	VerbForm=Inf	6	cop	_	_
5	se	él	PRON	_	Case=Acc,Dat|Person=3|PrepCase=Npr|PronType=Prs|Reflex=Yes	4	iobj	_	_
6	publicidad	publicidad	NOUN	_	Gender=Fem|Number=Sing	0	root	_	_
7	comercial	comercial	ADJ	_	Number=Sing	6	amod	_	_
8	y	y	CCONJ	_	_	9	cc	_	_
9	programas	programa	NOUN	_	Gender=Masc|Number=Plur	6	conj	_	_
10	estadounidenses	estadounidense	ADJ	_	Number=Plur	9	amod	_	SpaceAfter=No
11	.	.	PUNCT	_	_	6	punct	_	_

''',
    "text": 'También comenzó a emitirse publicidad comercial y programas estadounidenses. ',
    "document": Document('También comenzó a emitirse publicidad comercial y programas estadounidenses. ', [
        Token(0, 7, Pos('ADV'), 'también', 'advmod', head_sentence_index=5),
        Token(8, 15, Pos('AUX', feats = {'Mood': 'Ind', 'Number': 'Sing', 'Person': '3', 'Tense': 'Past', 'VerbForm': 'Fin'}),
              'comenzar', 'aux', head_sentence_index=5),
        Token(16, 17, Pos('ADP'), 'a', 'mark', head_sentence_index=3),
        Token(18, 26, Pos('VERB', feats={'VerbForm': 'Inf'}), 'emitir', 'cop', sub_index=0, text_replacement='emitir', head_sentence_index=5),
        Token(18, 26, Pos('PRON', feats={'Case': 'Acc,Dat', 'Person': '3', 'PrepCase': 'Npr', 'PronType': 'Prs', 'Reflex': 'Yes'}),
              'él', 'iobj', sub_index=1, text_replacement='se', head_sentence_index=3),
        Token(27, 37, Pos('NOUN', feats={'Gender': 'Fem', 'Number': 'Sing'}), 'publicidad', 'root', head_sentence_index=-1),
        Token(38, 47, Pos('ADJ', feats={'Number': 'Sing'}), 'comercial', 'amod', head_sentence_index=5),
        Token(48, 49, Pos('CCONJ'), 'y', 'cc', head_sentence_index=8),
        Token(50, 59, Pos('NOUN', feats={'Gender': 'Masc', 'Number': 'Plur'}), 'programa', 'conj', head_sentence_index=5),
        Token(60, 75, Pos('ADJ', feats={'Number': 'Plur'}), 'estadounidense', 'amod', head_sentence_index=8),
        Token(75, 76, Pos('PUNCT'), '.', 'punct', head_sentence_index=5)])
}

multiword_token_document2 = {
    "CoNLL-U": '''# sent_id = n01016019
# text = Varios analistas han sugerido que Huawei está en la mejor posición para beneficiarse del retroceso de Samsung.
# text_en = Several analysts have suggested Huawei is best placed to benefit from Samsung's setback.
1	Varios	_	DET	DT	Gender=Masc|Number=Plur	2	det	_	_
2	analistas	_	NOUN	NN	Gender=Masc|Number=Plur	4	nsubj	_	_
3	han	_	VERB	VBC	Aspect=Perf|Mood=Ind|Number=Plur|Person=3|Tense=Past|Voice=Act	4	aux	_	_
4	sugerido	_	VERB	VBN	_	0	root	_	_
5	que	_	ADP	IN	_	7	mark	_	_
6	Huawei	_	PROPN	NNP	Number=Sing	7	nsubj	_	_
7	está	_	VERB	VBC	Aspect=Imp|Mood=Ind|Number=Sing|Person=3|Tense=Pres|Voice=Act	4	ccomp	_	_
8	en	_	ADP	IN	_	11	case	_	_
9	la	_	DET	DT	Gender=Fem|Number=Sing	11	det	_	_
10	mejor	_	ADJ	JJR	Gender=Fem|Number=Sing	11	amod	_	_
11	posición	_	NOUN	NN	Gender=Fem|Number=Sing	7	obl	_	_
12	para	_	ADP	IN	_	13	case	_	_
13-14	beneficiarse	_	_	_	_	_	_	_	_
13	beneficiar	_	VERB	VB	Aspect=Imp|Voice=Act	7	xcomp	_	_
14	se	_	PRON	SE	Person=3	13	compound:prt	_	_
15-16	del	_	_	_	_	_	_	_	_
15	de	de	ADP	INDT	_	17	case	_	_
16	el	el	DET	_	Gender=Masc|Number=Sing	17	det	_	_
17	retroceso	_	NOUN	NN	Gender=Masc|Number=Sing	13	obl	_	_
18	de	_	ADP	IN	_	19	case	_	_
19	Samsung	_	PROPN	NNP	Number=Sing	17	nmod	_	SpaceAfter=No
20	.	_	PUNCT	.	_	4	punct	_	_

''',
    "text": 'Varios analistas han sugerido que Huawei está en la mejor posición para beneficiarse del retroceso de Samsung. ',
    "document": Document('Varios analistas han sugerido que Huawei está en la mejor posición para beneficiarse del retroceso de Samsung. ', [
        Token(0, 6, Pos('DET', 'DT', {'Gender': 'Masc', 'Number': 'Plur'}), deprel='det', head_sentence_index=1),
        Token(7, 16, Pos('NOUN', 'NN', {'Gender': 'Masc', 'Number': 'Plur'}), deprel='nsubj', head_sentence_index=3),
        Token(17, 20, Pos('VERB', 'VBC', {'Aspect': 'Perf', 'Mood': 'Ind', 'Number': 'Plur', 'Person': '3', 'Tense': 'Past', 'Voice': 'Act'}), deprel='aux', head_sentence_index=3),
        Token(21, 29, Pos('VERB', 'VBN'), deprel='root', head_sentence_index=-1),
        Token(30, 33, Pos('ADP', 'IN'), deprel='mark', head_sentence_index=6),
        Token(34, 40, Pos('PROPN', 'NNP', {'Number': 'Sing'}), deprel='nsubj', head_sentence_index=6),
        Token(41, 45, Pos('VERB', 'VBC', {'Aspect': 'Imp', 'Mood': 'Ind', 'Number': 'Sing', 'Person': '3', 'Tense': 'Pres', 'Voice': 'Act'}), deprel='ccomp', head_sentence_index=3),
        Token(46, 48, Pos('ADP', 'IN'), deprel='case', head_sentence_index=10),
        Token(49, 51, Pos('DET', 'DT', {'Gender': 'Fem', 'Number': 'Sing'}), deprel='det', head_sentence_index=10),
        Token(52, 57, Pos('ADJ', 'JJR', {'Gender': 'Fem', 'Number': 'Sing'}), deprel='amod', head_sentence_index=10),
        Token(58, 66, Pos('NOUN', 'NN', {'Gender': 'Fem', 'Number': 'Sing'}), deprel='obl', head_sentence_index=6),
        Token(67, 71, Pos('ADP', 'IN'), deprel='case', head_sentence_index=12),
        Token(72, 84, Pos('VERB', 'VB', {'Aspect': 'Imp', 'Voice': 'Act'}), deprel='xcomp',
              sub_index=0, text_replacement='beneficiar', head_sentence_index=6),
        Token(72, 84, Pos('PRON', 'SE', {'Person': '3'}), deprel='compound:prt',
              sub_index=1, text_replacement='se', head_sentence_index=12),
        Token(85, 88, Pos('ADP', 'INDT'), lemma='de', deprel='case',
              sub_index=0, text_replacement='de', head_sentence_index=16),
        Token(85, 88, Pos('DET', feats={'Gender': 'Masc', 'Number': 'Sing'}), lemma='el', deprel='det',
              sub_index=1, text_replacement='el', head_sentence_index=16),
        Token(89, 98, Pos('NOUN', 'NN', {'Gender': 'Masc', 'Number': 'Sing'}), deprel='obl', head_sentence_index=12),
        Token(99, 101, Pos('ADP', 'IN'), deprel='case', head_sentence_index=18),
        Token(102, 109, Pos('PROPN', 'NNP', {'Number': 'Sing'}), deprel='nmod', head_sentence_index=16),
        Token(109, 110, Pos('PUNCT', '.'), deprel='punct', head_sentence_index=3)
    ])
}
