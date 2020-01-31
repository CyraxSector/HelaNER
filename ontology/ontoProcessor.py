from nltk.chunk import RegexpParser
from nltk.corpus.reader import TaggedCorpusReader
from nltk.tag import hmm


def tagging(sent):
    reader = TaggedCorpusReader('E:\\Projects\\HelaNER\\data\\pos', r'.*\.pos')
    train_data = reader.tagged_sents()[:3000]
    trainer = hmm.HiddenMarkovModelTrainer()
    tagger = trainer.train_supervised(train_data)
    text = tagger.tag(sent.split())
    return text


def extract_sov(data):
    chunker = RegexpParser(r'''
        NP:
        {<JJ>*<NN.*><NNPA.*>*<NNPI.*>*<PRP.*>*}
        VP:
        {<JVB>*<NVB>*<V.*>*}
    ''')
    parsed_tree = chunker.parse(data)
    count = 0
    so_list = []
    sub = ''
    ob = ''
    verb = ''
    triple = []

    for subtree in parsed_tree.subtrees(filter=lambda t: t.label() == 'NP'):
        count = count + 1
    if count < 2:
        print('sentence is not correct \n')
    else:
        for subtree in parsed_tree.subtrees(filter=lambda t: t.label() == 'NP'):
            first_so = []
            result = subtree.leaves()
            print(result)
            for word, tag in result:
                first_so.append(word)
                so = ' '.join(first_so)
            so_list.append(so)
            print('SO List:' + str(so_list))
        if len(so_list) != 2:
            print('System has not taken the Subject-Object correctly')
        else:
            sub = so_list[0]
            ob = so_list[1]

    for subtree in parsed_tree.subtrees(filter=lambda t: t.label() == 'VP'):
        correct_verb = []
        result = subtree.leaves()
        for word, tag in result:
            correct_verb.append(word)
            ve = ' '.join(correct_verb)
        print('verb: ' + ve + '\n')
        verb = ve

    if sub and ob:
        triple = [sub, ob, verb]

    return triple
