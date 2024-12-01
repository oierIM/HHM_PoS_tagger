import numpy as np
from model.hmm import HMMPOSTagger
from collections import defaultdict, Counter

# def load_data():
    
#     training_corpus = "..."
#     emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus, vocab2idx)
#     states = sorted(tag_counts.keys())
#     alpha = 0.001
#     A = create_transition_matrix(transition_counts, tag_counts, alpha)
#     B = create_emission_matrix(emission_counts, tag_counts, list(vocab2idx), alpha)
#     save('A.npy', A)
#     save('B.npy', B)

def build_vocab(sentences):

    set(sentences)

    vocab = [k for k, v in freqs.items() if (v > 1 and k != '\n')]
    unk_toks = ["--unk--", "--unk_adj--", "--unk_adv--", "--unk_digit--", "--unk_noun--", "--unk_punct--", "--unk_upper--", "--unk_verb--"]
    vocab.extend(unk_toks)
    vocab.append("*")
    vocab.append(" ")
    return vocab

if __name__ == "__main__":
    # corpus = ""
    # vocab = build_vocab(corpus)

    tags = ['DET', 'ADJ', 'ADV', 'PREP', 'NOUN', 'VERB', '*', '<STOP>', '<UNK>']

    sentences = [['I', 'went', 'to', 'Brazil'],
                 ['God', 'loves', 'NLP'],
                 ['Is', 'me', 'Mario'],
                 ['Hello', 'Jeremy'],
                 ['Hello', 'Jeremy']]
    
    sentences = [[w.lower() for w in s] for s in sentences]
    
    pos_tags = [['NOUN', 'VERB', 'PREP', 'NOUN'],
                 ['NOUN', 'VERB', 'NOUN'],
                 ['VERB', 'NOUN', 'NOUN'],
                 ['ADV', 'NOUN'],
                 ['ADV', 'NOUN']]
    
    # for i in range(len(sentences)):
    #     sentences[i].insert(0, '*')
    #     sentences[i].append('<STOP>')

    #     pos_tags[i].insert(0, '*')
    #     pos_tags[i].append('<STOP>')

    vocab = set()
    for s in sentences:
        for word in s:
            vocab.add(word.lower())
    vocab.add('*')
    vocab.add('<STOP>')

    hmm = HMMPOSTagger(tags)

    hmm.train(sentences, pos_tags, vocab)

    print(hmm.emission_probs)
    # print('-----------------')
    # print(hmm.transition_probs)
    # print('-----------------')
    # print(hmm.emission_counts)
    # print('-----------------')

    # print(hmm.viterbi_alg(['Jeremy','loves','NLP']))
    

    


    