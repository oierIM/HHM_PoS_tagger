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

def build_vocab(corpus):

    tokens = corpus #betetzeko
    freqs = defaultdict(int)
    for tok in tokens:
        freqs[tok] += 1

    vocab = [k for k, v in freqs.items() if (v > 1 and k != '\n')]
    unk_toks = ["--unk--", "--unk_adj--", "--unk_adv--", "--unk_digit--", "--unk_noun--", "--unk_punct--", "--unk_upper--", "--unk_verb--"]
    vocab.extend(unk_toks)
    vocab.append("*")
    vocab.append(" ")
    return vocab

if __name__ == "__main__":
    corpus = ""
    vocab = build_vocab(corpus)