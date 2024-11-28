import numpy as np
from model.hmm import HMMPOSTagger
corpus_path = "WSJ_02-21.pos"

def load_data():
    
    training_corpus = "..."
    emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus, vocab2idx)
    states = sorted(tag_counts.keys())
    alpha = 0.001
    A = create_transition_matrix(transition_counts, tag_counts, alpha)
    B = create_emission_matrix(emission_counts, tag_counts, list(vocab2idx), alpha)
    save('A.npy', A)
    save('B.npy', B)

if __name__ == "__main__":
    load_data()