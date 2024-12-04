import numpy as np
from model.hmm import HMMPOSTagger
from collections import defaultdict, Counter
from conllu_dataloader import *

# def load_data():
    
#     training_corpus = "..."
#     emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus, vocab2idx)
#     states = sorted(tag_counts.keys())
#     alpha = 0.001
#     A = create_transition_matrix(transition_counts, tag_counts, alpha)
#     B = create_emission_matrix(emission_counts, tag_counts, list(vocab2idx), alpha)
#     save('A.npy', A)
#     save('B.npy', B)

def csv_to_list_of_lists(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        return [list(row) for row in reader]

if __name__ == "__main__":

        directories = ["datasets/gum", "datasets/ewt"]
        df = load_sentences_from_directories(directories)

        vocabulary = get_vocabulary(df)

        vocabulary.add('*')
        vocabulary.add("<STOP>")
        hmm = HMMPOSTagger(csv_to_list_of_lists('./datasets/dataset_pos_tags.csv'), vocabulary)
        print(csv_to_list_of_lists('./datasets/dataset_pos_tags.csv'))
        hmm.train(csv_to_list_of_lists('./datasets/dataset_sentences.csv'), csv_to_list_of_lists('./datasets/dataset_pos_tags.csv'))

        test1 = ['Jeremy','Loves','NLP']

        test = [['Jeremy', 'Loves', 'NLP'],
                ['Mario', 'is', 'god'],
                ['Kaixo', 'zer', 'moduz']]
        tags = [['NOUN', 'VERB', 'NOUN'],
                ['NOUN', 'VERB', 'NOUN'],
                ['<UNK>', '<UNK>', '<UNK>']]
        print(hmm.viterbi_alg(test1))
    

    


    