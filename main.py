import numpy as np
from model.hmm import HMMPOSTagger
from collections import defaultdict, Counter
from conllu_dataloader import *
#Kaixo
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

def csv_to_list(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        return [row for row in reader][0]

if __name__ == "__main__":

	sentences = csv_to_list_of_lists('./datasets/dataset_sentences.csv')
	pos_tags = csv_to_list_of_lists('./datasets/dataset_pos_tags.csv')
	vocabulary = csv_to_list('./datasets/dataset_vocab.csv')
	tags = csv_to_list('./datasets/dataset_tags.csv')

	tags.append('*')
	tags.append("<STOP>")
	vocabulary.append('*')
	vocabulary.append("<STOP>")

	hmm = HMMPOSTagger(tags, vocabulary)

	hmm.train(sentences, pos_tags)

	test1 = ['Jeremy','Loves','NLP']

	test = [['Jeremy', 'Loves', 'NLP'],
			['Mario', 'is', 'god'],
			['Kaixo', 'zer', 'moduz']]
	tags = [['NOUN', 'VERB', 'NOUN'],
			['NOUN', 'VERB', 'NOUN'],
			['<UNK>', '<UNK>', '<UNK>']]

	print(hmm.viterbi_alg(test1))





    