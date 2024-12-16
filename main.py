import numpy as np
from model.hmm import HMMPOSTagger
from collections import defaultdict, Counter
import conllu_dataloader
import csv
import time

def csv_to_list_of_lists(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        return [list(row) for row in reader]

def csv_to_list(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        return [row for row in reader][0]

if __name__ == "__main__":

	conllu_dataloader.load_datasets()
     
	print("\n\n")

	train_sentences = csv_to_list_of_lists('./datasets/train_dev_sentences.csv')
	train_pos_tags = csv_to_list_of_lists('./datasets/train_dev_pos_tags.csv')
	vocabulary = csv_to_list('./datasets/train_dev_vocab.csv')
	
	test_sentences = csv_to_list_of_lists('./datasets/test_sentences.csv')
	test_pos_tags = csv_to_list_of_lists('./datasets/test_pos_tags.csv')

	tags = csv_to_list('./datasets/tagset.csv')

	tags.append('*')
	tags.append("<STOP>")
	vocabulary.append('*')
	vocabulary.append("<STOP>")
    
	print("\n\nTraining models...")
	start = time.time()
	hmm = HMMPOSTagger(tags, vocabulary)
	hmm_ours = HMMPOSTagger(tags, vocabulary)

	hmm.train(train_sentences, train_pos_tags, change_vocab = True)
     
	hmm_ours.train(train_sentences, train_pos_tags, change_vocab = False)
    
	end = time.time()
	print(f"Training completed in {end-start}")
    

	
	test1 = ['her','dunking','was','suprememeably','supreme']

	test = [['Jeremy', 'Loves', 'NLP'],
			['Mario', 'is', 'god'],
			['Kaixo', 'zer', 'moduz']]
    
	tags = [['NOUN', 'VERB', 'NOUN'],
			['NOUN', 'VERB', 'NOUN'],
			['<UNK>', '<UNK>', '<UNK>']]

	viterbi_result = hmm_ours.viterbi_alg(test1)
	print(f'Original sentence= {test1}')
	print(f'Sentence = {viterbi_result[0]}')
	print(f'Tags applied = {viterbi_result[1]}')
     
	viterbi_result = hmm.viterbi_alg(test1)
	print(f'Sentence = {viterbi_result[0]}')
	print(f'Tags applied = {viterbi_result[1]}')
    
	
	print("OTHER VITERBI APPROACH:")
	print({f"Evaluation accuracy: {hmm.evaluate(test_sentences, test_pos_tags)}"})
    
	print("OUR VITERBI APPROACH:")
	print(f"Evaluation accuracy: {hmm_ours.evaluate(test_sentences, test_pos_tags)}")





    