from model.hmm import HMMPOSTagger

from collections import defaultdict, Counter
import utils.visualization_functions as visualization_functions
import utils.conllu_dataloader as conllu_dataloader
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

	conllu_dataloader.load_datasets(already_loaded=False)
     
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
    
	print("Training models...")
	start = time.time()
	hmm = HMMPOSTagger(tags, vocabulary)
	hmm_ours = HMMPOSTagger(tags, vocabulary)

	hmm.train(train_sentences, train_pos_tags, change_vocab = True)
     
	hmm_ours.train(train_sentences, train_pos_tags, change_vocab = False)
    
	end = time.time()
	print(f"Training completed in {end-start}")
    
	
	print("\n\nOTHER VITERBI APPROACH:")
	acc1, cm1, ut1, precision1, recall1, fscore1  = hmm.evaluate(test_sentences, test_pos_tags)
	print(f"Test accuracy: {acc1}")
	print(f"Test precision : {precision1}")
	print(f"Test recall: {recall1}")
	print(f"Test f1-score: {fscore1}")
    

	map_pos_tags = {'ADJ': 'ADJ',
                'PROPN':'NOUN', 
                'NOUN': 'NOUN',
                'PRON':'PRON',
                'CCONJ':'CONJ',
                'ADV':'ADV',
                'X':'X',
                'VERB':'VERB',
                '_':'_',
                'AUX':'VERB',
                'SYM':'.',
                'NUM':'NUM',
                'PART':'PRT',
                'DET':'DET',
                'INTJ':'INTJ',
                'ADP':'ADP',
                'SCONJ':'CONJ',
                'PUNCT':'.',
                }
    
	print("\nOUR VITERBI APPROACH:")
	acc2, cm2, ut2, precision2, recall2, fscore2 = hmm_ours.evaluate(test_sentences, test_pos_tags)
	print(f"Test accuracy: {acc2}")
	print(f"Test precision : {precision2}")
	print(f"Test recall: {recall2}")
	print(f"Test f1-score: {fscore2}")
     
	visualization_functions.plot_confusion_matrix(cm1, ut1)
	visualization_functions.plot_confusion_matrix(cm2, ut2)






    