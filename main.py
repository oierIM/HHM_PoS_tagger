from model.hmm import HMMPOSTagger

from collections import defaultdict, Counter
import utils.visualization_functions as visualization_functions
import utils.conllu_dataloader as conllu_dataloader
import utils.out_of_domain_evaluation as ood_dataloader
import csv
import time
import numpy as np

def csv_to_list_of_lists(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        return [list(row) for row in reader]

def csv_to_list(file_path):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        return [row for row in reader][0]

if __name__ == "__main__":

	conllu_dataloader.load_datasets(already_loaded=True)
     
	print("\n")

	# IN-DOMAIN TRAIN DATA
	train_sentences = csv_to_list_of_lists('./datasets/train_dev_sentences.csv')
	train_pos_tags = csv_to_list_of_lists('./datasets/train_dev_pos_tags.csv')
	vocabulary = csv_to_list('./datasets/train_dev_vocab.csv')
	
	# IN-DOMAIN TEST DATA
	test_sentences = csv_to_list_of_lists('./datasets/test_sentences.csv')
	test_pos_tags = csv_to_list_of_lists('./datasets/test_pos_tags.csv')
    
	# OUT-OF-DOMAIN TEST DATA
	ood_test_sentences, ood_test_tags = ood_dataloader.ptb_dataloader()
    
	tags = csv_to_list('./datasets/tagset.csv')

	tags.append('*')
	tags.append("<STOP>")
	vocabulary.append('*')
	vocabulary.append("<STOP>")
    
	print("\nTraining models...")
	start = time.time()
	hmm = HMMPOSTagger(tags, vocabulary)
	hmm_ours = HMMPOSTagger(tags, vocabulary)

	hmm.train(train_sentences, train_pos_tags, change_vocab = True)
     
	hmm_ours.train(train_sentences, train_pos_tags, change_vocab = False)
    
	end = time.time()
	print(f"Training completed in {end-start}")
    
	print("NOTE: All models are allways trained with Universal Dependencies PoS tags.")
	print("Re-mapping is only applied during evaluation.")
	
	
	print("\n\n\033[1mIn-Domain with no re-mapping\033[0m (Original Universal Dependencies tags):")
	acc1, cm1, ut1, precision1, recall1, fscore1  = hmm_ours.evaluate(test_sentences, test_pos_tags, mapping_mode="in_domain_no_mapping")
	print(f" - Test accuracy: {acc1}")
	print(f" - Test precision : {np.mean(precision1)}")
	print(f" - Test recall: {np.mean(recall1)}")
	print(f" - Test f1-score: {np.mean(fscore1)}")
	visualization_functions.plot_confusion_matrix(cm1, ut1, title="In-Domain PoS Tagging Confusion Matrix")
	visualization_functions.plot_f1_scores(ut1, precision1, recall1, fscore1, title="In-Domain Precision, Recall and F1-Scores for Each PoS Tag")

    
	print("\n\033[1mIn-Domain with re-mapped PoS tags\033[0m (NLTK universal tags):")
	acc2, cm2, ut2, precision2, recall2, fscore2 = hmm_ours.evaluate(test_sentences, test_pos_tags, mapping_mode="in_domain_mapping")
	print(f" - Test accuracy: {acc2}")
	print(f" - Test precision : {np.mean(precision2)}")
	print(f" - Test recall: {np.mean(recall2)}")
	print(f" - Test f1-score: {np.mean(fscore2)}")
	visualization_functions.plot_confusion_matrix(cm2, ut2, title="In-Domain Re-Mapped PoS Tagging Confusion Matrix")
	visualization_functions.plot_f1_scores(ut2, precision2, recall2, fscore2, title="In-Domain Precision, Recall and F1-Scores for Each Re-Mapped PoS Tag")


	print("\n\n\033[1mOut-of-Domain with re-mapped PoS tags\033[0m (NLTK universal tags):")
	acc3, cm3, ut3, precision3, recall3, fscore3  = hmm_ours.evaluate(ood_test_sentences, ood_test_tags, mapping_mode="out_domain_mapping")
	print(f" - Test accuracy: {acc3}")
	print(f" - Test precision : {np.mean(precision3)}")
	print(f" - Test recall: {np.mean(recall3)}")
	print(f" - Test f1-score: {np.mean(fscore3)}")
	visualization_functions.plot_confusion_matrix(cm3, ut3, title="Out-of-Domain PoS Tagging Confusion Matrix")
	visualization_functions.plot_f1_scores(ut3, precision3, recall3, fscore3, title="Out-of-Domain Precision, Recall and F1-Scores for Each PoS Tag")



    