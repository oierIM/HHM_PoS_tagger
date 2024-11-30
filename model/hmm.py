import numpy as np
from collections import defaultdict, Counter

class HMMPOSTagger:
    def __init__(self, tags):
        self.tags = tags
        self.num_tags= len(self.tags)
        self.transition_counts = defaultdict(lambda:defaultdict(int))
        self.emission_counts = defaultdict(lambda:defaultdict(int))
        self.transition_probs = defaultdict(lambda:defaultdict(float)) #Dictionary to save transition probabilities
        self.emmision_probs = defaultdict(lambda:defaultdict(float)) #Dictionary to save emmision probabilities
        self.word_counts = Counter() #To count each word
        self.tag_counts = Counter()

    def train(self, sentences, pos_tags, vocab):
        """
        Training the HMM given sentences and their pos_tags
        """

        self.get_counts(sentences, pos_tags, vocab)
        self.get_probs()

        
    def get_counts(self, sentences, pos_tags, vocab):

        prev_tag = '*'

        for sentence, tags in zip(sentences, pos_tags):
            for word, tag in zip(sentence, tags):

                if word not in vocab:
                    word = 'unk'
                self.transition_counts[prev_tag][tag] += 1
                self.emission_counts[tag][word] += 1
                self.tag_counts[tag] += 1
                self.word_counts[word] += 1
                prev_tag = tag
            self.transition_counts[prev_tag]["<STOP>"] +=1

    def get_probs(self):
        
        #Transition probs
        for tag in self.tags:

            next_tags = self.transition_counts[tag]

            for next_tag in next_tag.keys():

                next_tags/total_counts

    
    def vilterbi_alg(self, sentence):
        """
        Function that executes the Vilterbi algorithm to find the best tags for a given sentence
        """
        pass
    
    def evaluate(self, sentences, pos_tags):
        """
        Evaluate HMM with anothers splits' sentences and their pos tags
        """
        pass
