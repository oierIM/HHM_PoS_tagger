import numpy as np
from collections import defaultdict, Counter

class HMMPOSTagger:
    def __init__(self):
        self.tags = set() #Set of tags
        self.transition_probs = defaultdict(lambda:defaultdict(float)) #Dictionary to save transition probabilities
        self.emmision_probs = defaultdict(lambda:defaultdict(float)) #Dictionary to save emmision probabilities
        self.tag_counts = Counter() #To count each tag
        self.word_counts = Counter() #To count each word

    def train(self, sentences, pos_tags):
        """
        Training the HMM given sentences and their pos_tags
        """
        return 0
    
    def calculate_probabilities(self, transition_counts, emmision_counts):
        """
        Calculate transition and emission probabilities given their counts
        """
        return 0
    
    def vilterbi_alg(self, sentence):
        """
        Function that executes the Vilterbi algorithm to find the best tags for a given sentence
        """
        return 0
    
    def evaluate(self, sentences, pos_tags):
        """
        Evaluate HMM with anothers splits' sentences and their pos tags
        """
        return 0