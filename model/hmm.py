import numpy as np
from collections import defaultdict, Counter

class HMMPOSTagger:
    def __init__(self):
        self.tags = set() #Set of tagsç
        self.num_tags= len(self.tags)
        self.transition_probs = defaultdict(lambda:defaultdict(float)) #Dictionary to save transition probabilities
        self.emmision_probs = defaultdict(lambda:defaultdict(float)) #Dictionary to save emmision probabilities
        self.tag_counts = Counter() #To count each tag
        self.word_counts = Counter() #To count each word

    def train(self, sentences, pos_tags):
        """
        Training the HMM given sentences and their pos_tags
        """

        A = np.zeros(self.num_tags, self.num_tags)

        # get the unique transition tuples (prev POS, cur POS)
        trans_keys = set(transition_counts.keys())

        for i in range(num_tags):
            for j in range(num_tags):
                # initialize the count of (prev POS, cur POS)
                count = 0

                key = (all_tags[i], all_tags[j])
                if key in transition_counts:
                    count = transition_counts[key]
                count_prev_tag = tag_counts[all_tags[i]]

                A[i, j] = (count) / (count_prev_tag)

        return A
        #for s in sentences:
        #    for 
        #return 0
    
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
