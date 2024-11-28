import numpy as np
from collections import defaultdict, Counter

class HMMPOSTagger:
    def __init__(self):
        self.tags = set() #Set of tags
        self.num_tags= len(self.tags)
        self.transition_probs = defaultdict(lambda:defaultdict(float)) #Dictionary to save transition probabilities
        self.emmision_probs = defaultdict(lambda:defaultdict(float)) #Dictionary to save emmision probabilities
        self.tag_counts = Counter() #To count each tag
        self.word_counts = Counter() #To count each word

    def train(self, sentences, pos_tags):
        """
        Training the HMM given sentences and their pos_tags
        """

        transition_counts = defaultdict(int)
        tag_counts = defaultdict(int)

        prev_tag = '--s--'

        for tok_tag in training_corpus:

            tok, tag = get_word_tag(tok_tag, vocab2idx)
            transition_counts[(prev_tag, tag)] += 1
            emission_counts[(tag, tok)] += 1
            tag_counts[tag] += 1
            prev_tag = tag

        #transition matrix

        # get the unique transition tuples (prev POS, cur POS)
        trans_keys = set(transition_counts.keys())


        all_tags = sorted(tag_counts.keys())
        num_tags = len(all_tags)

        for i in range(num_tags):
            for j in range(num_tags):
                # initialize the count of (prev POS, cur POS)
                count = 0

                key = (all_tags[i], all_tags[j])
                if key in transition_counts:
                    count = transition_counts[key]
                count_prev_tag = tag_counts[all_tags[i]]

                self.transition_probs[i, j] = (count) / (count_prev_tag)

        #emission matrix

        num_tags = len(tag_counts)
        all_tags = sorted(tag_counts.keys())
        num_words = len(vocab2idx)


        B = np.zeros((num_tags, num_words))
        emis_keys = set(list(emission_counts.keys()))
        for i in range(num_tags):
            for j in range(num_words):
                count = 0

                key =  (all_tags[i], vocab2idx[j])
                if key in emission_counts:
                    count = emission_counts[key]
                count_tag = tag_counts[all_tags[i]]

                self.emmision_probs[i, j] = (count) / (count_tag)


        
    
    def calculate_probabilities(self, transition_counts, emmision_counts):
        """
        Calculate transition and emission probabilities given their counts
        """
        pass
    
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
