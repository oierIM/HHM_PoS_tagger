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

    def create_transition_matrix(self):
        all_tags = sorted(self.tag_counts.keys())

        # initialize the transition matrix 'A'
        transition = np.zeros((self.num_tags, self.num_tags))

        # get the unique transition tuples (prev POS, cur POS)
        trans_keys = set(self.transition_counts.keys())

        for i in range(self.num_tags):
            for j in range(self.num_tags):
                # initialize the count of (prev POS, cur POS)
                count = 0

                key = (all_tags[i], all_tags[j])
                if key in self.transition_counts:
                    count = self.transition_counts[key]
                count_prev_tag = self.tag_counts[all_tags[i]]

                transition[i, j] = (count) / (count_prev_tag)

        return transition

    def create_emission_matrix(self, emission_counts, tag_counts, vocab2idx):
        num_tags = len(tag_counts)
        all_tags = sorted(tag_counts.keys())
        num_words = len(vocab2idx)

        emission = np.zeros((num_tags, num_words))
        emis_keys = set(list(emission_counts.keys()))
        for i in range(num_tags):
            for j in range(num_words):
                count = 0

                key =  (all_tags[i], vocab2idx[j])
                if key in emission_counts:
                    count = emission_counts[key]
                count_tag = tag_counts[all_tags[i]]

                emission[i, j] = (count) / (num_words)
        return emission
    
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
