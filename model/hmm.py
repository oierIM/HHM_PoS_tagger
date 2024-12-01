import numpy as np
from collections import defaultdict, Counter

class HMMPOSTagger:
    def __init__(self, tags):
        self.tags = tags
        self.num_tags= len(self.tags)
        self.transition_counts = defaultdict(lambda:defaultdict(int))
        self.emission_counts = defaultdict(lambda:defaultdict(int))
        self.transition_probs = defaultdict(lambda:defaultdict(float)) #Dictionary to save transition probabilities
        self.emission_probs = defaultdict(lambda:defaultdict(float)) #Dictionary to save emmision probabilities
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
        for prev_tag, next_tags in self.transition_counts.items():
            total_transitions = sum(next_tags.values())
            for tag, count in next_tags.items():
                self.transition_probs[prev_tag][tag] = count / total_transitions

        #Emmision probs
        for tag, words in self.emission_counts.items():
            total_emissions = sum(words.values())
            for word, count in words.items():
                self.emission_probs[word][tag] = count / total_emissions

    
    def vilterbi_alg(self, sentence):
        """
        Function that executes the Vilterbi algorithm to find the best tags for a given sentence
        """
        best_probs = np.zeros((self.num_tags, self.num_tags))
        best_paths = np.zeros((self.num_tags, len(prep_tokens)), dtype=int)
        s_idx = states.index('--s--')

        for i in range(num_tags):
            if A[s_idx, i] == 0:
                best_probs[i, 0] = float('-inf')
            else:
                best_probs[i,0] = np.log(A[s_idx, i]) + np.log(B[i, vocab2idx[prep_tokens[0]]])


        prev_tag = '*'
        for word_idx in range(0, len(sentence)):
            cur_word = sentence[word_idx]
            for cur_tag in self.emission_probs[cur_word]:
                transition_prob = self.transition_probs[prev_tag][cur_tag]
                emission_prob = self.emission_probs[cur_word][cur_tag]
    
    def evaluate(self, sentences, pos_tags):
        """
        Evaluate HMM with anothers splits' sentences and their pos tags
        """
        pass
