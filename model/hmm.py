import numpy as np
from collections import defaultdict, Counter

class HMMPOSTagger:
    def __init__(self, tags):
        self.tags = tags
        self.tags_idx = {i: tag for i, tag in enumerate(self.tags)}
        self.Q= len(self.tags)
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
                self.emission_probs[tag][word] = count / total_emissions
    
    def viterbi_alg(self, sentence):
        """
        Function that executes the Vilterbi algorithm to find the best tags for a given sentence
        """
        A = self.transition_probs
        B = self.emission_probs
        T = len(sentence)

        #T+1 if in the sentence doesn't appear a <STOP> in the end
        # viterbi = np.zeros((self.Q, T+1))
        # backpointer = np.zeros((self.Q, T+1))

        viterbi = np.zeros((self.Q, T))
        backpointer = np.zeros((self.Q, T))

        for tag_idx in range(self.num_tags):
            viterbi[tag_idx][0] = (A['*'][self.tags_idx[tag_idx]] * B[sentence[0]].get(self.tags_idx[tag_idx], 1e-6))
            backpointer[tag_idx][0] = 0
        
        for t in range(1, T):
            for q in self.Q:
                viterbi[q, t] = np.max(viterbi[:, t-1] * A[:][q] * B[q][t])
                backpointer[q, t] = np.argmax(viterbi[:, t-1] * A[:][q] * B[q][t])
        
        #Last iteration for <STOP>
        # for q in self.Q:
        #     viterbi[q, -1] = np.max(viterbi[:, t-1] * A[:]['<STOP>'])
        #     backpointer[q, -1] = np.argmax(viterbi[:, t-1] * A[:]['<STOP>'])
        
        best_path_pointer = ['<STOP>']

        for t in range(T, 0, -1):
            best_path_pointer.insert(0, backpointer[best_path_pointer[0]][t])
        
        return best_path_pointer

    
    def evaluate(self, sentences, pos_tags):
        """
        Evaluate HMM with anothers splits' sentences and their pos tags
        """
        pass
