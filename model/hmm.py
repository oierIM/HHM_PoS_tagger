import numpy as np
from collections import defaultdict, Counter

class HMMPOSTagger:
    def __init__(self, tags):
        self.tags2idx = {tag: i for i, tag in enumerate(tags)}
        self.idx2tags = {i: tag for i, tag in enumerate(tags)}
        self.tags = list(self.idx2tags.keys())
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


        for sentence, tags in zip(sentences, pos_tags):
            prev_tag = self.tags2idx['*']
            for word, tag in zip(sentence, tags):
                
                tag_idx = self.tags2idx[tag]
                # if word not in vocab:
                #     word = 'unk'
                self.transition_counts[prev_tag][tag_idx] += 1
                self.emission_counts[tag_idx][word] += 1
                self.tag_counts[tag_idx] += 1
                self.word_counts[word] += 1
                prev_tag = tag_idx
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

        sentence = [w.lower() for w in sentence] 

        #T+1 if in the sentence doesn't appear a <STOP> in the end
        # viterbi = np.zeros((self.Q, T+1))
        # backpointer = np.zeros((self.Q, T+1))
        
        viterbi = np.zeros((self.Q, T))
        backpointer = np.zeros((self.Q, T), dtype=int)
 
        for tag_idx in range(self.Q):
            viterbi[tag_idx][0] = (A[self.tags2idx['*']][tag_idx] * B[tag_idx].get(sentence[0], 1e-6))

        for t in range(1, T):
            for q in range(self.Q):
                # print(str(self.idx2tags[q]) + " -----------------------------")

                # print([viterbi[q_p, t-1] for q_p in range(self.Q)])
                # print([A[q_p][q] for q_p in range(self.Q)])
                # print([B[q][t] for q_p in range(self.Q)])

                viterbi[q, t] = np.max([viterbi[q_p, t-1] * A[q_p][q] * B[q].get(sentence[t], 1e-6) for q_p in range(self.Q)])
                backpointer[q, t] = np.argmax([viterbi[q_p, t-1] * A[q_p][q] * B[q].get(sentence[t], 1e-6) for q_p in range(self.Q)])
        
        #Last iteration for <STOP>
        # for q in self.Q:
        #     viterbi[q, -1] = np.max(viterbi[:, t-1] * A[:]['<STOP>'])
        #     backpointer[q, -1] = np.argmax(viterbi[:, t-1] * A[:]['<STOP>'])
        
        best_path_pointer = [np.argmax(viterbi[:, T-1])]
        #print(viterbi)

        for t in range(T-1, 0, -1):
            #print(best_path_pointer[0])
            best_path_pointer.insert(0, backpointer[best_path_pointer[0]][t])
        
        return [self.idx2tags[idx] for idx in best_path_pointer]

    
    def evaluate(self, sentences, pos_tags):
        """
        Evaluate HMM with anothers splits' sentences and their pos tags
        """
        correct, total = 0, 0
        for sentence, true_tags in zip(sentences, pos_tags):
            pred_tags = self.viterbi_alg(sentence)
            for p, t in zip(pred_tags, true_tags):
                if p==t:
                    correct +=1
            total += len(true_tags)

        return correct / total
