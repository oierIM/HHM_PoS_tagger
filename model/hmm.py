import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from collections import defaultdict, Counter

class HMMPOSTagger:
    """
    Hitz-Marko Ezkutuen (HMM) oinarritutako Hitzen Etiketatzailea.
    Klase honek HMM bat erabiltzen du trantsizio eta emisio probabilitateak kalkulatzeko, 
    trebakuntza datuetatik abiatuta. Viterbi algoritmoa erabiltzen du etiketa sekuentzia 
    probableena aurkitzeko.
    """
    def __init__(self, tags, vocab):
        """
        HMMPOSTagger objektua hasieratzen du emandako etiketekin eta hiztegiarekin.
        
        Arg:
            tags (zerrenda): Posibleak diren hitz-etiketak.
            vocab (zerrenda): Ezagutzen diren hitzen hiztegia.
        """
        self.tags2idx = {tag: i for i, tag in enumerate(tags)}
        self.idx2tags = {i: tag for i, tag in enumerate(tags)}
        self.tags = list(self.idx2tags.keys())
        self.Q= len(self.tags)
        self.vocab = vocab
        self.transition_counts = defaultdict(lambda:defaultdict(int))
        self.emission_counts = defaultdict(lambda:defaultdict(int))
        self.transition_probs = defaultdict(lambda:defaultdict(float)) #Dictionary to save transition probabilities
        self.emission_probs = defaultdict(lambda:defaultdict(float)) #Dictionary to save emmision probabilities
        self.word_counts = Counter() #To count each word
        self.tag_counts = Counter()

    def train(self, sentences, pos_tags, change_vocab=False):
        """
        HMM entrenatzen du esaldi eta dagokien POS etiketak erabiliz.

        Arg:
            sentences (zerrenda-zerrendak): Esaldi bakoitzaren hitzak dituen azpizerrendak.
            pos_tags (zerrenda-zerrendak): Esaldi bakoitzari dagozkion POS etiketak.
            change_vocab (bool): True baldin badago, hiztegia eguneratuko du hitzen maiztasunaren arabera.
        """

        self.get_counts(sentences, pos_tags, change_vocab)
        self.get_probs()

        
    def get_counts(self, sentences, pos_tags, change_vocab = False):
        """
        Entrenamendu datuetatik trantsizio, emisio, hitz eta etiketen zenbaketak kalkulatzen ditu.
        
        Arg:
            sentences (zerrenda-zerrendak): Entrenamendu datuen esaldiak.
            pos_tags (zerrenda-zerrendak): Esaldi bakoitzari dagozkion POS etiketak.
            change_vocab (bool): True baldin badago, hiztegia eguneratuko du ohikoak ez diren hitzak baztertuz.
        """
        for sentence, tags in zip(sentences, pos_tags):
            prev_tag = self.tags2idx['*']
            for word, tag in zip(sentence, tags):
                
                tag_idx = self.tags2idx[tag]
                
                self.transition_counts[prev_tag][tag_idx] += 1
                self.emission_counts[tag_idx][word] += 1
                self.tag_counts[tag_idx] += 1
                self.word_counts[word] += 1
                prev_tag = tag_idx
            self.transition_counts[prev_tag]["<STOP>"] +=1

        if change_vocab:
            vocab = []
            unknowns = []
            for w in self.word_counts.keys(): 
                if self.word_counts[w] > 4:
                    vocab.append(w)
                else:
                    unknowns.append(w)
            self.vocab = vocab

            for u in unknowns:
                for tag in self.tags2idx.values():
                    # print(self.idx2tags[tag])
                    # if self.emission_counts[tag][u] > 0:
                    if tag==18 or tag==19:
                        break
                    self.emission_counts[tag]['<UNK>'] += self.emission_counts[tag][u]
                    self.emission_counts[tag].pop(u)

    def get_probs(self):
        """
        Zenbaketak probabilitate bihurtzen ditu trantsizio eta emisioetarako.
        """
        
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
    
    def viterbi_alg(self, sentence: list[int]):
        """
        Viterbi algoritmoa inplementatzen du esaldi baterako etiketarik probableenak aurkitzeko.
        
        Arg:
            sentence (zerrenda): Esaldiaren hitzen (tokenen) zerrenda.
        
        Itzultzen du:
            tuple: (hitzen zerrenda, aurresandako POS etiketen sekuentzia)
        """
        A = self.transition_probs
        B = self.emission_probs
        T = len(sentence)
        

        sentence = [w.lower() if w.lower() in self.vocab else '<UNK>' for w in sentence]
        
        viterbi = np.zeros((self.Q, T))
        backpointer = np.zeros((self.Q, T), dtype=int)
 
        for tag_idx in range(self.Q):
            viterbi[tag_idx][0] = (A[self.tags2idx['*']][tag_idx] * B[tag_idx].get(sentence[0], 1e-6))

        for t in range(1, T):
            for q in range(self.Q):
                
                viterbi[q, t] = np.max([viterbi[q_p, t-1] * A[q_p][q] * B[q].get(sentence[t], 1e-6) for q_p in range(self.Q)])
                backpointer[q, t] = np.argmax([viterbi[q_p, t-1] * A[q_p][q] * B[q].get(sentence[t], 1e-6) for q_p in range(self.Q)])
        
        best_path_pointer = [np.argmax(viterbi[:, T-1])]

        for t in range(T-1, 0, -1):
            best_path_pointer.insert(0, backpointer[best_path_pointer[0]][t])
        
        return [self.idx2tags[idx] for idx in best_path_pointer]

    
    def evaluate(self, sentences, pos_tags, mapping_mode="in_domain_no_mapping"):
        """
        HMM etiketatzailea proba-datuetan ebaluatzen du.

        Arg:
            sentences (zerrenda-zerrenda): Proba esaldiak.
            pos_tags (zerrenda-zerrenda): Esaldi bakoitzari dagozkion egiazko POS etiketak.
            mapping_mode (str): 'in_domain_no_mapping', 'in_domain_mapping', 'out_domain_mapping' aukeretako bat.

        Itzultzen du:
            float: Etiketatzailearen zehaztasuna proba-datuetan.
            list: Nahasketa-matrizea.
            list: Erabilitako etiketa bereziak.
            list: Zehaztasun-puntuazioak.
            list: Errebokazio-puntuazioak.
            list: F1 puntuazioak.
        """
        print("Evaluating...")

        map_pos_tags = {
            'ADJ': 'ADJ',
            'PROPN': 'NOUN',
            'NOUN': 'NOUN',
            'PRON': 'PRON',
            'CCONJ': 'CONJ',
            'ADV': 'ADV',
            'X': 'X',
            'VERB': 'VERB',
            '_': '_',
            'AUX': 'VERB',
            'SYM': '.',
            'NUM': 'NUM',
            'PART': 'PRT',
            'DET': 'DET',
            'INTJ': 'INTJ',
            'ADP': 'ADP',
            'SCONJ': 'CONJ',
            'PUNCT': '.',
        }

        correct, total = 0, 0

        all_true_tags = []
        all_pred_tags = []

        for sentence, true_tags in zip(sentences, pos_tags):
            pred_tags = self.viterbi_alg(sentence)

            if mapping_mode == "in_domain_mapping":
                pred_tags = [map_pos_tags[tag] for tag in pred_tags]
                true_tags = [map_pos_tags[tag] for tag in true_tags]
            elif mapping_mode == "out_domain_mapping":
                pred_tags = [map_pos_tags[tag] for tag in pred_tags]

            all_true_tags.extend(true_tags)
            all_pred_tags.extend(pred_tags)

            for p, t in zip(pred_tags, true_tags):
                if p == t:
                    correct += 1
            total += len(true_tags)

        unique_tags = sorted(set(all_true_tags + all_pred_tags))
        cm = confusion_matrix(all_true_tags, all_pred_tags, labels=unique_tags)

        precision, recall, f1, _ = precision_recall_fscore_support(all_true_tags, all_pred_tags)

        return correct / total, cm, unique_tags, precision, recall, f1