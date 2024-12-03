import numpy as np
from model.hmm import HMMPOSTagger
from collections import defaultdict, Counter
from conllu_dataloader import *

# def load_data():
    
#     training_corpus = "..."
#     emission_counts, transition_counts, tag_counts = create_dictionaries(training_corpus, vocab2idx)
#     states = sorted(tag_counts.keys())
#     alpha = 0.001
#     A = create_transition_matrix(transition_counts, tag_counts, alpha)
#     B = create_emission_matrix(emission_counts, tag_counts, list(vocab2idx), alpha)
#     save('A.npy', A)
#     save('B.npy', B)

if __name__ == "__main__":
    # corpus = ""
    # vocab = build_vocab(corpus)

#     tags = ['DET', 'ADJ', 'ADV', 'PREP', 'NOUN', 'VERB', '*', '<STOP>', '<UNK>']

#     sentences = [['I', 'went', 'to', 'Brazil'],
#                  ['God', 'loves', 'NLP'],
#                  ['Is', 'me', 'Mario'],
#                  ['Hello', 'Jeremy'],
#                  ['<UNK>', '<UNK>', 'me'],
#                  ['Mario', '<UNK>'],
#                  ['Mario', '<UNK>', 'loves'],
#                  ['Hello', 'Jeremy']]
    
#     sentences = [[w.lower() if w != '<UNK>' else w for w in s] for s in sentences]
    
#     pos_tags = [['NOUN', 'VERB', 'PREP', 'NOUN'],
#                  ['NOUN', 'VERB', 'NOUN'],
#                  ['VERB', 'NOUN', 'NOUN'],
#                  ['ADV', 'NOUN'],
#                  ['<UNK>', '<UNK>', 'NOUN'],
#                  ['NOUN', '<UNK>'],
#                  ['NOUN', '<UNK>', 'VERB'],
#                  ['ADV', 'NOUN']]
    
    # for i in range(len(sentences)):
    #     sentences[i].insert(0, '*')
    #     sentences[i].append('<STOP>')

    #     pos_tags[i].insert(0, '*')
    #     pos_tags[i].append('<STOP>')

#     vocab = set()
#     for s in sentences:
#         for word in s:
#             vocab.add(word.lower())
#     vocab.add('*')
#     vocab.add('<STOP>')
        directories = ["datasets/gum", "datasets/ewt"]
        df = load_sentences_from_directories(directories)
        print(df.iloc[-1])

        # Esandien batez besteko eta luzeera medianak
        avg_length, median_length = get_sentence_lengths(df)
        print(f"Average Sentence Length: {avg_length}")
        print(f"Median Sentence Length: {median_length}")

        # Datu multzoko vocab
        vocabulary = get_vocabulary(df)
        print(f"Vocabulary Size: {len(vocabulary)}")
        print(f"Vocabulary Sample: {list(vocabulary)[:10]}")

        # Universal PoS etiketak atera
        upos_tags = get_upos_tags(df)
        print(f"Unique UPoS Tags: {upos_tags}")


        hmm = HMMPOSTagger(upos_tags, vocabulary)

        hmm.train(df, upos_tags)

    #print(hmm.emission_probs)
    # print('-----------------')
    # print(hmm.transition_probs)
    # print('-----------------')
    # print(hmm.emission_counts)
    # print('-----------------')
        test1 = ['Jeremy','Loves','NLP']
        test = [['Jeremy', 'Loves', 'NLP'],
                ['Mario', 'is', 'god'],
                ['Kaixo', 'zer', 'moduz']]
        tags = [['NOUN', 'VERB', 'NOUN'],
                ['NOUN', 'VERB', 'NOUN'],
                ['<UNK>', '<UNK>', '<UNK>']]
        print(hmm.viterbi_alg(test1))
    

    


    