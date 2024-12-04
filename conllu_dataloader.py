"""
CoNLL-U format fields (source: https://universaldependencies.org/format.html)


    ID: Word index, integer starting at 1 for each new sentence; may be a range for multiword tokens; may be a decimal number for empty nodes (decimal numbers can be lower than 1 but must be greater than 0).
    FORM: Word form or punctuation symbol.
    LEMMA: Lemma or stem of word form.
    UPOS: Universal part-of-speech tag.
    XPOS: Optional language-specific (or treebank-specific) part-of-speech / morphological tag; underscore if not available.
    FEATS: List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
    HEAD: Head of the current word, which is either a value of ID or zero (0).
    DEPREL: Universal dependency relation to the HEAD (root iff HEAD = 0) or a defined language-specific subtype of one.
    DEPS: Enhanced dependency graph in the form of a list of head-deprel pairs.
    MISC: Any other annotation.


"""


import os
import pandas as pd
from conllu import parse_incr
import csv
from tqdm import tqdm
from itertools import cycle
from shutil import get_terminal_size
from threading import Thread
from time import sleep


class Loader:
    def __init__(self, desc="Loading...", end="Done!", timeout=0.1):
        """
        A loader-like context manager

        Args:
            desc (str, optional): The loader's description. Defaults to "Loading...".
            end (str, optional): Final print. Defaults to "Done!".
            timeout (float, optional): Sleep time between prints. Defaults to 0.1.
        """
        self.desc = desc
        self.end = end
        self.timeout = timeout

        self._thread = Thread(target=self._animate, daemon=True)
        self.steps = ["⢿", "⣻", "⣽", "⣾", "⣷", "⣯", "⣟", "⡿"]
        self.done = False

    def start(self):
        self._thread.start()
        return self

    def _animate(self):
        for c in cycle(self.steps):
            if self.done:
                break
            print(f"\r{self.desc} {c}", flush=True, end="")
            sleep(self.timeout)

    def __enter__(self):
        self.start()

    def stop(self):
        self.done = True
        cols = get_terminal_size((80, 20)).columns
        print("\r" + " " * cols, end="", flush=True)
        print(f"\r{self.end}", flush=True)

    def __exit__(self, exc_type, exc_value, tb):
        # handle exceptions with those variables ^
        self.stop()

def load_conllu_files(directory, prefixes):
    """
    Kargatu .conllu fitxategi guztiak emandako direktorioan. 
    Aurrizkirik zehazten ez bada (prefixes zerrenda hutsa), direktorioko fitxategi guztiak irakurriko ditu.

    :param directory: .conllu fitxategiak dituen direktorioaren bidea.
    :param prefixes: Bat datozen fitxategi izenaren aurrizkien zerrenda. Zerrenda hutsa bada, direktorioko fitxategi guztiak irakurriko ditu.
    :return: Bat datozen .conllu fitxategi guztietatik parseatutako esaldien zerrenda.
    """
    sentences = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.conllu') and (not prefixes or any(file_name.startswith(prefix) for prefix in prefixes)):
            file_path = os.path.join(directory, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                sentences.extend(list(parse_incr(f)))

    return sentences

def sentences_to_dataframe(sentences):
    """
    CoNLL-U formatuan dauden esaldien zerrenda pandas DataFrame batera bihurtu.

    :param sentences: Parseatutako esaldien zerrenda.
    :return: Pandas DataFrame bat, token, lema, upos eta beste eremu batzuk dituena.
    """
    rows = []
    for sentence_index, sentence in enumerate(sentences):
        for token in sentence:
            if isinstance(token, dict):
                rows.append({
                    "sentence_index": sentence_index,   # Esaldiaren indizea
                    "id": token.get("id"),              # Hitzaren indizea esaldian
                    "form": token.get("form"),          # Hitzaren agerpen originala
                    "lemma": token.get("lemma"),        # Hitzaren lema
                    "upos": token.get("upos"),          # PoS etiketa unibertsala <----- 
                    "xpos": token.get("xpos"),          # Treebank PoS etiketa edo etiketa morfologikoa
                    "feats": token.get("feats"),        # Morphological features
                    "head": token.get("head"),          # Hitzaren burua
                    "deprel": token.get("deprel"),      # Buruarekiko dependentziak
                    "deps": token.get("deps"),          # Dependentzia grafoa
                    "misc": token.get("misc")           # Bestelako metadatuak
                })
    return pd.DataFrame(rows)

def get_vocabulary(df):
    """
    Sortu hiztegi bat datuetan agertzen diren hitz guztiekin.

    :param df: Pandas DataFrame, esaldien datuekin.
    :return: Hitz bakarren multzo bat (vocabulary).
    """
    return set(df['form'])

def get_sentence_lengths(df):
    """
    Kalkulatu esaldien batez besteko eta mediana luzerak.

    :param df: Pandas DataFrame, esaldien datuekin.
    :return: Batez besteko eta mediana luzerak tuple gisa.
    """
    sentence_lengths = df.groupby('sentence_index')['id'].count()
    avg_length = sentence_lengths.mean()
    median_length = sentence_lengths.median()
    return avg_length, median_length


def get_upos_tags(df):
    """
    Atera datuetan agertzen diren 'upos' etiketak.

    :param df: Pandas DataFrame, esaldien datuekin.
    :return: 'upos' etiketak multzo bat gisa.
    """
    return set(df['upos'])


def load_sentences_from_directories(directories, prefixes=[]):
    """
    Kargatu .conllu fitxategien esaldi guztiak direktorio bat edo gehiagotik eta bateratu guztiak.

    :param directories: Direktorioen zerrenda edo multzo bat.
    :param prefixes: Aurrizkien zerrenda fitxategiak iragazteko (hutsa bada, fitxategi guztiak irakurriko dira).
    :return: Bateratutako esaldien zerrenda.
    """
    #prefixes = ["GUM_academic_", "GUM_bio_", "GUM_news_"]
    #prefixes = ["GUM_academic_"]
    all_sentences = []
    for directory in directories:
        all_sentences.extend(load_conllu_files(directory, prefixes))
    return sentences_to_dataframe(all_sentences)

def store_sentences_in_csv(df, path = './datasets/'):
    sentences = []

    for i in tqdm(range(max(df["sentence_index"])), desc="Storing sentences..."):
        sentences.append(df[df['sentence_index']==i]["form"].to_list())

    with open(path + 'dataset_sentences.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(sentences)

def store_pos_tags_in_csv(df, path = './datasets/'):
    sentences = []

    for i in tqdm(range(max(df["sentence_index"])), desc="Storing pos_tags... (Kaixo Jeremy :P)"):
        sentences.append(df[df['sentence_index']==i]["form"].to_list())

    with open(path + 'dataset_sentences.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(sentences)

def store_vocab_in_csv(vocab, path = './datasets/'):
    print('Storing vocab...')
    with open(path + 'dataset_vocab.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(list(vocab))

def store_vocab_in_csv(tags, path = './datasets/'):
    print('Storing tags...')
    with open(path + 'dataset_tags.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(list(tags))
def store_csvs(df, path = './datasets/'):
    store_sentences_in_csv(df, path)
    store_pos_tags_in_csv(df, path)

    store_vocab_in_csv(get_vocabulary(df), path)
    store_vocab_in_csv(get_upos_tags(df), path)

# Probak egiteko deskomentatu, bestela "import" bidez erabili
if __name__ == "__main__":

    directories = ["datasets/gum", "datasets/ewt"]


    loader = Loader("Loading dataset...").start()

    df = load_sentences_from_directories(directories)
    
    loader.stop()
        
    print('Loaded! \N{grinning face with smiling eyes}')

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

    path = './datasets/'
    store_csvs(df, path)
    print(f'Dataset stored in {path}')