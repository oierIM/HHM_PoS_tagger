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
    def __init__(self, desc="Datuak kargatzen...", end='Datuak kargatuta! \N{grinning face with smiling eyes}', timeout=0.1):
        """
        Karga-indikadore bat sortzen du.

        Args:
            desc (str): Deskribapena.
            end (str): Amaierako mezua.
            timeout (float): Tartea (segundutan). Defaults to 0.1.
        """
        self.desc = desc
        self.end = end
        self.timeout = timeout

        self._thread = Thread(target=self._animate, daemon=True)
        self.steps = ['|', '/', '-', '\\']
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
        self.stop()

def load_conllu_files(directory, suffix):
    """
    Kargatu .conllu fitxategiak emandako direktorioan, sufijo jakin bat duenak.

    :param directory: .conllu fitxategiak dituen direktorioaren bidea.
    :param suffix: Fitxategiaren sufijoa (adibidez, '-train', '-dev', '-test').
    :return: Bat datozen .conllu fitxategietatik parseatutako esaldien zerrenda.
    """
    sentences = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.conllu') and file_name.endswith(suffix + '.conllu'):
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
                    "upos": token.get("upos"),          # PoS etiketa unibertsala
                    "xpos": token.get("xpos"),          # Treebank PoS etiketa edo etiketa morfologikoa
                    "feats": token.get("feats"),        # Ezaugarri morfologikoak
                    "head": token.get("head"),          # Hitzaren burua
                    "deprel": token.get("deprel"),      # Buruarekiko dependentziak
                    "deps": token.get("deps"),          # Dependentzia grafoa
                    "misc": token.get("misc")           # Bestelako metadatuak
                })
    return pd.DataFrame(rows)

def load_train_and_dev_data(directories):
    """Train eta dev datuak kargatu zehaztutako direktorioetatik."""
    all_sentences = []
    for directory in directories:
        all_sentences.extend(load_conllu_files(directory, '-train'))
        all_sentences.extend(load_conllu_files(directory, '-dev'))
    return sentences_to_dataframe(all_sentences)

def load_test_data(directories):
    """Test-datuak kargatu zehaztutako direktorioetatik."""
    all_sentences = []
    for directory in directories:
        all_sentences.extend(load_conllu_files(directory, '-test'))
    return sentences_to_dataframe(all_sentences)

def get_vocabulary(df):
    """Datu multzoan agertzen diren hitz guztiekin hiztegia sortu."""
    return set(df['form'])

def get_sentence_lengths(df):
    """Esaldien batez besteko eta mediana luzerak kalkulatu."""
    sentence_lengths = df.groupby('sentence_index')['id'].count()
    avg_length = sentence_lengths.mean()
    median_length = sentence_lengths.median()
    return avg_length, median_length

def get_upos_tags(df):
    """Datuetan agertzen diren 'upos' etiketak atera."""
    return set(df['upos'])

def store_sentences_in_csv(df, path, filename):
    sentences = []
    for i in tqdm(range(max(df["sentence_index"])), desc=f"{filename} gordetzen"):
        sentences.append(df[df['sentence_index'] == i]["form"].to_list())
    with open(os.path.join(path, filename), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(sentences)

def store_pos_tags_in_csv(df, path, filename):
    sentences = []
    for i in tqdm(range(max(df["sentence_index"])), desc=f"{filename} gordetzen"):
        sentences.append(df[df['sentence_index'] == i]["upos"].to_list())
    with open(os.path.join(path, filename), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(sentences)

def store_vocab_in_csv(vocab, path, filename):
    print(f'Hiztegia gordetzen: {filename}')
    with open(os.path.join(path, filename), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(list(vocab))

def store_csvs(df, path, prefix):
    store_sentences_in_csv(df, path, f'{prefix}_sentences.csv')
    store_pos_tags_in_csv(df, path, f'{prefix}_pos_tags.csv')
    store_vocab_in_csv(get_vocabulary(df), path, f'{prefix}_vocab.csv')
    store_vocab_in_csv(get_upos_tags(df), path, f'tagset.csv')


def load_datasets(already_loaded=False):
    directories = ["datasets/gum", "datasets/ewt"]
    if(not already_loaded):
        loader = Loader("Datu multzoak kargatzen...").start()

        train_dev_df = load_train_and_dev_data(directories)
        test_df = load_test_data(directories)

        loader.stop()

        # Entrenamendu eta garapen multzoa aztertu
        avg_length, median_length = get_sentence_lengths(train_dev_df)
        print(f"[train+dev] Esaldien batez besteko luzera: {avg_length}")
        print(f"[train+dev] Esaldien mediana luzera: {median_length}")
        print(f"[train+dev] Hiztegiaren tamaina: {len(get_vocabulary(train_dev_df))}")
        print(f"[train+dev] UPoS etiketak: {get_upos_tags(train_dev_df)}")

        # Entrenamendu eta test multzoak gorde
        path = './datasets/'
        store_csvs(train_dev_df, path, 'train_dev')
        store_csvs(test_df, path, 'test')

        print(u'Datu multzoak gorde dira! \u2713')
    else:
        print("Skipping preprocessing of the datasets...")