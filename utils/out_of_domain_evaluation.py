import nltk
from nltk.corpus import treebank
from nltk.tag import map_tag


def ptb_dataloader():
    """
    Extracts sentences and universal tags from the Penn Treebank corpus.

    Returns:
        sentences (list of list of str): List of sentences, where each sentence is a list of words.
        tags (list of list of str): List of tag sequences, where each sequence is a list of universal tags corresponding to words.
    """

    nltk.download('treebank')

    tagged_sents = treebank.tagged_sents()

    sentences = []
    tags = []

    for sent in tagged_sents:
        words, pos_tags = zip(*sent)
        universal_tags = [map_tag('en-ptb', 'universal', tag) for tag in pos_tags]
        sentences.append(list(words))
        tags.append(universal_tags)

    return sentences, tags

# Extract sentences and tags
# sentences, tags = ptb_dataloader()

# print("First sentence:", sentences[0])
# print("First sentence tags:", tags[0])
# print("Set of all tags:", {tag for tag_list in tags for tag in tag_list})