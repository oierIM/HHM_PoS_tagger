import numpy as np
from collections import defaultdict, Counter

class HHMPOSTagger:
    def __init__(self):
        self.tags = set() #Set of tags
        self.transition_probs = defaultdict(lambda:defaultdict(float)) #Dictionary to save transition probabilities
        self.emmision_probs = defaultdict(lambda:defaultdict(float)) #Dictionary to save emmision probabilities
        self.tag_counts = Counter() #To count each tag
        self.word_counts = Counter() #To count each word