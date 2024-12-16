from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np

def plot_confusion_matrix(cm, unique_tags):
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=unique_tags, yticklabels=unique_tags
    )
    plt.xlabel("Predicted Tags")
    plt.ylabel("True Tags")
    plt.title("Confusion Matrix of POS Tagging")
    plt.savefig("confussion_matrix")