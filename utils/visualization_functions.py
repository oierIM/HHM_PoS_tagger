from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import numpy as np

def plot_confusion_matrix(cm, unique_tags, title):
    """
    Plots the Confusion Matrix.

    Args:
    cm (sklearn.metrics.confusion_matrix): Confusion matrix.
    unique_tags (list of str): List of unique PoS tags.
    title (str): Title to use. Some posible titles: "In-Domain Confusion Matrix of POS Tagging", "Out-of-Domain Confusion Matrix of POS Tagging", ...
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=unique_tags, yticklabels=unique_tags
    )
    plt.xlabel("Predicted Tags")
    plt.ylabel("True Tags")
    plt.title(title)
    plt.savefig("confussion_matrix")

def plot_f1_scores(unique_tags, precision, recall, f1_scores, title):
    """
    Plots the Precision, Recall and F1-scores for each tag as grouped bar plots.
    The average F1-score is shown as a horizontal line.

    Args:
    unique_tags (list of str): List of POS tags.
    precision (list of float): Precision scores for each tag.
    recall (list of float): Recall scores for each tag.
    f1_scores (list of float): F1-scores for each tag.
    title (str): Title to use. Some posible titles: "In-Domain Precision, Recall and F1-Scores for Each POS Tag", "Out-of-Domain Precision, Recall and F1-Scores for Each POS Tag", ...
    """
    # Calculate the average F1 score
    avg_f1_score = np.mean(f1_scores)

    # Data preparation for grouped bar plot
    metrics = ['Precision', 'Recall', 'F1-Score']
    metrics_data = [precision, recall, f1_scores]
    x = np.arange(len(unique_tags))  # Tag indices for the x-axis

    # Define bar width and figure size
    bar_width = 0.2
    plt.figure(figsize=(16, 4))

    colors = ['#a6a6a6', '#737373', '#000000']  # Light grey, dark grey, black
    
    for i, (metric, data, color) in enumerate(zip(metrics, metrics_data, colors)):
        plt.bar(x + i * bar_width, data, width=bar_width, label=metric, color=color)

    plt.axhline(avg_f1_score, color='red', linestyle='--', label=f'Avg F1-Score: {avg_f1_score:.2f}')

    # Customize the plot
    plt.xticks(x + bar_width, unique_tags, rotation=45, ha='right')
    plt.xlabel('POS Tags', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title(title, fontsize=16)
    plt.legend()
    plt.tight_layout()

    plt.savefig("f1-scores", dpi=300)
    plt.show()