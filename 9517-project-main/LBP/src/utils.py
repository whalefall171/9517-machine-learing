import matplotlib.pyplot as plt

def plot_metrics(acc, prec, rec, f1, classifier_name="Classifier"):
    metrics = [acc, prec, rec, f1]
    names = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    colors = ['#4caf50', '#2196f3', '#ff9800', '#9c27b0']

    plt.figure(figsize=(8, 5))
    plt.bar(names, metrics, color=colors)
    plt.ylim(0, 1)
    for i, v in enumerate(metrics):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')
    plt.title(f"{classifier_name} Performance Metrics")
    plt.ylabel("Score")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
