# generate_graphs.py
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def generate_graphs(json_file, class_names):
    # Load the experiment results
    with open(json_file, 'r') as f:
        experiment_results = json.load(f)

    # Initialize lists to store metrics across epochs
    epochs = []
    train_accuracies = []
    eval_accuracies = []
    train_f1_scores = []
    eval_f1_scores = []
    train_recalls = []
    eval_recalls = []
    train_precisions = []
    eval_precisions = []

    # Extract the metrics for each epoch
    for epoch, results in experiment_results.items():
        epochs.append(int(epoch))
        train_accuracies.append(results['train_metrics']['accuracy'])
        eval_accuracies.append(results['eval_metrics']['accuracy'])
        train_f1_scores.append(results['train_metrics']['f1_score'])
        eval_f1_scores.append(results['eval_metrics']['f1_score'])
        train_recalls.append(results['train_metrics']['recall'])
        eval_recalls.append(results['eval_metrics']['recall'])
        train_precisions.append(results['train_metrics']['precision'])
        eval_precisions.append(results['eval_metrics']['precision'])

    # Accuracy vs Epochs
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, label='Training Accuracy', marker='o')
    plt.plot(epochs, eval_accuracies, label='Evaluation Accuracy', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.legend()
    plt.grid()
    plt.savefig('../results/graphs/accuracy_vs_epochs.png')
    plt.close()

    # F1 Score vs Epochs
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_f1_scores, label='Training F1 Score', marker='o')
    plt.plot(epochs, eval_f1_scores, label='Evaluation F1 Score', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.title('F1 Score vs Epochs')
    plt.legend()
    plt.grid()
    plt.savefig('../results/graphs/f1_score_vs_epochs.png')
    plt.close()

    # Precision vs Epochs
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_precisions, label='Training Precision', marker='o')
    plt.plot(epochs, eval_precisions, label='Evaluation Precision', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Precision')
    plt.title('Precision vs Epochs')
    plt.legend()
    plt.grid()
    plt.savefig('../results/graphs/precision_vs_epochs.png')
    plt.close()

    # Recall vs Epochs
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_recalls, label='Training Recall', marker='o')
    plt.plot(epochs, eval_recalls, label='Evaluation Recall', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Recall')
    plt.title('Recall vs Epochs')
    plt.legend()
    plt.grid()
    plt.savefig('../results/graphs/recall_vs_epochs.png')
    plt.close()

    # Confusion Matrix for the final epoch
    final_epoch = str(max(epochs))
    conf_matrix = np.array(experiment_results[final_epoch]['eval_metrics']['confusion_matrix'])

    plt.figure(figsize=(12, 10))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='.2f', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix for Epoch {final_epoch}')
    plt.savefig('../results/graphs/confusion_matrix_epoch_{}.png'.format(final_epoch))
    plt.close()

    # Generate a text file summarizing metrics for each class across epochs
    with open('../results/tables/metrics_summary.txt', 'w') as summary_file:
        header = "Class      | Recall  | Precision | F1 Score | Accuracy\n"
        summary_file.write(header)
        summary_file.write("-" * len(header) + "\n")
        for epoch, results in experiment_results.items():
            summary_file.write(f"Epoch {epoch}\n")
            for class_idx, class_name in enumerate(class_names):
                recall = results['eval_metrics'].get('recall', [])[class_idx] if isinstance(results['eval_metrics'].get('recall', []), list) else results['eval_metrics']['recall']
                precision = results['eval_metrics'].get('precision', [])[class_idx] if isinstance(results['eval_metrics'].get('precision', []), list) else results['eval_metrics']['precision']
                f1_score = results['eval_metrics'].get('f1_score', [])[class_idx] if isinstance(results['eval_metrics'].get('f1_score', []), list) else results['eval_metrics']['f1_score']
                accuracy = results['eval_metrics'].get('accuracy', [])[class_idx] if isinstance(results['eval_metrics'].get('accuracy', []), list) else results['eval_metrics']['accuracy']
                summary_file.write(f"{class_name:<10} | {recall:.4f} | {precision:.4f} | {f1_score:.4f} | {accuracy:.4f}\n")
            summary_file.write("\n")


if __name__ == '__main__':
    # Example usage
    json_file = 'experiment_results.json'
    class_names = ['ABE', 'ART', 'BAS', 'BLA', 'EBO', 'EOS', 'FGC', 'HAC', 'KSC', 'LYI',
                   'LYT', 'MMZ', 'MON', 'MYB', 'NGB', 'NGS', 'NIF', 'OTH', 'PEB', 'PLM', 'PMO']
    generate_graphs(json_file, class_names)
