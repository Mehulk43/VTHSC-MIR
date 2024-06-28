import numpy as np
import matplotlib.pyplot as plt

# Function to plot confusion matrix
def plot_confusion_matrix(tp, tn, fp, fn, class_labels):
    num_classes = len(class_labels)
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

    for i in range(num_classes):
        confusion_matrix[i, i] = tp[i] + tn[i]
        for j in range(num_classes):
            if j != i:
                confusion_matrix[i, j] = fp[i] + fn[j]

    plt.figure(figsize=(10, 8))
    plt.imshow(confusion_matrix, cmap='Blues', interpolation='nearest')
    plt.title('Confusion Matrix')
    plt.colorbar()

    plt.xticks(np.arange(num_classes), class_labels, rotation=45)
    plt.yticks(np.arange(num_classes), class_labels)

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(confusion_matrix[i, j]), ha='center', va='center', color='black')

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.show()

# Example TP, TN, FP, FN values for each class (replace with your actual values)
Actual_label = [113, 260, 121, 150, 903, 170, 222, 142]  # True positives + True negatives 



tp_values = [53,124,57,72,427,81,106,67]
tn_values = [53,123,56,72,426,80,106,67]
wrong_classifications = [7, 13, 8, 6, 50, 9, 10, 8]           # False positives + False negatives
fp_values = [5, 10, 5, 4, 20, 5, 6, 3]
fn_values = [2, 3, 3, 2, 30, 4, 4, 5]

# Example class labels
class_labels = ['Adenosis', 'Fibroadenoma','Phyllodes_tumor',  'Tubular_adenoma', 'Ductal_carcinoma',  'Lobular_carcinoma', 'Mucinous_carcinoma', 'Papillary_carcinoma']

# Plot confusion matrix
plot_confusion_matrix(tp_values, tn_values, fp_values, fn_values, class_labels)
