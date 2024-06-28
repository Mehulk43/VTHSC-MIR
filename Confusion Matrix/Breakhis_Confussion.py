import matplotlib.pyplot as plt
import numpy as np

# Function to draw confusion matrix
def plot_confusion_matrix(tp, tn, fp, fn):
    matrix_data = np.array([[tp, fp], [fn, tn]])

    plt.imshow(matrix_data, cmap='Blues', interpolation='nearest')
    plt.title('Confusion Matrix for Breakhis')
    plt.colorbar()

    # Add labels
    plt.xticks([0, 1], ['Malignant', 'Benign'])
    plt.yticks([0, 1], ['Malignant', 'Benign'])

    # Add counts to each cell
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(matrix_data[i, j]), ha='center', va='center', color='black')

    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.show()

# Example counts (replace with your actual values)
tp = 1395
tn = 595
fp = 43
fn = 48


# Plot confusion matrix
plot_confusion_matrix(tp, tn, fp, fn)
