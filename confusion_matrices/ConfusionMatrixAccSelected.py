import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn.functional import softmax
import PolLettDS.PolLettDS as pld
from conv_network.ConvolutionalNN import ConvolutionalNN

# Parameters
d = 1200
num_classes = 80
image_size = 64
block_size = image_size * image_size
a = block_size * d

# Load model
model = ConvolutionalNN(num_classes=num_classes)
model.load_state_dict(torch.load("../cnn_model.pth"))
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load data
loaded_data, loaded_labels, labels_count = pld.load_pol_lett_db_from_files(
    '../PolLettDS/pol_lett_db.bin',
    '../PolLettDS/pol_lett_db_labels.bin')

# Map labels to indices
unique_labels = np.unique(loaded_labels)
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
mapped_labels = np.array([label_mapping[label] for label in loaded_labels])

# Reshape
total_samples = len(mapped_labels)
assert len(loaded_data) == total_samples * block_size
images = np.reshape(loaded_data, (total_samples, image_size, image_size))

# Collect examples
examples = []
targets = []
for i in range(a, a + 80 * block_size, block_size):
    examples.append(loaded_data[i:i + block_size].reshape(64, 64))
    targets.append(mapped_labels[i // block_size])

if len(examples) < num_classes:
    raise ValueError("Nie udało się zebrać wystarczającej liczby przykładów.")

# Prepare input
x_batch = torch.tensor(np.stack(examples), dtype=torch.float32).unsqueeze(1).to(device)
y_true = torch.tensor(targets, dtype=torch.int64)

# Prediction
with torch.no_grad():
    logits = model(x_batch)
    probs = softmax(logits, dim=1).cpu().numpy()

# Build confusion matrix
conf_matrix_soft = np.zeros((num_classes, num_classes), dtype=np.float64)
for true_idx, prob_dist in zip(y_true, probs):
    conf_matrix_soft[true_idx] += prob_dist

row_sums = conf_matrix_soft.sum(axis=1, keepdims=True)
conf_matrix_percent = np.divide(conf_matrix_soft, row_sums, out=np.zeros_like(conf_matrix_soft),
                                where=row_sums != 0) * 100

# Class labels
classes = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'ą', 'b', 'c', 'ć', 'd', 'e', 'ę', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'ł', 'm', 'n', 'ń', 'o', 'ó', 'p', 'q',
    'r', 's', 'ś', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ź', 'ż',
    'A', 'Ą', 'B', 'C', 'Ć', 'D', 'E', 'Ę', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'Ł', 'M', 'N', 'Ń', 'O', 'Ó', 'P', 'Q',
    'R', 'S', 'Ś', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ź', 'Ż'
]


# Helper: extract and plot
def extract_conf_matrix_subset(conf_matrix, true_indices, pred_indices, labels):
    subset = conf_matrix[np.ix_(true_indices, pred_indices)]
    x_labels = [labels[i] for i in pred_indices]
    y_labels = [labels[i] for i in true_indices]
    return subset, x_labels, y_labels


def plot_conf_matrix(matrix, x_labels, y_labels, title, filename):
    plt.figure(figsize=(len(x_labels) * 0.6, len(y_labels) * 0.6))
    ax = sns.heatmap(matrix, xticklabels=x_labels, yticklabels=y_labels,
                annot=True, fmt='.1f', cmap='Blues', cbar=False, annot_kws={"size": 14})
    plt.xlabel("Predykcja", fontsize=25)
    plt.ylabel("Prawdziwa klasa", fontsize=25)
    ax.tick_params(axis='both', labelsize=18)
    plt.title(title, fontsize=30)
    plt.tight_layout()
    plt.savefig(filename, format='eps')
    plt.close()


# Chars to analyze
selected_chars = ['0', '6', 'l', 'ł', 'o', 'p', 't', 'u', 'v', 'z', 'ź', 'ż', 'B', 'L', 'Ł', 'O', 'U', 'P', 'V', 'Ź']

missing = [char for char in selected_chars if char not in classes]
if missing:
    raise ValueError(f"Brakuje znaków w 'classes': {missing}")

# Selected indices
selected_indices = [classes.index(char) for char in selected_chars]

selected_matrix, x_labels, y_labels = extract_conf_matrix_subset(
    conf_matrix_percent, selected_indices, selected_indices, classes)

plot_conf_matrix(
    selected_matrix,
    x_labels,
    y_labels,
    title="Wybrane znaki – macierz konfuzji",
    filename="conf_selected_chars.eps"
)
