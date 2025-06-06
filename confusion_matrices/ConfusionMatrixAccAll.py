import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn.functional import softmax
import PolLettDS.PolLettDS as pld
from conv_network.ConvolutionalNN import ConvolutionalNN

# Parameters
d = 1200  # start index
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
    '../PolLettDS/pol_lett_ds.bin',
    '../PolLettDS/pol_lett_ds_labels.bin')

# Map labels to indices
unique_labels = np.unique(loaded_labels)
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
mapped_labels = np.array([label_mapping[label] for label in loaded_labels])

# Transform data (n_samples, 64, 64)
total_samples = len(mapped_labels)
assert len(loaded_data) == total_samples * block_size, "Rozmiar danych nie pasuje do liczby etykiet"

images = np.reshape(loaded_data, (total_samples, image_size, image_size))

# Get data
examples = []
targets = []
chosen_indices = []

for i in range(a, a + 80 * block_size, block_size):
    examples.append(loaded_data[i:i + block_size].reshape(64, 64))
    targets.append(mapped_labels[i // block_size])

if len(examples) < num_classes:
    raise ValueError(f"Nie udało się zebrać {num_classes} przykładów z poprawnymi etykietami.")

# Prepare batch
x_batch = torch.tensor(np.stack(examples), dtype=torch.float32).unsqueeze(1).to(device)
y_true = torch.tensor(targets, dtype=torch.int64)

# Prediction
with torch.no_grad():
    logits = model(x_batch)
    probs = softmax(logits, dim=1).cpu().numpy()  # shape: (80, 80)

# Confusion matrix
conf_matrix_soft = np.zeros((num_classes, num_classes), dtype=np.float64)

for true_idx, prob_dist in zip(y_true, probs):
    conf_matrix_soft[true_idx] += prob_dist

# Count percents
row_sums = conf_matrix_soft.sum(axis=1, keepdims=True)
conf_matrix_percent = np.divide(conf_matrix_soft, row_sums, out=np.zeros_like(conf_matrix_soft), where=row_sums != 0) * 100

classes = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'ą', 'b', 'c', 'ć', 'd', 'e', 'ę', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'ł', 'm', 'n', 'ń', 'o', 'ó', 'p', 'q',
    'r', 's', 'ś', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'ź', 'ż',
    'A', 'Ą', 'B', 'C', 'Ć', 'D', 'E', 'Ę', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'Ł', 'M', 'N', 'Ń', 'O', 'Ó', 'P', 'Q',
    'R', 'S', 'Ś', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Ź', 'Ż'
]

# Visualization
plt.figure(figsize=(50, 50))
sns.heatmap(conf_matrix_percent, xticklabels=classes, yticklabels=classes,
            annot=True, fmt='.1f', cmap='Blues', cbar=True)

plt.xlabel("Predykcja")
plt.ylabel("Prawdziwa klasa")
plt.title("Miękka macierz konfuzji (% prawdopodobieństw)")
plt.tight_layout()
plt.savefig("confusion_matrix_acc.png", dpi=300)
plt.show()
