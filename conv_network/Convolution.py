import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import PolLettDB.PolLettDB as pld
from Augmenter import Augmenter
from ChartDrawer import ChartDrawer
from Trainer import Trainer
from conv_network.ConvolutionNN import ConvolutionNN
from single_layer_network.ImageDataset import ImageDataset

INPUT_SIZE = 64 * 64
NUM_CLASSES = 80


def random_split_data(X, y, split_sizes):
    assert sum(split_sizes) == 1.0, "Sumaryczny podział musi wynosić 100%"

    # 🔀 Losowe permutowanie indeksów (gwarantuje, że dane i etykiety pozostaną zgodne)
    indices = np.random.permutation(len(X))
    X, y = X[indices], y[indices]

    # Obliczenie rozmiarów zbiorów
    train_size = int(len(X) * split_sizes[0])
    val_size = int(len(X) * split_sizes[1])

    # Podział na trzy zbiory
    X_train, X_val, X_test = np.split(X, [train_size, train_size + val_size])
    y_train, y_val, y_test = np.split(y, [train_size, train_size + val_size])

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvolutionNN(NUM_CLASSES).to(device)

loaded_data, loaded_labels, labels_count = pld.load_pol_lett_db_from_files(
    'PolLettDB/pol_lett_db.bin',
    'PolLettDB/pol_lett_db_labels.bin')

raw_data = loaded_data
raw_labels = loaded_labels

images = raw_data.reshape(4160, 64, 64)

unique_labels = np.unique(raw_labels)
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

mapped_labels = np.array([label_mapping[label] for label in raw_labels])
labels = torch.tensor(mapped_labels, dtype=torch.long)

(train_data, train_labels), (val_data, val_labels), (test_data, test_labels) = (
    random_split_data(images, labels, [0.7, 0.15, 0.15]))

# augmenter = Augmenter()
# augmented_train_data, augmented_train_labels = augmenter.augment_dataset(train_data, train_labels)
augmented_train_data = train_data
augmented_train_labels = train_labels

train_images = np.expand_dims(augmented_train_data, axis=1)
train_images = torch.tensor(train_images, dtype=torch.float32)

val_images = np.expand_dims(val_data, axis=1)
val_images = torch.tensor(val_images, dtype=torch.float32)

test_images = np.expand_dims(test_data, axis=1)
test_images = torch.tensor(test_images, dtype=torch.float32)

train_dataset = ImageDataset(train_images, augmented_train_labels)
val_dataset = ImageDataset(val_images, val_labels)
test_dataset = ImageDataset(test_images, test_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

trainer = Trainer(device)

loss_history, accuracy_history = trainer.train_validate_test(model, train_loader, val_loader, test_loader, criterion,
                                                             optimizer,
                                                             num_epochs=200)

epochs = range(1, len(loss_history) + 1)

chart_drawer = ChartDrawer(epochs)
chart_drawer.draw_loss_chart(loss_history)
chart_drawer.draw_accuracy_chart(accuracy_history)
