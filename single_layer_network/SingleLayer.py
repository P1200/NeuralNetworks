import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import PolLettDB.PolLettDB as pld
from ChartDrawer import ChartDrawer
from Trainer import Trainer
from single_layer_network.ImageDataset import ImageDataset
from single_layer_network.SingleLayerNN import SingleLayerNN

INPUT_SIZE = 64 * 64
NUM_CLASSES = 80


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SingleLayerNN(INPUT_SIZE, NUM_CLASSES).to(device)

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

images = np.expand_dims(images, axis=1)
images = torch.tensor(images, dtype=torch.float32)

dataset = ImageDataset(images, labels)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

trainer = Trainer(device)

start_time = time.time()
loss_history, accuracy_history = trainer.train_and_test(model, train_loader, test_loader, criterion,
                                                        optimizer,
                                                        num_epochs=200)
end_time = time.time()
total_time = end_time - start_time

print(f"\nTotal time of training: {total_time:.2f} seconds")

epochs = range(1, len(loss_history) + 1)

chart_drawer = ChartDrawer(epochs)
chart_drawer.draw_loss_chart(loss_history)
chart_drawer.draw_accuracy_chart(accuracy_history)
