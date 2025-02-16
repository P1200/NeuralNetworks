import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, random_split
import PolLettDB.PolLettDB as pld
from single_layer_network.ImageDataset import ImageDataset
from single_layer_network.SingleLayerNN import SingleLayerNN

INPUT_SIZE = 64 * 64
NUM_CLASSES = 80


def train_validate_test(model, train_loader, val_loader, test_loader, criterion, optimizer, num_epochs):
    loss_history = []
    accuracy_history = []

    for epoch in range(num_epochs):
        avg_train_loss = train(criterion, model, optimizer, train_loader)

        avg_val_loss, val_accuracy = validate(criterion, model, val_loader)

        avg_test_loss, test_accuracy = test(criterion, model, test_loader)

        loss_history.append((avg_train_loss, avg_val_loss, avg_test_loss))
        accuracy_history.append((val_accuracy, test_accuracy))
        print(f"Epoch [{epoch + 1}/{num_epochs}] - "
              f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, "
              f"Test Accuracy: Test Accuracy: {test_accuracy:.2f}%")
    return loss_history, accuracy_history


def test(criterion, model, test_loader):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct / total
    return avg_test_loss, test_accuracy


def validate(criterion, model, val_loader):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct / total
    return avg_val_loss, val_accuracy


def train(criterion, model, optimizer, train_loader):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_loader)
    return avg_train_loss


def augment_dataset(images, labels, shift_pixels=5, rotation_degrees=1):
    h, w = images[0].shape[:2]

    augmented_images = []
    augmented_labels = []

    for image, label in zip(images, labels):
        augmented_images.append(image)  # Dodaj oryginalny obraz
        augmented_labels.append(label)

        # Przesunięcie i obrót dla różnych wariantów
        for angle in [-rotation_degrees, 0, rotation_degrees]:
            for dx in [-shift_pixels, 0, shift_pixels]:
                for dy in [-shift_pixels, 0, shift_pixels]:
                    if dx == 0 and dy == 0 and angle == 0:
                        continue  # Pomijamy oryginalny obrazek

                    # Przesunięcie
                    M_translate = np.float32([[1, 0, dx], [0, 1, dy]])
                    translated_image = cv2.warpAffine(image, M_translate, (w, h), borderMode=cv2.BORDER_REFLECT)

                    # Obrót
                    M_rotate = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)
                    rotated_image = cv2.warpAffine(translated_image, M_rotate, (w, h), borderMode=cv2.BORDER_REFLECT)

                    augmented_images.append(rotated_image)
                    augmented_labels.append(label)

    # Konwersja listy do tablicy NumPy (znacznie szybsze niż append)
    return np.array(augmented_images), np.array(augmented_labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SingleLayerNN(INPUT_SIZE, NUM_CLASSES).to(device)

loaded_data, loaded_labels, labels_count = pld.load_pol_lett_db_from_files(
    'PolLettDB/pol_lett_db.bin',
    'PolLettDB/pol_lett_db_labels.bin')

raw_data = loaded_data
raw_labels = loaded_labels

images = raw_data.reshape(4160, 64, 64)

augmented_data, augmented_labels = augment_dataset(images, raw_labels)

images = np.expand_dims(augmented_data, axis=1)
images = torch.tensor(images, dtype=torch.float32)

unique_labels = np.unique(augmented_labels)
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

mapped_labels = np.array([label_mapping[label] for label in augmented_labels])
labels = torch.tensor(mapped_labels, dtype=torch.long)

dataset = ImageDataset(images, labels)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

loss_history, accuracy_history = train_validate_test(model, train_loader, val_loader, test_loader, criterion, optimizer,
                                                     num_epochs=100)

train_losses = [loss[0] for loss in loss_history]
val_losses = [loss[1] for loss in loss_history]
test_losses = [loss[2] for loss in loss_history]

val_acc = [loss[0] for loss in accuracy_history]
test_acc = [loss[1] for loss in accuracy_history]

epochs = range(1, len(loss_history) + 1)

plt.figure(figsize=(8, 6))
plt.plot(epochs, train_losses, label='Train Loss', marker='o')
plt.plot(epochs, val_losses, label='Validation Loss', marker='o')
plt.plot(epochs, test_losses, label='Test Loss', marker='o')
plt.title('Train, Validation and Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(epochs, val_acc, label='Validation Acc', marker='o')
plt.plot(epochs, test_acc, label='Test Acc', marker='o')
plt.title('Validation and Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
