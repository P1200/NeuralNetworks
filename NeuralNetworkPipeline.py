import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import PolLettDS.PolLettDS as pld
from ChartDrawer import ChartDrawer
from LogUtils import log_to_file, get_script_name
from Trainer import Trainer
from ImageDataset import ImageDataset


class NeuralNetworkPipeline:
    def __init__(self, model, input_size=64 * 64, num_classes=80, batch_size=32, learning_rate=0.01, num_epochs=200):
        self.model = model
        self.input_size = input_size
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.trainer = Trainer(self.device)

    def prepare_data(self, data_path='PolLettDS/pol_lett_ds.bin', labels_path='PolLettDS/pol_lett_ds_labels.bin'):
        loaded_data, loaded_labels, _ = pld.load_pol_lett_db_from_files(data_path, labels_path)

        # Reshape images
        images = loaded_data.reshape(-1, 64, 64)
        images = np.expand_dims(images, axis=1)
        images = torch.tensor(images, dtype=torch.float32)

        # Map labels to indices
        unique_labels = np.unique(loaded_labels)
        label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        mapped_labels = np.array([label_mapping[label] for label in loaded_labels])
        labels = torch.tensor(mapped_labels, dtype=torch.long)

        # Create dataset
        dataset = ImageDataset(images, labels)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size

        self.train_dataset, self.test_dataset = random_split(dataset, [train_size, test_size])
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False)

    def train_and_evaluate(self):
        print("Training started...")
        start_time = time.time()

        loss_history, accuracy_history = self.trainer.train_and_test(
            self.model, self.train_loader, self.test_loader,
            self.criterion, self.optimizer, num_epochs=self.num_epochs
        )

        end_time = time.time()
        total_time = end_time - start_time
        print(f"\nTotal training time: {total_time:.2f} seconds")
        log_entry = [total_time, " ", " ", " ", " "]
        log_to_file(log_entry, get_script_name() + ".txt")

        return loss_history, accuracy_history, total_time

    def visualize_results(self, loss_history, accuracy_history):
        epochs = range(1, len(loss_history) + 1)
        chart_drawer = ChartDrawer(epochs)
        chart_drawer.draw_loss_chart(loss_history)
        chart_drawer.draw_accuracy_chart(accuracy_history)

    def run_pipeline(self, visualize_results=True):
        self.prepare_data()
        loss_history, accuracy_history, train_time = self.train_and_evaluate()
        if visualize_results:
            self.visualize_results(loss_history, accuracy_history)
        return loss_history, accuracy_history, train_time
