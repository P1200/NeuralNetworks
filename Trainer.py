import torch

from LogUtils import log_to_file, get_script_name


class Trainer:
    def __init__(self, device):
        self.device = device

    def train_and_test(self, model, train_loader, test_loader, criterion, optimizer, num_epochs):
        loss_history = []
        accuracy_history = []

        for epoch in range(num_epochs):
            avg_train_loss = self.train(criterion, model, optimizer, train_loader)

            avg_test_loss, test_accuracy = self.test(criterion, model, test_loader)

            loss_history.append((avg_train_loss, avg_test_loss))
            accuracy_history.append(test_accuracy)
            print(f"Epoch [{epoch + 1}/{num_epochs}] - "
                  f"Train Loss: {avg_train_loss:.4f}, "
                  f"Test Accuracy: Test Accuracy: {test_accuracy:.2f}%")
            log_entry = [" ", epoch, avg_train_loss, avg_test_loss, test_accuracy]
            log_to_file(log_entry, get_script_name() + ".txt")
        return loss_history, accuracy_history

    def test(self, criterion, model, test_loader):
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = 100 * correct / total
        return avg_test_loss, test_accuracy

    def train(self, criterion, model, optimizer, train_loader):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

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
