from matplotlib import pyplot as plt


class ChartDrawer:
    def __init__(self, epochs_number):
        self.epochs_number = epochs_number

    def draw_loss_chart(self, loss_history):
        train_losses = [loss[0] for loss in loss_history]
        test_losses = [loss[1] for loss in loss_history]

        plt.figure(figsize=(8, 6))
        plt.plot(self.epochs_number, train_losses, label='Train Loss', marker='o', color='blue')
        plt.plot(self.epochs_number, test_losses, label='Test Loss', marker='o', color='red')
        plt.title('Train, Validation and Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.show()

    def draw_accuracy_chart(self, accuracy_history):
        plt.figure(figsize=(8, 6))
        plt.plot(self.epochs_number, accuracy_history, label='Test Acc', marker='o', color='red')
        plt.title('Validation and Test Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.show()
