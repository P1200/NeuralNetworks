from matplotlib import pyplot as plt


class ChartDrawer:
    def __init__(self, epochs_number):
        self.epochs_number = epochs_number

    def draw_loss_chart(self, loss_history):
        train_losses = [loss[0] for loss in loss_history]
        test_losses = [loss[1] for loss in loss_history]

        plt.figure(figsize=(8, 6))
        plt.plot(self.epochs_number, train_losses, label='Strata na zbiorze uczącym', marker='o', color='blue')
        plt.plot(self.epochs_number, test_losses, label='Strata na zbiorze testowym', marker='o', color='red')
        plt.title('Strata na zbiorze uczącym i testowym')
        plt.xlabel('Numer epoki')
        plt.ylabel('Strata')
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.show()

    def draw_accuracy_chart(self, accuracy_history):
        plt.figure(figsize=(8, 6))
        plt.plot(self.epochs_number, accuracy_history, label='Dokładność na zbiorze testowym', marker='o', color='red')
        plt.title('Dokładność na zbiorze testowym')
        plt.xlabel('Numer epoki')
        plt.ylabel('Dokładność')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.show()
