import numpy as np

from LogUtils import log_hardware_info, get_script_name
from NeuralNetworkPipeline import NeuralNetworkPipeline
from multi_layer_network.MultiLayerNN import MultiLayerNN

repeat_teaching_number = 100
samples_to_mean = 10

mean_accuracy = 0
mean_train_time = 0
for individual in range(repeat_teaching_number):
    model = MultiLayerNN(input_size=64 * 64, num_classes=80)
    pipeline = NeuralNetworkPipeline(model)
    loss_history, accuracy_history, train_time = pipeline.run_pipeline(False)
    mean_accuracy += np.sum(np.sort(accuracy_history)[-samples_to_mean:][::-1])
    mean_train_time += train_time

print("Mean accuracy in " + str(repeat_teaching_number) + " models is " + str(mean_accuracy / (repeat_teaching_number * samples_to_mean)))
print("Mean train time: " + str(mean_train_time / repeat_teaching_number))
log_hardware_info(get_script_name() + ".txt")
