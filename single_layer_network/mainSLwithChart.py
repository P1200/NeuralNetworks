import numpy as np

from NeuralNetworkPipeline import NeuralNetworkPipeline
from single_layer_network.SingleLayerNN import SingleLayerNN

model = SingleLayerNN(input_size=64 * 64, num_classes=80)
pipeline = NeuralNetworkPipeline(model)
pipeline.run_pipeline()
