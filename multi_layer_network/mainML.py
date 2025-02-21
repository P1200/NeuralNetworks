from NeuralNetworkPipeline import NeuralNetworkPipeline
from multi_layer_network.MultiLayerNN import MultiLayerNN

model = MultiLayerNN(input_size=64 * 64, num_classes=80)
pipeline = NeuralNetworkPipeline(model)
pipeline.run_pipeline()
