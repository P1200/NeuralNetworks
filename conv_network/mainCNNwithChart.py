from NeuralNetworkPipeline import NeuralNetworkPipeline
from conv_network.ConvolutionalNN import ConvolutionalNN

model = ConvolutionalNN(num_classes=80)
pipeline = NeuralNetworkPipeline(model)
pipeline.run_pipeline()
