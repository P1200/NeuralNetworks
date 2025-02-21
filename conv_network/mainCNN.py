from NeuralNetworkPipeline import NeuralNetworkPipeline
from conv_network.ConvolutionNN import ConvolutionNN

model = ConvolutionNN(num_classes=80)
pipeline = NeuralNetworkPipeline(model)
pipeline.run_pipeline()
