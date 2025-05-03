import torch

from NeuralNetworkPipeline import NeuralNetworkPipeline
from conv_network.ConvolutionalNN import ConvolutionalNN

model = ConvolutionalNN(num_classes=80)
pipeline = NeuralNetworkPipeline(model, num_epochs=20)
pipeline.run_pipeline()

model_save_path = "cnn_model.pth"
torch.save(model.state_dict(), model_save_path)
