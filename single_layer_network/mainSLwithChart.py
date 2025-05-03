import torch

from NeuralNetworkPipeline import NeuralNetworkPipeline
from single_layer_network.SingleLayerNN import SingleLayerNN

model = SingleLayerNN(input_size=64 * 64, num_classes=80)
pipeline = NeuralNetworkPipeline(model)
pipeline.run_pipeline()

model_save_path = "sl_model.pth"
torch.save(model.state_dict(), model_save_path)
