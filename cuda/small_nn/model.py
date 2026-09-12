import os
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def save_float32(path: str, tensor: torch.Tensor) -> None:
    array = tensor.detach().cpu().contiguous().numpy().astype(np.float32)
    array.tofile(path)

class Classifier(nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_dim: int):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_size)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self.layer1(tensor)
        tensor = F.relu(tensor)
        tensor = self.layer2(tensor)
        return tensor


input_size = 4096
output_size = 16
hidden_dim = 256
batch_size = 32

torch.manual_seed(42)
random_tensor: torch.Tensor = torch.randn(batch_size, input_size)
model = Classifier(input_size, output_size, hidden_dim)
with torch.no_grad():
    return_tensor = model(random_tensor)

print(return_tensor[0])

# Save activations as [features, batch] to match the layout model.cu expects.
save_float32("tensors/input.bin", random_tensor.T)
save_float32("tensors/layer1_weights.bin", model.layer1.weight)
save_float32("tensors/layer1_bias.bin", model.layer1.bias)
save_float32("tensors/layer2_weights.bin", model.layer2.weight)
save_float32("tensors/layer2_bias.bin", model.layer2.bias)
save_float32("tensors/output.bin", return_tensor.T)
