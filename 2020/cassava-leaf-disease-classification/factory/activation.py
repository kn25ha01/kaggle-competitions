import torch
import torch.nn.functional as F
from torch import nn


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


if __name__ == "__main__":
    class model_sample(nn.Module):
        def __init__(self):
            super().__init__()
            self.activation = Mish()

        def forward(self, x):
            return self.activation(x)

    model = model_sample()
    x = torch.Tensor([1,2,3])
    y = model(x)
    print(f'x : {x}')
    print(f'y : {y}')
