import torch
from torch import nn


class KMSA(nn.Module):
    def __init__(self, opt):
        super(KMSA, self).__init__()
        self.linear1 = nn.Linear(300, 64, bias=True)
        self.linear2 = nn.Linear(64, 1, bias=True)

    def forward(self, input_data):
        data = torch.mean(input_data, dim=1)

        pred = torch.tanh(self.linear1(data))
        prediction = self.linear2(pred)

        return prediction


def build_model(opt):
    model = KMSA(opt)
    return model
