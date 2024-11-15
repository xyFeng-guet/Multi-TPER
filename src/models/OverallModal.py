import torch
from torch import nn


class KMSA(nn.Module):
    def __init__(self, opt):
        super(KMSA, self).__init__()
        self.linear1 = nn.Linear(300, 64, bias=True)
        self.linear2 = nn.Linear(64, 2, bias=True)

    def forward(self, input_data):
        input_data = torch.column_stack((input_data['au'], input_data['em'], input_data['hp'], input_data['bp']))    # person_num * seq_num * seq_len(300)
        data = torch.mean(input_data, dim=1)

        pred = torch.sigmoid(self.linear1(data))
        prediction = self.linear2(pred)

        return prediction


def build_model(opt):
    model = KMSA(opt)
    return model
