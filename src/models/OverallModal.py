import torch
from torch import nn


class KMSA(nn.Module):
    def __init__(self, opt):
        super(KMSA, self).__init__()

    def forward(self, input_data):
        uni_fea, uni_senti = self.UniEncKI(input_data)    # [T, V, A]

        multimodal_features, nce_loss = self.DyMultiFus(uni_fea)

        prediction = self.CLS(multimodal_features)     # uni_fea['T'], uni_fea['V'], uni_fea['A']

        return prediction, nce_loss


def build_model(opt):
    model = KMSA(opt)
    return model
