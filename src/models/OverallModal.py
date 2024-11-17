import torch
from torch import nn
from EncoderModule import UnimodalEncoder
from FusionModual import MultimodalFusion


class TPER(nn.Module):
    def __init__(self, opt):
        super(TPER, self).__init__()
        # Unimodal Embedding and Encoder
        self.UniEncoder = UnimodalEncoder(opt)

        # Multimodal Interaction and Fusion
        self.MultiFusion = MultimodalFusion(opt)

        # Classification Prediction
        self.CLS = SentiCLS(fusion_dim=256, n_class=opt.n_class)

    def forward(self, input_data):
        au, em, hp, bp = input_data['au'], input_data['em'], input_data['hp'], input_data['bp']

        # Unimodal Encoding
        unimodal_features = self.UniEncoder(au, em, hp, bp)

        # Dynamic Multimodal Fusion using High-level Semantic Features
        multimodal_features = self.MultiFusion(unimodal_features)

        # Sentiment Classification
        prediction = self.CLS(unimodal_features, multimodal_features)     # uni_fea['T'], uni_fea['V'], uni_fea['A']

        return prediction


class SentiCLS(nn.Module):
    def __init__(self, fusion_dim, n_class):
        super(SentiCLS, self).__init__()
        self.cls_layer = nn.Sequential(
            nn.Linear(256, 64, bias=True),
            nn.GELU(),
            nn.Linear(64, 32, bias=True),
            nn.GELU(),
            nn.Linear(32, n_class, bias=True)
        )

    def forward(self, unimodal_features, fusion_features):
        fusion_features = torch.mean(fusion_features, dim=-2)
        output = self.cls_layer(fusion_features)
        return output


def build_model(opt):
    model = TPER(opt)
    return model
