import torch
from torch import nn
from models.EncoderModule import UnimodalEncoder
from models.FusionModual import MultimodalFusion


class TPER(nn.Module):
    def __init__(self, opt):
        super(TPER, self).__init__()
        # Unimodal Embedding and Encoder
        self.UniEncoder = UnimodalEncoder(opt)

        # Multimodal Interaction and Fusion
        # self.MultiFusion = MultimodalFusion(opt)

        # Classification Prediction
        self.CLS = SentiCLS(fusion_dim=512*4, n_class=opt.num_class)

    def forward(self, input_data):
        au, em, hp, bp = input_data['au'], input_data['em'], input_data['hp'], input_data['bp']
        padding_mask = input_data['padding_mask']

        # Unimodal Encoding
        unimodal_features = self.UniEncoder(au, em, hp, bp, padding_mask)

        # Dynamic Multimodal Fusion using High-level Semantic Features
        # multimodal_features = self.MultiFusion(unimodal_features)

        # Sentiment Classification
        prediction = self.CLS(unimodal_features)     # uni_fea['T'], uni_fea['V'], uni_fea['A']

        return prediction


class SentiCLS(nn.Module):
    def __init__(self, fusion_dim, n_class):
        super(SentiCLS, self).__init__()
        self.cls_layer = nn.Sequential(
            nn.Linear(fusion_dim, 128, bias=True),
            nn.GELU(),
            nn.Linear(128, 32, bias=True),
            # nn.GELU(),
            nn.Linear(32, n_class, bias=True)
        )

    def forward(self, uni_features):
        # fusion_features = torch.mean(fusion_features, dim=-2)
        # output = self.cls_layer(fusion_features)
        # return output
        for Type in ['au', 'em', 'hp', 'bp']:
            uni_features[Type] = torch.mean(uni_features[Type], dim=1)
        features = torch.cat((uni_features['au'], uni_features['em'], uni_features['hp'], uni_features['bp']), dim=1)
        output = self.cls_layer(features)
        return output


def build_model(opt):
    model = TPER(opt)
    return model
