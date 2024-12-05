import torch
from torch import nn
from models.EncoderModule import UnimodalEncoder
from models.FusionModual import MultimodalFusion
import torch.nn.functional as F


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
        lengths = input_data['length']

        # Unimodal Encoding
        unimodal_features = self.UniEncoder(au, em, hp, bp, padding_mask, lengths)

        # Dynamic Multimodal Fusion using High-level Semantic Features
        # multi_features = self.MultiFusion(unimodal_features)

        # Sentiment Classification
        # prediction = self.CLS(multi_features)     # uni_fea['T'], uni_fea['V'], uni_fea['A']
        prediction = self.CLS(unimodal_features)

        return prediction


class SentiCLS(nn.Module):
    def __init__(self, fusion_dim, n_class, classifier_dropout=0.1):
        super(SentiCLS, self).__init__()
        self.cls_layer = nn.Sequential(
            nn.Dropout(p=classifier_dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_dim // 2, n_class)
        )

    def forward(self, multi_features):
        # multi_features = torch.mean(multi_features, dim=-2)
        for Type in ['au', 'em', 'hp', 'bp']:
            multi_features[Type] = torch.mean(multi_features[Type], dim=1)
        multi_features = torch.cat((multi_features['au'], multi_features['em'], multi_features['hp'], multi_features['bp']), dim=-1)
        # multi_features = multi_features.reshape(multi_features.shape[0], -1)
        output = self.cls_layer(multi_features)
        return output


def build_model(opt):
    model = TPER(opt)
    return model
