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
        self.MultiFusion = MultimodalFusion(opt)

        # Classification Prediction
        self.CLS = SentiCLS(fusion_dim=256*8, n_class=opt.num_class)

    def forward(self, input_data):
        au, em, hp, bp = input_data['au'], input_data['em'], input_data['hp'], input_data['bp']
        padding_mask = input_data['padding_mask']
        lengths = input_data['length']

        # Unimodal Encoding
        uni_token, uni_utterance = self.UniEncoder(au, em, hp, bp, padding_mask, lengths)

        # Dynamic Multimodal Fusion using High-level Semantic Features
        multi_fea = self.MultiFusion(uni_token)

        # Sentiment Classification
        # prediction = self.CLS(multi_features)     # uni_fea['T'], uni_fea['V'], uni_fea['A']
        prediction = self.CLS(uni_utterance, multi_fea)

        return prediction


class SentiCLS(nn.Module):
    def __init__(self, fusion_dim, n_class, classifier_dropout=0.1):
        super(SentiCLS, self).__init__()
        self.cls_layer = nn.Sequential(
            nn.GELU(),
            nn.Dropout(p=classifier_dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Linear(fusion_dim // 2, n_class)
        )

    def forward(self, uni_fea, multi_fea):
        multi_fea = torch.mean(multi_fea, dim=1)
        # for Type in ['au', 'em', 'hp', 'bp']:
        #     multi_fea[Type] = torch.mean(multi_fea[Type], dim=1)

        joint_fea = torch.cat((uni_fea['au'], uni_fea['em'], uni_fea['hp'], uni_fea['bp'], multi_fea), dim=-1)
        output = self.cls_layer(joint_fea)
        return output


def build_model(opt):
    model = TPER(opt)
    return model
