import copy
import torch
from torch import nn, einsum
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class PositionEncoding(nn.Module):
    """Construct the CLS token, position and patch embeddings.
    """
    def __init__(self, num_patches, fea_size, tf_hidden_dim, drop_out):
        super(PositionEncoding, self).__init__()
        # self.cls_token = nn.parameter.Parameter(torch.ones(1, 1, tf_hidden_dim))
        self.proj = nn.Linear(fea_size, tf_hidden_dim)
        self.position_embeddings = nn.parameter.Parameter(torch.zeros(1, num_patches, tf_hidden_dim))
        self.dropout = nn.Dropout(drop_out)

    def forward(self, embeddings):
        # batch_size = embeddings.shape[0]
        embeddings = self.proj(embeddings)
        # cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class TransforEncoderBlock(nn.Module):
    def __init__(self, fea_size, num_patches, nhead, dim_feedforward, num_layers, pos_dropout=0., tf_dropout=0.2):
        super(TransforEncoderBlock, self).__init__()
        self.pos_encoder = PositionEncoding(
            num_patches=num_patches,
            fea_size=fea_size,
            tf_hidden_dim=dim_feedforward,
            drop_out=pos_dropout
        )

        tfencoder_layer = TransformerEncoderLayer(dim_feedforward, nhead, dim_feedforward // 2, dropout=tf_dropout, activation='gelu', batch_first=True)
        self.tfencoder = TransformerEncoder(tfencoder_layer, num_layers)

    def forward(self, src, src_key_padding_mask):
        src = self.pos_encoder(src)
        output, hidden_list = self.tfencoder(src, mask=None, src_key_padding_mask=src_key_padding_mask)
        return output, hidden_list


class TransforEncoder(nn.Module):
    def __init__(self, modality, num_patches, fea_size, dim_feedforward, nhead=8, num_layers=4):
        super(TransforEncoder, self).__init__()
        self.tfencoder = TransforEncoderBlock(
            fea_size=fea_size,
            num_patches=num_patches,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=num_layers
        )
        self.layernorm = nn.LayerNorm(dim_feedforward)

    def forward(self, inputs, key_padding_mask):
        tf_last_hidden_state, tf_hidden_state_list = self.tfencoder(inputs, src_key_padding_mask=key_padding_mask)

        # TransformerEncoder提取的领域知识
        spci_know = self.layernorm(tf_last_hidden_state)

        # 对不同知识进行整合
        uni_hidden = torch.cat([spci_know], dim=-1)
        return uni_hidden

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class UnimodalEncoder(nn.Module):
    def __init__(self, opt):
        super(UnimodalEncoder, self).__init__()
        # All Encoders of Each Modality
        self.enc_t = TransforEncoder(modality="T", num_patches=opt.seq_lens[0], fea_size=768)
        self.enc_v = TransforEncoder(modality="V", num_patches=opt.seq_lens[1], fea_size=709)
        self.enc_a = TransforEncoder(modality="A", num_patches=opt.seq_lens[2], fea_size=33)

        # LSTM Encoder for learning sequential features

    def forward(self, inputs_data_mask):
        hidden_t = self.enc_t(inputs_data_mask)
        hidden_v = self.enc_v(inputs_data_mask)
        hidden_a = self.enc_a(inputs_data_mask)

        return {'T': hidden_t, 'V': hidden_v, 'A': hidden_a}
