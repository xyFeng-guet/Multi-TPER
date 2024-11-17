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
        output = self.tfencoder(src, mask=None, src_key_padding_mask=src_key_padding_mask)
        return output


class TransforEncoder(nn.Module):
    def __init__(self, modality, num_patches, fea_size, dim_feedforward, nhead=4, num_layers=3):
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
        tf_last_hidden_state = self.tfencoder(inputs, src_key_padding_mask=key_padding_mask)

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
        self.enc_au = TransforEncoder(modality="au", num_patches=opt.seq_lens[0], fea_size=300, dim_feedforward=512)
        self.enc_em = TransforEncoder(modality="em", num_patches=opt.seq_lens[1], fea_size=300, dim_feedforward=512)
        self.enc_hp = TransforEncoder(modality="hp", num_patches=opt.seq_lens[2], fea_size=300, dim_feedforward=512)
        self.enc_bp = TransforEncoder(modality="bp", num_patches=opt.seq_lens[3], fea_size=300, dim_feedforward=512)

        # LSTM Encoder for learning sequential features

    def forward(self, au, em, hp, bp, mask):
        hidden_au = self.enc_au(au, mask['au'])
        hidden_em = self.enc_em(em, mask['em'])
        hidden_hp = self.enc_hp(hp, mask['hp'])
        hidden_bp = self.enc_bp(bp, mask['bp'])

        return {'au': hidden_au, 'em': hidden_em, 'hp': hidden_hp, 'bp': hidden_bp}
