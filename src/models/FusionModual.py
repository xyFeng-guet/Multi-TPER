import copy
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(FeedForward, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)
        return self.net(x)


class MultiHAtten(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super(MultiHAtten, self).__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        b, n, _, h = *q.shape, self.heads
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class CrossTransformer(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.):
        super(CrossTransformer, self).__init__()
        self.cross_attn = MultiHAtten(dim, heads=8, dim_head=64, dropout=dropout)
        self.layernorm1 = nn.LayerNorm(dim)
        self.layernorm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, source_x, target_x):
        target_x_tmp = self.cross_attn(target_x, source_x, source_x)
        target_x = self.layernorm1(target_x_tmp + target_x)
        target_x = self.layernorm2(self.ffn(target_x) + target_x)
        return target_x


class DyRout_block(nn.Module):
    def __init__(self, opt, dropout):
        super(DyRout_block, self).__init__()
        self.f_au = CrossTransformer(dim=opt.hidden_size, mlp_dim=opt.ffn_size, dropout=dropout)
        self.f_em = CrossTransformer(dim=opt.hidden_size, mlp_dim=opt.ffn_size, dropout=dropout)
        self.f_hp = CrossTransformer(dim=opt.hidden_size, mlp_dim=opt.ffn_size, dropout=dropout)
        self.f_bp = CrossTransformer(dim=opt.hidden_size, mlp_dim=opt.ffn_size, dropout=dropout)

        self.layernorm_au = nn.LayerNorm(512)
        self.layernorm_em = nn.LayerNorm(512)
        self.layernorm_hp = nn.LayerNorm(512)
        self.layernorm_bp = nn.LayerNorm(512)

    def forward(self, source, au, em, hp, bp):
        cross_f_au = self.f_au(target_x=source, source_x=au)
        cross_f_em = self.f_em(target_x=source, source_x=em)
        cross_f_hp = self.f_hp(target_x=source, source_x=hp)
        cross_f_bp = self.f_bp(target_x=source, source_x=bp)

        output = self.layernorm_au(cross_f_au) + self.layernorm_em(cross_f_em) + self.layernorm_hp(cross_f_hp) + self.layernorm_bp(cross_f_bp)

        return output


class DyRoutTrans_block(nn.Module):
    def __init__(self, opt):
        super(DyRoutTrans_block, self).__init__()
        self.mhatt1 = DyRout_block(opt, dropout=0.)
        self.mhatt2 = MultiHAtten(opt.hidden_size, dropout=0.)
        self.ffn = FeedForward(opt.hidden_size, opt.ffn_size, dropout=0.)

        self.norm1 = nn.LayerNorm(opt.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(opt.hidden_size, eps=1e-6)
        self.norm3 = nn.LayerNorm(opt.hidden_size, eps=1e-6)

    def forward(self, source, au, em, hp, bp, mask):
        source = self.norm1(source + self.mhatt1(source, au, em, hp, bp))
        source = self.norm2(source + self.mhatt2(q=source, k=source, v=source))
        source = self.norm3(source + self.ffn(source))
        return source


class MultimodalFusion(nn.Module):
    def __init__(self, opt):
        super(MultimodalFusion, self).__init__()
        self.opt = opt

        # Length Align
        self.len_au = nn.Linear(opt.seq_lens[0], 8)
        self.len_em = nn.Linear(opt.seq_lens[1], 8)
        self.len_hp = nn.Linear(opt.seq_lens[2], 8)
        self.len_bp = nn.Linear(opt.seq_lens[3], 8)

        # Dimension Align
        self.dim_au = nn.Linear(512, 512)
        self.dim_em = nn.Linear(512, 512)
        self.dim_hp = nn.Linear(512, 512)
        self.dim_bp = nn.Linear(512, 512)

        fusion_block = DyRoutTrans_block(opt)
        self.dec_list = self._get_clones(fusion_block, 4)

    def forward(self, uni_fea, uni_mask=None):
        hidden_au = self.len_au(self.dim_au(uni_fea['au']).permute(0, 2, 1)).permute(0, 2, 1)
        hidden_em = self.len_em(self.dim_em(uni_fea['em']).permute(0, 2, 1)).permute(0, 2, 1)
        hidden_hp = self.len_hp(self.dim_hp(uni_fea['hp']).permute(0, 2, 1)).permute(0, 2, 1)
        hidden_bp = self.len_bp(self.dim_bp(uni_fea['bp']).permute(0, 2, 1)).permute(0, 2, 1)

        source = hidden_au + hidden_em + hidden_hp + hidden_bp
        for i, dec in enumerate(self.dec_list):
            source = dec(source, hidden_au, hidden_em, hidden_hp, hidden_bp, uni_mask)

        return source

    def _get_clones(self, module, N):
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
