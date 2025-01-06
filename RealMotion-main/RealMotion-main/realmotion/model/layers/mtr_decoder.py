import copy
import math
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerDecoder(nn.Module):
    def __init__(self,
                 embed_dim,
                 future_steps,
                 num_decoder_layers=6,
                 num_heads=8,
                 dropout=0.1,                 
                 ):
        super().__init__()
        in_channels = embed_dim
        self.future_steps = future_steps
        self.embed_dim = embed_dim
        self.num_decoder_layers = num_decoder_layers

        self.future_traj_mlps = build_mlps(
            c_in=2 * future_steps, mlp_channels=[embed_dim, embed_dim], ret_before_act=True, without_norm=True
        )
        self.traj_fusion_mlps = build_mlps(
            c_in=embed_dim * 2, mlp_channels=[embed_dim, embed_dim], ret_before_act=True, without_norm=True
        )

        decoder_layer = TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4, dropout=dropout,
            activation="relu",
        )
        self.decoder_layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(num_decoder_layers)])
        self.decoder_layers[0].first_layer = True

        self.motion_reg_head = build_mlps(
            c_in=in_channels,
            mlp_channels=[embed_dim, embed_dim, self.future_steps * 2], ret_before_act=True
        )
        self.motion_cls_head = build_mlps(
            c_in=in_channels,
            mlp_channels=[embed_dim, embed_dim, 1], ret_before_act=True
        )

        # self.motion_reg_heads = nn.ModuleList([copy.deepcopy(motion_reg_head) for _ in range(num_decoder_layers)])
        # self.motion_cls_heads = nn.ModuleList([copy.deepcopy(motion_cls_head) for _ in range(num_decoder_layers)])

    def forward(self, query_feat, init_traj, x, x_pos, valid_mask):
        B, num_modes = init_traj.size(0), init_traj.size(1)

        query_pos_embed = gen_sineembed_for_position(init_traj[:, :, -1], self.embed_dim)
        key_pos_embed = gen_sineembed_for_position(x_pos, self.embed_dim)
        motion_query = query_feat
        pred_trajs = init_traj
        for layer_idx in range(self.num_decoder_layers):
            # query object feature
            motion_query = self.decoder_layers[layer_idx](motion_query, x, query_pos_embed, key_pos_embed, key_padding_mask=~valid_mask)
        # motion prediction
        pred_feat = motion_query.reshape(-1, self.embed_dim)
        pred_scores = self.motion_cls_head(pred_feat).view(B, num_modes)
        pred_trajs = self.motion_reg_head(pred_feat).view(B, num_modes, self.future_steps, 2) + pred_trajs.detach()

        return pred_trajs, pred_scores


def build_mlps(c_in, mlp_channels=None, ret_before_act=False, without_norm=False):
    layers = []
    num_layers = len(mlp_channels)

    for k in range(num_layers):
        if k + 1 == num_layers and ret_before_act:
            layers.append(nn.Linear(c_in, mlp_channels[k], bias=True))
        else:
            if without_norm:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=True), nn.ReLU()]) 
            else:
                layers.extend([nn.Linear(c_in, mlp_channels[k], bias=False), nn.BatchNorm1d(mlp_channels[k]), nn.ReLU()])
            c_in = mlp_channels[k]

    return nn.Sequential(*layers)


def gen_sineembed_for_position(pos_tensor, hidden_dim=256):
    """Mostly copy-paste from https://github.com/IDEA-opensource/DAB-DETR/
    """
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    half_hidden_dim = hidden_dim // 2
    scale = 2 * math.pi
    dim_t = torch.arange(half_hidden_dim, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / half_hidden_dim)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos



class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", first_layer=False):
        super().__init__()
        self.first_layer = first_layer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        def _get_activation_fn(activation):
            """Return an activation function given a string"""
            if activation == "relu":
                return F.relu
            if activation == "gelu":
                return F.gelu
            if activation == "glu":
                return F.glu
            raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, query, key, query_pos_embed=None, key_pos_embed=None, attn_mask=None, key_padding_mask=None):

        q = k = self.with_pos_embed(query, query_pos_embed)
        v = self.with_pos_embed(query, query_pos_embed) if self.first_layer else query
        query2 = self.self_attn(q, k, value=v)[0]
        query = query + self.dropout1(query2)
        query = self.norm1(query)

        query2 = self.multihead_attn(query=self.with_pos_embed(query, query_pos_embed),
                                     key=self.with_pos_embed(key, key_pos_embed),
                                     value=self.with_pos_embed(key, key_pos_embed) if self.first_layer else key,
                                     attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        return query