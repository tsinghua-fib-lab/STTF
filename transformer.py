import json
import pdb
import time
import pandas as pd
import dgl
import torch
import torch.nn as nn
import math
from math import sqrt
import copy
from dgl.nn.pytorch import RelGraphConv
from typing import Optional, Any, Union, Callable
from torch import Tensor
import torch.nn.functional as F
import einops
import networkx as nx
from torch.utils.checkpoint import checkpoint
import numpy as np
import numpy.ma as ma

# device = torch.device('cuda' if torch.cuda.is_available( ) else 'cpu')
# device = torch.device('cpu')


class PositionalEncoder(nn.Module):
    def __init__(self, dropout = 0.1, max_seq_len = 100, d_model = 32, batch_first = True):
        super( ).__init__( )
        self.dropout = nn.Dropout(p = dropout)
        self.max_seq_len = max_seq_len
        self.batch_first = batch_first
        self.d_model = d_model

        self.x_dim = 1 if batch_first else 0
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print(x.shape)
        # print(self.pe.shape)
        x = x + self.pe.squeeze(1)[:x.size(self.x_dim)]

        return self.dropout(x)


class DSW_embedding(nn.Module):
    def __init__(self, seg_len, d_model):
        super(DSW_embedding, self).__init__( )
        self.seg_len = seg_len
        self.linear = nn.Linear(seg_len, d_model)

    def forward(self, x):
        batch, ts_len, ts_dim = x.shape

        x_segment = einops.rearrange(x, 'b (seg_num seg_len) d -> (b d seg_num) seg_len', seg_len = self.seg_len)

        x_embed = self.linear(x_segment)

        x_embed = einops.rearrange(x_embed, '(b d seg_num) d_model -> b d seg_num d_model', b = batch, d = ts_dim)

        return x_embed


class Transformer_vanilla(nn.Module):
    def __init__(self, input_size = 1, seq_len = 12, hidden_dim = 64, n_head = 8, encoder_num_layers = 2,
                 decoder_num_layers = 2,
                 predict_len = 1):
        super(Transformer_vanilla, self).__init__( )
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.predict_len = predict_len

        self.encoder_input_layer = nn.Linear(self.input_size, self.hidden_dim)
        self.positional_encoding_layer_encoder = PositionalEncoder(d_model = self.hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model = self.hidden_dim, nhead = n_head,
                                                   batch_first = True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers = self.encoder_num_layers, norm = None)

        self.positional_encoding_layer_decoder = PositionalEncoder(d_model = self.hidden_dim)
        self.decoder_input_layer = nn.Linear(self.input_size, self.hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model = self.hidden_dim, nhead = n_head, batch_first = True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers = self.decoder_num_layers)

        self.output_layer = nn.Linear(self.hidden_dim, self.predict_len)

        self.mask = nn.Transformer.generate_square_subsequent_mask(seq_len + predict_len)

    def forward(self, x_en):
        num_nodes = x_en.shape[1]
        encoder_len = x_en.shape[2]
        # decoder_len = x_de.shape[2]
        batch_size = x_en.shape[0]
        channel = x_en.shape[3]
        traffic_en = x_en[:, :, :, 0]
        traffic_en = traffic_en.reshape(batch_size * num_nodes, encoder_len, 1)
        x_de = torch.zeros([batch_size * num_nodes, self.seq_len + self.predict_len, 1])
        x_de[:, :self.seq_len, :] = traffic_en

        traffic_en = self.encoder_input_layer(traffic_en)
        traffic_en = self.positional_encoding_layer_encoder(traffic_en)
        traffic_en = self.encoder(traffic_en)
        # print(traffic_en.shape)
        x_de = self.decoder_input_layer(x_de)
        x_de = self.positional_encoding_layer_decoder(x_de)
        x_de = self.decoder(x_de, memory = traffic_en, tgt_mask = self.mask)
        x_de = self.output_layer(x_de)
        x_de = x_de[:, -1 * self.predict_len:, :]
        x_de = x_de.reshape(batch_size, num_nodes, self.predict_len)

        return x_de


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerEncoderLayer_cross(nn.Module):
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device = None, dtype = None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer_cross, self).__init__( )
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout = dropout, batch_first = batch_first,
                                               **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps = layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps = layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        # We can't test self.activation in forward() in TorchScript,
        # so stash some information about it instead.
        if activation is F.relu:
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu:
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerEncoderLayer_cross, self).__setstate__(state)

    def forward(self, src: Tensor, src_q: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_q, src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_q, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, x_q: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x_q, x, x,
                           attn_mask = attn_mask,
                           key_padding_mask = key_padding_mask,
                           need_weights = False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class TransformerEncoder_cross(nn.Module):
    __constants__ = ['norm']

    def __init__(self, encoder_layer, num_layers, norm = None):
        super(TransformerEncoder_cross, self).__init__( )
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: Tensor, src_q: Tensor) -> Tensor:

        for mod in self.layers:
            output = mod(src, src_q)

        if self.norm is not None:
            output = self.norm(output)

        return output


class Transformer_cross(nn.Module):
    def __init__(self, input_size = 1, seq_len = 12, encoder_hidden_dim = 32, decoder_hidden_dim = 64, predict_len = 1,
                 n_head = 8, encoder_num_layers = 2, decoder_num_layers = 2, ):
        super(Transformer_cross, self).__init__( )
        self.input_size = input_size
        self.seq_len = seq_len
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.predict_len = predict_len
        self.n_head = n_head
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers

        self.encoder_input_layer_traffic = nn.Linear(self.input_size, self.encoder_hidden_dim)
        self.positional_encoding_layer_traffic = PositionalEncoder(d_model = self.encoder_hidden_dim)
        encoder_layer_traffic = TransformerEncoderLayer_cross(d_model = self.encoder_hidden_dim, nhead = self.n_head,
                                                              batch_first = True)
        self.encoder_traffic = TransformerEncoder_cross(encoder_layer_traffic, num_layers = self.encoder_num_layers)

        self.encoder_input_layer_user = nn.Linear(self.input_size, self.encoder_hidden_dim)
        self.positional_encoding_layer_user = PositionalEncoder(d_model = self.encoder_hidden_dim)
        encoder_layer_user = TransformerEncoderLayer_cross(d_model = self.encoder_hidden_dim, nhead = self.n_head,
                                                           batch_first = True)
        self.encoder_user = TransformerEncoder_cross(encoder_layer_user, num_layers = self.encoder_num_layers)

        self.decoder_input_layer = nn.Linear(self.input_size, self.decoder_hidden_dim)
        self.positional_encoding_layer_decoder = PositionalEncoder(d_model = self.decoder_hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model = self.decoder_hidden_dim, nhead = n_head,
                                                   batch_first = True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers = self.decoder_num_layers)

        self.output_layer = nn.Linear(self.decoder_hidden_dim, self.predict_len)

        self.mask = nn.Transformer.generate_square_subsequent_mask(seq_len + predict_len)

    def forward(self, x):
        num_nodes = x.shape[1]
        seq_len = x.shape[2]
        batch_size = x.shape[0]
        traffic = x[:, :, :, 0]
        user = x[:, :, :, 1]
        traffic = traffic.reshape(batch_size * num_nodes, seq_len, 1)
        user = user.reshape(batch_size * num_nodes, seq_len, 1)

        x_de = torch.zeros([batch_size * num_nodes, self.seq_len + self.predict_len, 1])
        x_de[:, :self.seq_len, :] = traffic

        traffic = self.encoder_input_layer_traffic(traffic)
        traffic = self.positional_encoding_layer_traffic(traffic)

        user = self.encoder_input_layer_user(user)
        user = self.positional_encoding_layer_user(user)

        traffic_en = self.encoder_traffic(traffic, user)
        user_en = self.encoder_user(user, traffic)

        x_en = torch.concat([traffic_en, user_en], 2)

        x_de = self.decoder_input_layer(x_de)
        x_de = self.positional_encoding_layer_decoder(x_de)
        # print(x_de.shape,x_en.shape)
        x_de = self.decoder(x_de, memory = x_en, tgt_mask = self.mask)
        x_de = self.output_layer(x_de)
        x_de = x_de[:, -1 * self.predict_len:, :]
        x_de = x_de.reshape(batch_size, num_nodes, self.predict_len)
        # print(x_de.shape)
        return x_de


class LinearAttentionHead(nn.Module):
    """
    Linear attention, as proposed by the linformer paper
    """

    def __init__(self, attention_bias = None, dim = 64, dropout = 0.2, causal_mask = None, ):
        super(LinearAttentionHead, self).__init__( )
        self.dim = dim
        self.dropout = nn.Dropout(dropout)
        self.P_bar = None
        self.causal_mask = causal_mask
        self.atten_bias = attention_bias

    def forward(self, Q, K, V, **kwargs):
        """
        Assume Q, K, V have same dtype
        E, F are `nn.Linear` modules
        """

        input_mask = kwargs["input_mask"] if "input_mask" in kwargs else None
        embeddings_mask = kwargs["embeddings_mask"] if "embeddings_mask" in kwargs else None

        # Instead of classic masking, we have to do this, because the classic mask is of size nxn
        if input_mask is not None:
            # This is for k, v
            mask = input_mask[:, :, None]
            K = K.masked_fill_(~mask, 0.0)
            V = V.masked_fill_(~mask, 0.0)
            del mask

        if embeddings_mask is not None:
            mask = embeddings_mask[:, :, None]
            Q = Q.masked_fill_(~mask, 0.0)
            del mask

        K = K.transpose(1, 2)

        Q = torch.matmul(Q, K)

        P_bar = Q / torch.sqrt(torch.tensor(self.dim).type(Q.type( ))).to(Q.device)
        if self.causal_mask is not None:
            self.causal_mask = self.causal_mask.to(Q.device)
            P_bar = P_bar.masked_fill_(~self.causal_mask, float('-inf'))
        if self.atten_bias is not None:
            for i in range(P_bar.shape[0]):
                # print(P_bar.shape,self.atten_bias['dis'].shape)
                P_bar[i] = P_bar[i] + self.atten_bias['dis'] + self.atten_bias['dtw'] + self.atten_bias['pattern'] + \
                           self.atten_bias['poi']
        # print('add',P_bar.shape)
        P_bar = P_bar.softmax(dim = -1)
        attn_map = P_bar
        # print(attn_map.shape)
        # Only save this when visualizing
        if "visualize" in kwargs and kwargs["visualize"] == True:
            self.P_bar = P_bar

        P_bar = self.dropout(P_bar)
        out_tensor = torch.matmul(P_bar, V)

        return out_tensor

    def change_attention_bias(self, atten_bias):
        self.atten_bias = atten_bias


class spatial_attn(nn.Module):

    def __init__(self, attention_bias = None, num_nodes = 64, dim = 8, channels = 64, dim_k = 128, nhead = 8,
                 dropout = 0.2, checkpoint_level = 'C2', causal_mask = None, w_o_intermediate_dim = None,
                 decoder_mode = False, ):
        super(spatial_attn, self).__init__( )
        self.heads = nn.ModuleList( )
        self.num_nodes = num_nodes
        self.dim_k = dim_k
        self.channels = channels
        self.causal_mask = causal_mask
        self.checkpoint_level = checkpoint_level
        self.w_o_intermediate_dim = w_o_intermediate_dim
        self.attention_bias = attention_bias
        self.decoder_mode = decoder_mode
        self.to_q = nn.ModuleList( )
        self.to_k = nn.ModuleList( )
        self.to_v = nn.ModuleList( )

        for _ in range(nhead):
            attn = LinearAttentionHead(attention_bias = attention_bias)
            self.heads.append(attn)
            self.to_q.append(nn.Linear(channels, dim, bias = False))
            self.to_k.append(nn.Linear(channels, dim, bias = False))
            self.to_v.append(nn.Linear(channels, dim, bias = False))
        if w_o_intermediate_dim is None:
            self.w_o = nn.Linear(dim * nhead, channels)
        else:
            self.w_o_1 = nn.Linear(dim * nhead, w_o_intermediate_dim)
            self.w_o_2 = nn.Linear(w_o_intermediate_dim, channels)
        self.mh_dropout = nn.Dropout(dropout)

    def forward(self, tensor, **kwargs):
        batch_size, input_len, channels = tensor.shape
        assert not (self.decoder_mode and "embeddings" not in kwargs), "Embeddings must be supplied if decoding"
        assert not ("embeddings" in kwargs and (
            kwargs["embeddings"].shape[0], kwargs["embeddings"].shape[1], kwargs["embeddings"].shape[2]) != (
                        batch_size, input_len, channels)), "Embeddings size must be the same as the input tensor"
        head_outputs = []
        for index, head in enumerate(self.heads):
            Q = self.to_q[index](tensor)
            K = self.to_k[index](tensor) if not self.decoder_mode else self.to_k[index](kwargs["embeddings"])
            V = self.to_v[index](tensor) if not self.decoder_mode else self.to_v[index](kwargs["embeddings"])
            if self.checkpoint_level == "C2":
                head_outputs.append(checkpoint(head, Q, K, V))
            else:
                head_outputs.append(head(Q, K, V, **kwargs))
        out = torch.cat(head_outputs, dim = -1)
        if self.w_o_intermediate_dim is None:
            out = self.w_o(out)
        else:
            out = self.w_o_1(out)
            out = self.w_o_2(out)
        out = self.mh_dropout(out)
        return out

    def change_attention_bias(self, attention_bias):
        self.attention_bias = attention_bias
        for head in self.heads:
            head.change_attention_bias(attention_bias)


class SelfAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v):
        super(SelfAttention, self).__init__( )
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v

        # 定义线性变换函数
        self.linear_q = nn.Linear(dim_q, dim_k, bias = False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias = False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias = False)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x):
        # x: batch, n, dim_q
        # 根据文本获得相应的维度

        batch, n, dim_q = x.shape
        assert dim_q == self.dim_q

        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v
        # q*k的转置 并*开根号后的dk
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        # 归一化获得attention的相关系数
        dist = torch.softmax(dist, dim = -1)  # batch, n, n
        # attention系数和v相乘，获得最终的得分
        att = torch.bmm(dist, v)
        return att


class CausalConv1d(torch.nn.Conv1d):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride = 1,
                 dilation = 1,
                 groups = 1,
                 bias = True):
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = 0,
            dilation = dilation,
            groups = groups,
            bias = bias)

        self.__padding = (kernel_size - 1) * dilation

    def forward(self, input):
        return super(CausalConv1d, self).forward(F.pad(input, (self.__padding, 0)))


class context_embedding(torch.nn.Module):
    def __init__(self, in_channels = 1, embedding_size = 256, k = 5):
        super(context_embedding, self).__init__( )
        self.causal_convolution = CausalConv1d(in_channels, embedding_size, kernel_size = k)

    def forward(self, x):
        x = self.causal_convolution(x)
        return torch.tanh(x)


class GraphAttention_vanilla(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, attention_dropout = 0.3):
        super(GraphAttention, self).__init__( )
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.dropout = nn.Dropout(attention_dropout)

        # 定义线性变换函数
        self.linear_q = nn.Linear(dim_q, dim_k, bias = False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias = False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias = False)
        self._norm_fact = 1 / sqrt(dim_k)

    def forward(self, x, bias = None):
        # x: batch, n, dim_q
        # 根据文本获得相应的维度

        batch, n, dim_q = x.shape
        assert dim_q == self.dim_q

        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v
        # q*k的转置 并*开根号后的dk
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        # 归一化获得attention的相关系数
        if bias is not None:
            dist = dist + bias
        dist = torch.softmax(dist, dim = -1)  # batch, n, n
        dist = self.dropout(dist)
        # attention系数和v相乘，获得最终的得分
        att = torch.bmm(dist, v)
        return att


class Graph_attn(nn.Module):  # 一跳邻居：10/20，二跳邻居：50
    def __init__(self, hidden_dim = 16, dim_k = 32, attention_bias = None, num_max_degree = 100, encoder_layers = 1,
                 attention_dropout = 0.3,
                 num_hop = 2, degree_nodes = None, connect_nodes = None):
        super(Graph_attn, self).__init__( )
        self.hidden_dim = hidden_dim
        self.dim_k = dim_k
        self.attention_bias = attention_bias
        self.num_max_degree = num_max_degree
        self.encoder_layer = encoder_layers
        self.attention_dropout = attention_dropout
        self.num_hop = num_hop
        self.degree_nodes = degree_nodes
        self.connect_nodes = connect_nodes
        if self.degree_nodes is not None:
            self.degree_embedding = nn.Embedding(int(torch.max(degree_nodes)) + 1, 4)
            self.hidden_dim += 4

        self.dis_emb = nn.Embedding(4, 1)
        self.degree_emb_hop1 = nn.Embedding(10, 1)
        self.degree_emb_hop2 = nn.Embedding(50, 1)
        self.graph_att = GraphAttention(self.hidden_dim, dim_k, hidden_dim)

    def forward(self, x):  # x: batch, num_nodes,embedding   attention_bias: batch, n, n,
        if self.degree_nodes is not None:
            node_spatial_embedding = self.degree_embedding(self.degree_nodes)
            x = torch.cat([x, node_spatial_embedding], dim = 1)
        extra_nodes = torch.zeros(1, x.shape[1]).to(x.device)
        x = torch.cat([x, extra_nodes], dim = 0)
        graph_embedding = torch.index_select(x, dim = 0, index = self.connect_nodes)
        graph_embedding = graph_embedding.reshape(x.shape[0] - 1, 51, -1)
        attention_bias = self.dis_emb(self.attention_bias)
        attention_bias = attention_bias.reshape(x.shape[0] - 1, 51, -1)
        x_output = self.graph_att(graph_embedding, attention_bias)
        x_output = x_output[:, 0, :]
        return x_output


class sstt_rgcn(nn.Module):
    def __init__(self, g_base, etype_base, input_size = 1, seq_len = 12, seg_len = 3, encoder_hidden_dim = 64,
                 decoder_hidden_dim = 64, predict_len = 1, dim_k = 32,
                 n_head = 8, encoder_num_layers = 2, decoder_num_layers = 2, attention_bias = None,
                 layer_norm_eps: float = 1e-5,
                 dim_feedforward = 2048, ff_dropout = 0.5):
        super(sstt_rgcn, self).__init__( )
        self.input_size = input_size
        self.seq_len = seq_len
        self.seg_len = seg_len
        self.patch_num = int(seq_len / seg_len)
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.predict_len = predict_len
        self.dim_k = dim_k
        self.n_head = n_head
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.attention_bias = attention_bias
        self.g_base = g_base
        self.etype_base = etype_base

        self.patch_traffic = DSW_embedding(seg_len, self.encoder_hidden_dim)
        self.positional_encoding_layer_traffic = PositionalEncoder(d_model = self.encoder_hidden_dim)
        self.encoder_traffic_temporal = nn.ModuleList( )
        for i in range(self.encoder_num_layers):
            self.encoder_traffic_temporal.append(TransformerEncoder_cross(
                TransformerEncoderLayer_cross(d_model = self.encoder_hidden_dim, nhead = self.n_head,
                                              batch_first = True), num_layers = 1))

        self.encoder_traffic_spatial = nn.ModuleList( )
        for i in range(self.encoder_num_layers):
            self.encoder_traffic_spatial.append(
                RelGraphConv(self.encoder_hidden_dim, self.encoder_hidden_dim, num_rels = 4, regularizer = 'basis',
                             num_bases = 4))

        self.patch_user = DSW_embedding(seg_len, self.encoder_hidden_dim)
        self.positional_encoding_layer_user = PositionalEncoder(d_model = self.encoder_hidden_dim)
        self.encoder_user_temporal = nn.ModuleList( )
        for i in range(self.encoder_num_layers):
            self.encoder_user_temporal.append(TransformerEncoder_cross(
                TransformerEncoderLayer_cross(d_model = self.encoder_hidden_dim, nhead = self.n_head,
                                              batch_first = True), num_layers = 1))

        self.encoder_user_spatial = nn.ModuleList( )
        for i in range(self.encoder_num_layers):
            self.encoder_user_spatial.append(
                RelGraphConv(self.encoder_hidden_dim, self.encoder_hidden_dim, num_rels = 4, regularizer = 'basis',
                             num_bases = 4))

        self.encoder_attn = SelfAttention(2 * self.encoder_hidden_dim, dim_k, self.decoder_hidden_dim)

        self.patch_decoder = DSW_embedding(seg_len, self.decoder_hidden_dim)
        self.positional_encoding_layer_decoder = PositionalEncoder(d_model = self.decoder_hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model = self.decoder_hidden_dim, nhead = n_head,
                                                   batch_first = True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers = self.decoder_num_layers)

        self.output_layer_traffic = nn.Linear(self.decoder_hidden_dim, self.predict_len)
        self.output_layer_user = nn.Linear(self.decoder_hidden_dim, self.predict_len)

    def forward(self, x):
        num_nodes = x.shape[1]
        seq_len = x.shape[2]
        batch_size = x.shape[0]
        traffic = x[:, :, :, 0]
        user = x[:, :, :, 1]
        traffic = traffic.reshape(batch_size * num_nodes, seq_len, 1)
        user = user.reshape(batch_size * num_nodes, seq_len, 1)
        x_de_traffic = torch.zeros([batch_size * num_nodes, self.seq_len, 1]).to(x.device)
        x_de_traffic[:, :-1 * self.predict_len, :] = traffic[:, self.predict_len:, :]
        x_de_user = torch.zeros([batch_size * num_nodes, self.seq_len, 1]).to(x.device)
        x_de_user[:, :-1 * self.predict_len, :] = user[:, self.predict_len:, :]

        self.mask = nn.Transformer.generate_square_subsequent_mask(self.patch_num).to(x.device)

        traffic = self.patch_traffic(traffic).squeeze( )
        traffic = self.positional_encoding_layer_traffic(traffic)
        user = self.patch_user(user).squeeze( )
        user = self.positional_encoding_layer_user(user)

        for i in range(self.encoder_num_layers):
            traffic = self.encoder_traffic_temporal[i](traffic, user)
            user = self.encoder_user_temporal[i](user, traffic)

        traffic = traffic.permute(1, 0, 2).reshape(self.patch_num * batch_size, num_nodes, -1)
        user = user.permute(1, 0, 2).reshape(self.patch_num * batch_size, num_nodes, -1)
        traffic_en = torch.zeros(traffic.shape).to(traffic.device)
        user_en = torch.zeros(user.shape).to(user.device)
        for j in range(self.encoder_num_layers):
            for i in range(traffic_en.shape[0]):
                traffic_en[i] = self.encoder_traffic_spatial[j](self.g_base, traffic[i], self.etype_base)
                user_en[i] = self.encoder_user_spatial[j](self.g_base, user[i], self.etype_base)

        traffic_en = traffic_en.reshape(self.patch_num, batch_size, num_nodes, -1)
        user_en = user_en.reshape(self.patch_num, batch_size, num_nodes, -1)
        traffic_en = traffic_en.permute(1, 2, 0, 3)
        user_en = user_en.permute(1, 2, 0, 3)
        traffic_en = traffic_en.reshape(batch_size * num_nodes, self.patch_num, -1)
        user_en = user_en.reshape(batch_size * num_nodes, self.patch_num, -1)

        x_en = torch.concat([traffic_en, user_en], 2)
        x_en = self.encoder_attn(x_en)

        x_de_traffic = self.patch_decoder(x_de_traffic).squeeze( )
        x_de_traffic = self.positional_encoding_layer_decoder(x_de_traffic)
        x_de_traffic = self.decoder(x_de_traffic, memory = x_en, tgt_mask = self.mask)
        x_de_traffic = x_de_traffic[:, -1 * self.predict_len:, :]
        x_de_traffic = self.output_layer_traffic(x_de_traffic)
        x_de_traffic = x_de_traffic.reshape(batch_size, num_nodes, self.predict_len)

        x_de_user = self.patch_decoder(x_de_user).squeeze( )
        x_de_user = self.positional_encoding_layer_decoder(x_de_user)
        x_de_user = self.decoder(x_de_user, memory = x_en, tgt_mask = self.mask)
        x_de_user = x_de_user[:, -1 * self.predict_len:, :]
        x_de_user = self.output_layer_traffic(x_de_user)
        x_de_user = x_de_user.reshape(batch_size, num_nodes, self.predict_len)

        x_de = torch.cat([x_de_traffic, x_de_user], dim = -1)
        return x_de


class stst_rgcn(nn.Module):
    def __init__(self, g_base, etype_base, input_size = 1, seq_len = 12, seg_len = 3, encoder_hidden_dim = 64,
                 decoder_hidden_dim = 64, predict_len = 1, dim_k = 32,
                 n_head = 8, encoder_num_layers = 2, decoder_num_layers = 2, attention_bias = None,
                 layer_norm_eps: float = 1e-5,
                 dim_feedforward = 2048, ff_dropout = 0.5):
        super(stst_rgcn, self).__init__( )
        self.input_size = input_size
        self.seq_len = seq_len
        self.seg_len = seg_len
        self.patch_num = int(seq_len / seg_len)
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.predict_len = predict_len
        self.dim_k = dim_k
        self.n_head = n_head
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.attention_bias = attention_bias
        self.g_base = g_base
        self.etype_base = etype_base

        self.patch_traffic = DSW_embedding(seg_len, self.encoder_hidden_dim)
        self.positional_encoding_layer_traffic = PositionalEncoder(d_model = self.encoder_hidden_dim)
        self.encoder_traffic_temporal = nn.ModuleList( )
        for i in range(self.encoder_num_layers):
            self.encoder_traffic_temporal.append(TransformerEncoder_cross(
                TransformerEncoderLayer_cross(d_model = self.encoder_hidden_dim, nhead = self.n_head,
                                              batch_first = True), num_layers = 1))

        self.encoder_traffic_spatial = nn.ModuleList( )
        for i in range(self.encoder_num_layers):
            self.encoder_traffic_spatial.append(
                RelGraphConv(self.encoder_hidden_dim, self.encoder_hidden_dim, num_rels = 4, regularizer = 'basis',
                             num_bases = 4))

        self.patch_user = DSW_embedding(seg_len, self.encoder_hidden_dim)
        self.positional_encoding_layer_user = PositionalEncoder(d_model = self.encoder_hidden_dim)
        self.encoder_user_temporal = nn.ModuleList( )
        for i in range(self.encoder_num_layers):
            self.encoder_user_temporal.append(TransformerEncoder_cross(
                TransformerEncoderLayer_cross(d_model = self.encoder_hidden_dim, nhead = self.n_head,
                                              batch_first = True), num_layers = 1))

        self.encoder_user_spatial = nn.ModuleList( )
        for i in range(self.encoder_num_layers):
            self.encoder_user_spatial.append(
                RelGraphConv(self.encoder_hidden_dim, self.encoder_hidden_dim, num_rels = 4, regularizer = 'basis',
                             num_bases = 4))

        self.encoder_attn = SelfAttention(2 * self.encoder_hidden_dim, dim_k, self.decoder_hidden_dim)

        self.patch_decoder = DSW_embedding(seg_len, self.decoder_hidden_dim)
        self.positional_encoding_layer_decoder = PositionalEncoder(d_model = self.decoder_hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model = self.decoder_hidden_dim, nhead = n_head,
                                                   batch_first = True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers = self.decoder_num_layers)

        self.output_layer_traffic = nn.Linear(self.decoder_hidden_dim, self.predict_len)
        self.output_layer_user = nn.Linear(self.decoder_hidden_dim, self.predict_len)

    def forward(self, x):
        num_nodes = x.shape[1]
        seq_len = x.shape[2]
        batch_size = x.shape[0]
        traffic = x[:, :, :, 0]
        user = x[:, :, :, 1]
        traffic = traffic.reshape(batch_size * num_nodes, seq_len, 1)
        user = user.reshape(batch_size * num_nodes, seq_len, 1)
        x_de_traffic = torch.zeros([batch_size * num_nodes, self.seq_len, 1]).to(x.device)
        x_de_traffic[:, :-1 * self.predict_len, :] = traffic[:, self.predict_len:, :]
        x_de_user = torch.zeros([batch_size * num_nodes, self.seq_len, 1]).to(x.device)
        x_de_user[:, :-1 * self.predict_len, :] = user[:, self.predict_len:, :]

        self.mask = nn.Transformer.generate_square_subsequent_mask(self.patch_num).to(x.device)

        traffic = self.patch_traffic(traffic).squeeze( )
        traffic = self.positional_encoding_layer_traffic(traffic)
        user = self.patch_user(user).squeeze( )
        user = self.positional_encoding_layer_user(user)

        traffic_replace, user_replace = traffic, user
        traffic = self.encoder_traffic_temporal[0](traffic_replace, user_replace)
        user = self.encoder_user_temporal[0](user_replace, traffic_replace)
        traffic = traffic.permute(1, 0, 2).reshape(self.patch_num * batch_size, num_nodes, -1)
        user = user.permute(1, 0, 2).reshape(self.patch_num * batch_size, num_nodes, -1)
        traffic_en = torch.zeros(traffic.shape).to(traffic.device)
        user_en = torch.zeros(user.shape).to(user.device)
        for j in range(traffic.shape[0]):
            traffic_en[j] = self.encoder_traffic_spatial[0](self.g_base, traffic[j], self.etype_base)
            user_en[j] = self.encoder_user_spatial[0](self.g_base, user[j], self.etype_base)
        traffic_en = traffic_en.permute(1, 0, 2)
        user_en = user_en.permute(1, 0, 2)

        # i==1
        traffic_replace, user_replace = traffic_en, user_en
        traffic = self.encoder_traffic_temporal[0](traffic_replace, user_replace)
        user = self.encoder_user_temporal[0](user_replace, traffic_replace)
        traffic = traffic.permute(1, 0, 2).reshape(self.patch_num * batch_size, num_nodes, -1)
        user = user.permute(1, 0, 2).reshape(self.patch_num * batch_size, num_nodes, -1)
        traffic_en = torch.zeros(traffic.shape).to(traffic.device)
        user_en = torch.zeros(user.shape).to(user.device)
        for j in range(traffic.shape[0]):
            traffic_en[j] = self.encoder_traffic_spatial[0](self.g_base, traffic[j], self.etype_base)
            user_en[j] = self.encoder_user_spatial[0](self.g_base, user[j], self.etype_base)

        traffic_en = traffic_en.reshape(self.patch_num, batch_size, num_nodes, -1)
        user_en = user_en.reshape(self.patch_num, batch_size, num_nodes, -1)
        traffic_en = traffic_en.permute(1, 2, 0, 3)
        user_en = user_en.permute(1, 2, 0, 3)
        traffic_en = traffic_en.reshape(batch_size * num_nodes, self.patch_num, -1)
        user_en = user_en.reshape(batch_size * num_nodes, self.patch_num, -1)

        x_en = torch.concat([traffic_en, user_en], 2)
        x_en = self.encoder_attn(x_en)

        x_de_traffic = self.patch_decoder(x_de_traffic).squeeze( )
        x_de_traffic = self.positional_encoding_layer_decoder(x_de_traffic)
        x_de_traffic = self.decoder(x_de_traffic, memory = x_en, tgt_mask = self.mask)
        x_de_traffic = x_de_traffic[:, -1 * self.predict_len:, :]
        x_de_traffic = self.output_layer_traffic(x_de_traffic)
        x_de_traffic = x_de_traffic.reshape(batch_size, num_nodes, self.predict_len)

        x_de_user = self.patch_decoder(x_de_user).squeeze( )
        x_de_user = self.positional_encoding_layer_decoder(x_de_user)
        x_de_user = self.decoder(x_de_user, memory = x_en, tgt_mask = self.mask)
        x_de_user = x_de_user[:, -1 * self.predict_len:, :]
        x_de_user = self.output_layer_traffic(x_de_user)
        x_de_user = x_de_user.reshape(batch_size, num_nodes, self.predict_len)

        x_de = torch.cat([x_de_traffic, x_de_user], dim = -1)
        return x_de


class GraphAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v, attention_dropout = 0.3, dim_feedforward = 2048,
                 layer_norm_eps: float = 1e-5, dropout = 0.5):
        super(GraphAttention, self).__init__( )
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.att_dropout = nn.Dropout(attention_dropout)

        # 定义线性变换函数
        self.linear_q = nn.Linear(dim_q, dim_k, bias = False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias = False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias = False)
        self._norm_fact = 1 / sqrt(dim_k)

        self.linear1 = nn.Linear(dim_v, dim_feedforward, )
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_v, )
        self.norm1 = nn.LayerNorm(dim_v, eps = layer_norm_eps, )
        self.norm2 = nn.LayerNorm(dim_v, eps = layer_norm_eps, )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, bias = None,mask = None):
        # x: batch, n, dim_q
        # 根据文本获得相应的维度

        batch, n, dim_q = x.shape
        # print(x.shape)
        assert dim_q == self.dim_q

        q = self.linear_q(x)  # batch, n, dim_k
        k = self.linear_k(x)  # batch, n, dim_k
        v = self.linear_v(x)  # batch, n, dim_v
        # q*k的转置 并*开根号后的dk
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, n, n
        # 归一化获得attention的相关系数
        if bias is not None:
            dist = dist + bias
        dist = torch.softmax(dist, dim = -1)  # batch, n, n
        if mask is not None:
            dist = dist * mask
        dist = self.att_dropout(dist)
        # attention系数和v相乘，获得最终的得分
        att = torch.bmm(dist, v)
        # print(att.shape)
        att = self.norm1(att) + x
        att = self.norm2(att + self.linear2(self.dropout(torch.relu(self.linear1(att)))))
        return att



class Cross_Graph(nn.Module):
    def __init__(self, input_size = 1, seq_len = 12, seg_len = 3, encoder_hidden_dim = 64,
                 decoder_hidden_dim = 64, predict_len = 1, dim_k = 32,
                 n_head = 8, encoder_num_layers = 2, decoder_num_layers = 2, attention_bias = None,
                 layer_norm_eps: float = 1e-5,
                 dim_feedforward = 2048, ff_dropout = 0.5):
        super(Cross_Graph, self).__init__( )
        self.input_size = input_size
        self.seq_len = seq_len
        self.seg_len = seg_len
        self.patch_num = int(seq_len / seg_len)
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.predict_len = predict_len
        self.dim_k = dim_k
        self.n_head = n_head
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.attention_bias = attention_bias

        self.patch_traffic = DSW_embedding(seg_len, self.encoder_hidden_dim)
        self.positional_encoding_layer_traffic = PositionalEncoder(d_model = self.encoder_hidden_dim)
        self.encoder_traffic_temporal = nn.ModuleList( )
        for i in range(self.encoder_num_layers):
            self.encoder_traffic_temporal.append(TransformerEncoder_cross(
                TransformerEncoderLayer_cross(d_model = self.encoder_hidden_dim, nhead = self.n_head,
                                              batch_first = True), num_layers = 1))

        self.encoder_traffic_spatial = nn.ModuleList( )
        for i in range(self.encoder_num_layers):
            self.encoder_traffic_spatial.append(
                GraphAttention(self.encoder_hidden_dim, 32, self.encoder_hidden_dim))

        self.patch_user = DSW_embedding(seg_len, self.encoder_hidden_dim)
        self.positional_encoding_layer_user = PositionalEncoder(d_model = self.encoder_hidden_dim)
        self.encoder_user_temporal = nn.ModuleList( )
        for i in range(self.encoder_num_layers):
            self.encoder_user_temporal.append(TransformerEncoder_cross(
                TransformerEncoderLayer_cross(d_model = self.encoder_hidden_dim, nhead = self.n_head,
                                              batch_first = True), num_layers = 1))

        self.encoder_user_spatial = nn.ModuleList( )
        for i in range(self.encoder_num_layers):
            self.encoder_user_spatial.append(
                GraphAttention(self.encoder_hidden_dim, 32, self.encoder_hidden_dim))

        self.encoder_attn = SelfAttention(2 * self.encoder_hidden_dim, dim_k, self.decoder_hidden_dim)

        self.patch_decoder = DSW_embedding(seg_len, self.decoder_hidden_dim)
        self.positional_encoding_layer_decoder = PositionalEncoder(d_model = self.decoder_hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model = self.decoder_hidden_dim, nhead = n_head,
                                                   batch_first = True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers = self.decoder_num_layers)

        self.output_layer_traffic = nn.Linear(self.decoder_hidden_dim, self.predict_len)
        self.output_layer_user = nn.Linear(self.decoder_hidden_dim, self.predict_len)

    def forward(self, x):
        num_neighbors = x.shape[1]
        seq_len = x.shape[2]
        num_nodes = x.shape[0]
        traffic = x[:, :, :, 0]
        user = x[:, :, :, 1]
        x_de_traffic = torch.zeros([num_nodes, self.seq_len, 1])
        x_de_traffic[:, :-1 * self.predict_len, 0] = traffic[:, 0, self.predict_len:]
        x_de_user = torch.zeros([num_nodes, self.seq_len, 1])
        x_de_user[:, :-1 * self.predict_len, 0] = user[:, 0, self.predict_len:]

        traffic = traffic.reshape(num_nodes * num_neighbors, seq_len, 1)
        user = user.reshape(num_nodes * num_neighbors, seq_len, 1)

        self.mask = nn.Transformer.generate_square_subsequent_mask(self.patch_num)

        traffic = self.patch_traffic(traffic).squeeze( )
        traffic = self.positional_encoding_layer_traffic(traffic)
        user = self.patch_user(user).squeeze( )
        user = self.positional_encoding_layer_user(user)
        # print(traffic.shape,user.shape)
        for i in range(self.encoder_num_layers):
            traffic = self.encoder_traffic_temporal[i](traffic, user)
            user = self.encoder_user_temporal[i](user, traffic)

        traffic = traffic.permute(1, 0, 2).reshape(self.patch_num * num_nodes, num_neighbors, -1)
        user = user.permute(1, 0, 2).reshape(self.patch_num * num_nodes, num_neighbors, -1)
        # print(traffic.shape)
        # for k in range(int(traffic_embedding.shape[0] / self.batch)): # [k*self.batch:k*self.batch+self.batch]
        for j in range(self.encoder_num_layers):
            traffic = self.encoder_traffic_spatial[j](traffic)
            user = self.encoder_user_spatial[j](user)
            break

        traffic_en = traffic.reshape(self.patch_num, num_nodes, num_neighbors, -1)
        user_en = user.reshape(self.patch_num, num_nodes, num_neighbors, -1)
        traffic_en = traffic_en.permute(1, 2, 0, 3)
        user_en = user_en.permute(1, 2, 0, 3)
        traffic_en = traffic_en.reshape(num_nodes, num_neighbors, self.patch_num, -1)
        user_en = user_en.reshape(num_nodes, num_neighbors, self.patch_num, -1)
        # print(traffic_en.shape)
        x_en = torch.concat([traffic_en, user_en], 3)
        x_en = x_en[:, 0, :, :, ]
        x_en = self.encoder_attn(x_en)
        # print(x_en.shape)

        x_de_traffic = self.patch_decoder(x_de_traffic).squeeze( )
        x_de_traffic = self.positional_encoding_layer_decoder(x_de_traffic)
        x_de_traffic = self.decoder(x_de_traffic, memory = x_en, tgt_mask = self.mask)
        x_de_traffic = x_de_traffic[:, -1 * self.predict_len:, :]
        x_de_traffic = self.output_layer_traffic(x_de_traffic)
        x_de_traffic = x_de_traffic.reshape(num_nodes, self.predict_len, 1)

        x_de_user = self.patch_decoder(x_de_user).squeeze( )
        x_de_user = self.positional_encoding_layer_decoder(x_de_user)
        x_de_user = self.decoder(x_de_user, memory = x_en, tgt_mask = self.mask)
        x_de_user = x_de_user[:, -1 * self.predict_len:, :]
        x_de_user = self.output_layer_traffic(x_de_user)
        x_de_user = x_de_user.reshape(num_nodes, self.predict_len, 1)

        x_de = torch.cat([x_de_traffic, x_de_user], dim = -1)
        return x_de


# x = torch.rand([12,21,12,2])
# model = Cross_Graph()
# y = model(x)
# print(y.shape)

class Cross_Graph_hier(nn.Module):
    def __init__(self, input_size = 1, seq_len = 12, seg_len = 3, encoder_hidden_dim = 64,
                 decoder_hidden_dim = 64, predict_len = 1, dim_k = 32, graph_num = 4,
                 n_head = 8, encoder_num_layers = 2, decoder_num_layers = 2, attention_bias = None,
                 layer_norm_eps: float = 1e-5,
                 dim_feedforward = 2048, ff_dropout = 0.5):
        super(Cross_Graph_hier, self).__init__( )
        self.input_size = input_size
        self.seq_len = seq_len
        self.seg_len = seg_len
        self.patch_num = int(seq_len / seg_len)
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.predict_len = predict_len
        self.dim_k = dim_k
        self.graph_num = graph_num
        self.n_head = n_head
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.attention_bias = attention_bias

        self.patch_traffic = DSW_embedding(seg_len, self.encoder_hidden_dim)
        self.positional_encoding_layer_traffic = PositionalEncoder(d_model = self.encoder_hidden_dim)
        self.encoder_traffic_temporal = nn.ModuleList( )
        for i in range(self.encoder_num_layers):
            self.encoder_traffic_temporal.append(TransformerEncoder_cross(
                TransformerEncoderLayer_cross(d_model = self.encoder_hidden_dim, nhead = self.n_head,
                                              batch_first = True), num_layers = 1))

        self.encoder_traffic_spatial = nn.ModuleList( )
        for i in range(self.graph_num):
            self.encoder_traffic_spatial.append(
                GraphAttention(self.encoder_hidden_dim * self.patch_num, 64, self.encoder_hidden_dim * self.patch_num))

        self.patch_user = DSW_embedding(seg_len, self.encoder_hidden_dim)
        self.positional_encoding_layer_user = PositionalEncoder(d_model = self.encoder_hidden_dim)
        self.encoder_user_temporal = nn.ModuleList( )
        for i in range(self.encoder_num_layers):
            self.encoder_user_temporal.append(TransformerEncoder_cross(
                TransformerEncoderLayer_cross(d_model = self.encoder_hidden_dim, nhead = self.n_head,
                                              batch_first = True), num_layers = 1))

        self.encoder_user_spatial = nn.ModuleList( )
        for i in range(self.graph_num):
            self.encoder_user_spatial.append(
                GraphAttention(self.encoder_hidden_dim * self.patch_num, 64, self.encoder_hidden_dim * self.patch_num))

        self.spatial_fu_att_traffic = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.patch_num * self.encoder_hidden_dim, nhead = 8, batch_first=True), num_layers = 1)
        self.spatial_fu_att_user = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.patch_num * self.encoder_hidden_dim, nhead = 8, batch_first=True), num_layers = 1)

        # self.encoder_attn = SelfAttention(2 * self.encoder_hidden_dim, dim_k, self.decoder_hidden_dim)

        self.patch_decoder = DSW_embedding(seg_len, 2 * self.decoder_hidden_dim)
        self.positional_encoding_layer_decoder = PositionalEncoder(d_model = 2 * self.decoder_hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model = 2 * self.decoder_hidden_dim, nhead = n_head,
                                                   batch_first = True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers = self.decoder_num_layers)

        self.output_layer_traffic = nn.Linear(self.decoder_hidden_dim, self.predict_len)
        self.output_layer_user = nn.Linear(self.decoder_hidden_dim, self.predict_len)

    def forward(self, x):
        assert self.graph_num == x.shape[1]
        num_neighbors = x.shape[2]
        seq_len = x.shape[3]
        num_nodes = x.shape[0]
        traffic = x[:, :, :, :, 0]
        user = x[:, :, :, :, 1]
        traffic_mask = traffic.detach().cpu().numpy()
        mask_idx = np.where(traffic_mask[:,:,:,0] == 0, 0,1)

        # print(mask_idx,mask_idx.shape)
        mask = np.ones([num_nodes,self.graph_num,num_neighbors])
        mask[mask_idx == 1] = 0
        # print(mask)
        mask = ma.make_mask(mask, )
        # print(mask)
        if mask.any():
            # print('1')
            mask = mask.repeat(mask.shape[2],1).reshape(num_nodes,self.graph_num,num_neighbors,num_neighbors)
            mask = torch.tensor(mask,).to(x.device).permute(1,0,2,3)
        else:
            # print('2')
            mask = torch.ones(self.graph_num,num_nodes,num_neighbors,num_neighbors).to(x.device)
        # print(mask[0,0],mask.shape)
        # y = y.repeat(y.shape[1], 0).reshape(2, 7, 7)
        x_de_traffic = torch.zeros([num_nodes, self.seq_len, 1]).to(x.device)
        x_de_traffic[:, :-1 * self.predict_len, 0] = traffic[:, 0, 0, self.predict_len:]
        x_de_user = torch.zeros([num_nodes, self.seq_len, 1]).to(x.device)
        x_de_user[:, :-1 * self.predict_len, 0] = user[:, 0, 0, self.predict_len:]
        x_de = torch.concat((x_de_traffic, x_de_user), dim = 1)

        traffic = traffic.permute(1, 0, 2, 3).reshape(self.graph_num, num_nodes, num_neighbors, seq_len, 1)
        user = user.permute(1, 0, 2, 3).reshape(self.graph_num, num_nodes, num_neighbors, seq_len, 1)
        # self.mask = nn.Transformer.generate_square_subsequent_mask(self.patch_num).to(x.device)

        traffic = traffic.reshape(self.graph_num * num_nodes * num_neighbors, seq_len, 1)
        user = user.reshape(self.graph_num * num_nodes * num_neighbors, seq_len, 1)

        # print(traffic.shape)
        traffic = self.patch_traffic(traffic).squeeze( )
        # print(traffic.shape)
        traffic = self.positional_encoding_layer_traffic(traffic)
        user = self.patch_user(user).squeeze( )
        user = self.positional_encoding_layer_user(user)

        for i in range(self.encoder_num_layers):
            traffic = self.encoder_traffic_temporal[i](traffic, user)
            user = self.encoder_user_temporal[i](user, traffic)
        # print(traffic.shape, user.shape)

        traffic = traffic.permute(1, 0, 2).reshape(self.patch_num, self.graph_num, num_nodes, num_neighbors,
                                                   -1).permute(1, 2, 3, 0, 4)
        user = user.permute(1, 0, 2).reshape(self.patch_num, self.graph_num, num_nodes, num_neighbors, -1).permute(1,
                                                                                                                   2, 3,
                                                                                                                   0, 4)
        # print(traffic.shape, user.shape)
        traffic = traffic.reshape(self.graph_num, num_nodes, num_neighbors, -1)
        user = user.reshape(self.graph_num, num_nodes, num_neighbors, -1)
        traffic_cls = torch.rand(traffic[0].shape).to(traffic.device).unsqueeze(0)
        user_cls = torch.rand(user[0].shape).to(user.device).unsqueeze(0)
        traffic = torch.concat([traffic,traffic_cls],dim = 0)
        user = torch.concat([user,user_cls],dim = 0)
        # print(traffic.shape, user.shape)

        traffic_en = torch.zeros(traffic.shape).to(traffic.device)
        user_en = torch.zeros(user.shape).to(user.device)
        for i in range(self.graph_num):
            traffic_en[i] = self.encoder_traffic_spatial[i](traffic[i],mask = mask[i])
            user_en[i] = self.encoder_user_spatial[i](user[i],mask = mask[i])

        # print(traffic_en.shape, user_en.shape)
        traffic_en = traffic_en[:, :, 0, :]
        user_en = user_en[:, :, 0, :]

        traffic_en = traffic_en.permute(1, 0, 2)
        user_en = user_en.permute(1, 0, 2)  # graph, nodes, patch*embedding

        traffic_en = self.spatial_fu_att_traffic(traffic_en)
        user_en = self.spatial_fu_att_user(user_en)
        # print(traffic_en.shape, user_en.shape)
        traffic_en = traffic_en[:,-1,:].reshape(num_nodes,self.patch_num,self.encoder_hidden_dim)
        user_en = user_en[:,-1,:].reshape(num_nodes,self.patch_num,self.encoder_hidden_dim)

        # print(traffic_en.shape, user_en.shape) #nodes, patch, embedding
        x_en = torch.concat([traffic_en, user_en], 2)
        # x_en = self.encoder_attn(x_en)
        # print(x_en.shape,x_de.shape)
        #
        x_de = self.patch_decoder(x_de).squeeze( )
        # print(x_de.shape)
        x_de = self.positional_encoding_layer_decoder(x_de)
        x_de = self.decoder(x_de, memory = x_en)
        x_de_traffic = x_de[:, -1, :self.decoder_hidden_dim]
        x_de_traffic = self.output_layer_traffic(x_de_traffic)
        x_de_traffic = x_de_traffic.unsqueeze(-1)
        x_de_user = x_de[:, -1, self.decoder_hidden_dim:]
        x_de_user = self.output_layer_traffic(x_de_user)
        x_de_user = x_de_user.unsqueeze(-1)
        # print(x_de_traffic.shape)
        x_de = torch.cat([x_de_traffic, x_de_user], dim = -1)
        return x_de

# x = torch.randint(2,3,[12, 4, 21, 12, 2],dtype = torch.float)
# model = Cross_Graph_hier()
# y = model(x)
# print('output', y.shape)

class Cross_Graph_hier_nopatch(nn.Module):
    def __init__(self, input_size = 1, seq_len = 12, seg_len = 3, encoder_hidden_dim = 64,
                 decoder_hidden_dim = 64, predict_len = 1, dim_k = 32, graph_num = 4,
                 n_head = 8, encoder_num_layers = 2, decoder_num_layers = 2, attention_bias = None,
                 layer_norm_eps: float = 1e-5,
                 dim_feedforward = 2048, ff_dropout = 0.5):
        super(Cross_Graph_hier_nopatch, self).__init__( )
        self.input_size = input_size
        self.seq_len = seq_len
        self.seg_len = seg_len
        self.patch_num = seq_len
        self.encoder_hidden_dim = encoder_hidden_dim
        self.decoder_hidden_dim = decoder_hidden_dim
        self.predict_len = predict_len
        self.dim_k = dim_k
        self.graph_num = graph_num
        self.n_head = n_head
        self.encoder_num_layers = encoder_num_layers
        self.decoder_num_layers = decoder_num_layers
        self.attention_bias = attention_bias

        # self.patch_traffic = DSW_embedding(seg_len, self.encoder_hidden_dim)
        self.patch_traffic = nn.Linear(1, self.encoder_hidden_dim)
        self.positional_encoding_layer_traffic = PositionalEncoder(d_model = self.encoder_hidden_dim)
        self.encoder_traffic_temporal = nn.ModuleList( )
        for i in range(self.encoder_num_layers):
            self.encoder_traffic_temporal.append(TransformerEncoder_cross(
                TransformerEncoderLayer_cross(d_model = self.encoder_hidden_dim, nhead = self.n_head,
                                              batch_first = True), num_layers = 1))

        self.encoder_traffic_spatial = nn.ModuleList( )
        for i in range(self.graph_num):
            self.encoder_traffic_spatial.append(
                GraphAttention(self.encoder_hidden_dim * self.patch_num, 64, self.encoder_hidden_dim * self.patch_num))

        # self.patch_user = DSW_embedding(seg_len, self.encoder_hidden_dim)
        self.patch_user = nn.Linear(1, self.encoder_hidden_dim)
        self.positional_encoding_layer_user = PositionalEncoder(d_model = self.encoder_hidden_dim)
        self.encoder_user_temporal = nn.ModuleList( )
        for i in range(self.encoder_num_layers):
            self.encoder_user_temporal.append(TransformerEncoder_cross(
                TransformerEncoderLayer_cross(d_model = self.encoder_hidden_dim, nhead = self.n_head,
                                              batch_first = True), num_layers = 1))

        self.encoder_user_spatial = nn.ModuleList( )
        for i in range(self.graph_num):
            self.encoder_user_spatial.append(
                GraphAttention(self.encoder_hidden_dim * self.patch_num, 64, self.encoder_hidden_dim * self.patch_num))

        self.spatial_fu_att_traffic = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.patch_num * self.encoder_hidden_dim, nhead = 8, batch_first=True), num_layers = 1)
        self.spatial_fu_att_user = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.patch_num * self.encoder_hidden_dim, nhead = 8, batch_first=True), num_layers = 1)

        # self.encoder_attn = SelfAttention(2 * self.encoder_hidden_dim, dim_k, self.decoder_hidden_dim)

        # self.patch_decoder = DSW_embedding(seg_len, 2 * self.decoder_hidden_dim)
        self.patch_decoder = nn.Linear(1, 2 *self.encoder_hidden_dim)
        self.positional_encoding_layer_decoder = PositionalEncoder(d_model = 2 * self.decoder_hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model = 2 * self.decoder_hidden_dim, nhead = n_head,
                                                   batch_first = True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers = self.decoder_num_layers)

        self.output_layer_traffic = nn.Linear(self.decoder_hidden_dim, self.predict_len)
        self.output_layer_user = nn.Linear(self.decoder_hidden_dim, self.predict_len)

    def forward(self, x):
        assert self.graph_num == x.shape[1]
        num_neighbors = x.shape[2]
        seq_len = x.shape[3]
        num_nodes = x.shape[0]
        traffic = x[:, :, :, :, 0]
        user = x[:, :, :, :, 1]
        traffic_mask = traffic.detach().cpu().numpy()
        mask_idx = np.where(traffic_mask[:,:,:,0] == 0, 0,1)

        # print(mask_idx,mask_idx.shape)
        mask = np.ones([num_nodes,self.graph_num,num_neighbors])
        mask[mask_idx == 1] = 0
        # print(mask)
        mask = ma.make_mask(mask, )
        # print(mask)
        if mask.any():
            # print('1')
            mask = mask.repeat(mask.shape[2],1).reshape(num_nodes,self.graph_num,num_neighbors,num_neighbors)
            mask = torch.tensor(mask,).to(x.device).permute(1,0,2,3)
        else:
            # print('2')
            mask = torch.ones(self.graph_num,num_nodes,num_neighbors,num_neighbors).to(x.device)
        # print(mask[0,0],mask.shape)
        # y = y.repeat(y.shape[1], 0).reshape(2, 7, 7)
        x_de_traffic = torch.zeros([num_nodes, self.seq_len, 1]).to(x.device)
        x_de_traffic[:, :-1 * self.predict_len, 0] = traffic[:, 0, 0, self.predict_len:]
        x_de_user = torch.zeros([num_nodes, self.seq_len, 1]).to(x.device)
        x_de_user[:, :-1 * self.predict_len, 0] = user[:, 0, 0, self.predict_len:]
        x_de = torch.concat((x_de_traffic, x_de_user), dim = 1)

        traffic = traffic.permute(1, 0, 2, 3).reshape(self.graph_num, num_nodes, num_neighbors, seq_len, 1)
        user = user.permute(1, 0, 2, 3).reshape(self.graph_num, num_nodes, num_neighbors, seq_len, 1)
        # self.mask = nn.Transformer.generate_square_subsequent_mask(self.patch_num).to(x.device)

        traffic = traffic.reshape(self.graph_num * num_nodes * num_neighbors, seq_len, 1)
        user = user.reshape(self.graph_num * num_nodes * num_neighbors, seq_len, 1)

        # print(traffic.shape)
        traffic = self.patch_traffic(traffic).squeeze( )
        # print(traffic.shape)
        traffic = self.positional_encoding_layer_traffic(traffic)
        user = self.patch_user(user).squeeze( )
        user = self.positional_encoding_layer_user(user)

        for i in range(self.encoder_num_layers):
            traffic = self.encoder_traffic_temporal[i](traffic, user)
            user = self.encoder_user_temporal[i](user, traffic)
        # print(traffic.shape, user.shape)

        traffic = traffic.permute(1, 0, 2).reshape(self.patch_num, self.graph_num, num_nodes, num_neighbors,
                                                   -1).permute(1, 2, 3, 0, 4)
        user = user.permute(1, 0, 2).reshape(self.patch_num, self.graph_num, num_nodes, num_neighbors, -1).permute(1,
                                                                                                                   2, 3,
                                                                                                                   0, 4)
        # print(traffic.shape, user.shape)
        traffic = traffic.reshape(self.graph_num, num_nodes, num_neighbors, -1)
        user = user.reshape(self.graph_num, num_nodes, num_neighbors, -1)
        traffic_cls = torch.rand(traffic[0].shape).to(traffic.device).unsqueeze(0)
        user_cls = torch.rand(user[0].shape).to(user.device).unsqueeze(0)
        traffic = torch.concat([traffic,traffic_cls],dim = 0)
        user = torch.concat([user,user_cls],dim = 0)
        # print(traffic.shape, user.shape)
        traffic_en = torch.zeros(traffic.shape).to(traffic.device)
        user_en = torch.zeros(user.shape).to(user.device)
        for i in range(self.graph_num):
            traffic_en[i] = self.encoder_traffic_spatial[i](traffic[i],mask = mask[i])
            user_en[i] = self.encoder_user_spatial[i](user[i],mask = mask[i])

        # print(traffic_en.shape, user_en.shape)
        traffic_en = traffic_en[:, :, 0, :]
        user_en = user_en[:, :, 0, :]

        traffic_en = traffic_en.permute(1, 0, 2)
        user_en = user_en.permute(1, 0, 2)  # graph, nodes, patch*embedding

        traffic_en = self.spatial_fu_att_traffic(traffic_en)
        user_en = self.spatial_fu_att_user(user_en)
        # print(traffic_en.shape, user_en.shape)
        traffic_en = traffic_en[:,-1,:].reshape(num_nodes,self.patch_num,self.encoder_hidden_dim)
        user_en = user_en[:,-1,:].reshape(num_nodes,self.patch_num,self.encoder_hidden_dim)

        # print(traffic_en.shape, user_en.shape) #nodes, patch, embedding
        x_en = torch.concat([traffic_en, user_en], 2)
        # x_en = self.encoder_attn(x_en)
        # print(x_en.shape,x_de.shape)
        #
        x_de = self.patch_decoder(x_de).squeeze( )
        # print(x_de.shape)
        x_de = self.positional_encoding_layer_decoder(x_de)
        x_de = self.decoder(x_de, memory = x_en)
        x_de_traffic = x_de[:, -1, :self.decoder_hidden_dim]
        x_de_traffic = self.output_layer_traffic(x_de_traffic)
        x_de_traffic = x_de_traffic.unsqueeze(-1)
        x_de_user = x_de[:, -1, self.decoder_hidden_dim:]
        x_de_user = self.output_layer_traffic(x_de_user)
        x_de_user = x_de_user.unsqueeze(-1)
        # print(x_de_traffic.shape)
        x_de = torch.cat([x_de_traffic, x_de_user], dim = -1)
        return x_de



# x = torch.randint(2,3,[12, 4, 21, 12, 2],dtype = torch.float)
# model = Cross_Graph_hier_nopatch()
# y = model(x)
# print('output', y.shape)