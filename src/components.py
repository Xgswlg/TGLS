import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import numpy as np
from typing import Optional
import math

def initializer_1d(input_tensor, initializer):
    assert len(input_tensor.size()) == 1
    input_tensor = input_tensor.view(-1, 1)
    input_tensor = initializer(input_tensor)
    return input_tensor.view(-1)

class MLPRepScorer(nn.Module):
    def __init__(self, input_size, inner_size, output_size, dropout=0.0):
        super(MLPRepScorer, self).__init__()
        self.rep_layer = nn.Linear(input_size, inner_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.scorer = nn.Linear(inner_size, output_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.rep_layer.weight)
        initializer_1d(self.rep_layer.bias, nn.init.xavier_uniform_)
        nn.init.xavier_uniform_(self.scorer.weight)
        initializer_1d(self.scorer.bias, nn.init.xavier_uniform_)

    def forward(self, x):
        rep = self.dropout_layer(
            F.relu(self.rep_layer.forward(x))
        )
        scores = self.scorer.forward(rep)
        return scores


# Copied from transformers.models.marian.modeling_marian.MarianSinusoidalPositionalEmbedding with Marian->RoFormer
class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(
            self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        super().__init__(num_positions, embedding_dim)
        self.weight = self._init_weight(self.weight)

    @staticmethod
    def _init_weight(out: nn.Parameter):
        """
        Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
        the 2nd half of the vector. [dim // 2:]
        """
        n_pos, dim = out.shape
        position_enc = np.array(
            [
                [pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)]
                for pos in range(n_pos)
            ]
        )
        out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
        sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
        out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
        out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
        out.detach_()
        return out

    @torch.no_grad()
    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length,
            past_key_values_length + seq_len,
            dtype=torch.long,
            device=self.weight.device,
        )
        return super().forward(positions)

class RGAT(nn.Module):

    def __init__(self, hidden_size, whole_tag_size, num_layers=2, gcn_dropout=0.2):
        super(RGAT, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(gcn_dropout)
        self.whole_tag_size = whole_tag_size
        # gat layer
        self.W = nn.ModuleDict()
        self.lns = nn.ModuleList()
        self.relu = nn.ReLU(inplace=True)

        for layer in range(num_layers):

            self.lns.append(nn.LayerNorm(hidden_size, elementwise_affine=False))
            self.W[str(layer)] = nn.ModuleList()
            for _ in range(whole_tag_size):
                self.W[str(layer)].append(nn.Linear(hidden_size, hidden_size, bias=False))


    def forward(self, adj, feature):
        # adj: [b, whole_tag_size, l, l]  feature: [b, l, h]

        for layer in range(self.num_layers):
            addition = torch.zeros_like(feature)
            for rel in range(self.whole_tag_size):
                h_r = self.W[str(layer)][rel](feature)
                h_r = self.relu(adj[:, rel, :, :].bmm(h_r))
                addition = addition + h_r

            feature = self.lns[layer](self.relu(addition) / self.whole_tag_size)
            feature = self.dropout(feature) if layer < self.num_layers - 1 else feature

        return feature

class TokenPairModeling(nn.Module):
    def __init__(self, hidden_size, essential_tag_size, whole_tag_size, data_set='mpqa', d_k=64, dropout=0.3, graph_stack_num=1):
        super().__init__()
        self.data_set = data_set
        self.dropout = nn.Dropout(dropout)
        self.header_num = whole_tag_size + 1
        self.relu = nn.ReLU()
        self.essential_tag_size = essential_tag_size
        self.whole_tag_size = whole_tag_size
        self.d_k = d_k
        self.embed_positions = RoFormerSinusoidalPositionalEmbedding(
            512,
            d_k,
        )
        self.graph_stack_num = graph_stack_num
        self.rgat = nn.ModuleList([RGAT(hidden_size, self.header_num) for _ in range(graph_stack_num)])

        # prediction layer: attention scoring and adaptive thresholding
        self.W_Q = MLPRepScorer(hidden_size*2, (hidden_size*2+d_k * essential_tag_size) // 2, d_k * essential_tag_size)
        self.W_K = MLPRepScorer(hidden_size*2, (hidden_size*2+d_k * essential_tag_size) // 2, d_k * essential_tag_size)
        self.W_Q4Threshold = nn.Linear(hidden_size*2, d_k * 1)
        self.W_K4Threshold = nn.Linear(hidden_size*2, d_k * 1)

        # graph layer: attention scoring and adaptive thresholding
        self.W_Q4Graph = nn.ModuleList([MLPRepScorer(hidden_size, (hidden_size + d_k * self.header_num) // 2, d_k * self.header_num) for _ in range(graph_stack_num)])
        self.W_K4Graph = nn.ModuleList([MLPRepScorer(hidden_size, (hidden_size + d_k * self.header_num) // 2, d_k * self.header_num) for _ in range(graph_stack_num)])
        self.W_Q4GraphThreshold = nn.ModuleList([nn.Linear(hidden_size, d_k * 1) for _ in range(graph_stack_num)])
        self.W_K4GraphThreshold = nn.ModuleList([nn.Linear(hidden_size, d_k * 1) for _ in range(graph_stack_num)])

    @staticmethod
    def apply_rotary_position_embeddings(sinusoidal_pos, query_layer):
        # https://kexue.fm/archives/8265
        # sin [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        # cos [batch_size, num_heads, sequence_length, embed_size_per_head//2]
        sin, cos = sinusoidal_pos.chunk(2, dim=-1)
        # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
        # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
        cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
        # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
        rotate_half_query_layer = torch.stack(
            [-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1
        ).reshape_as(query_layer)
        query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos

        return query_layer

    def graph_layer_forward(self, graph_inputs, attention_mask, sinusoidal_pos, layer):
        seq_len = graph_inputs.size()[-2]
        batch_size = graph_inputs.size()[0]
        # graph layer thresholding
        ts_q4graph = (self.W_Q4GraphThreshold[layer](graph_inputs) * attention_mask).view(batch_size, -1, 1,
                                                                                   self.d_k).transpose(1, 2)
        ts_k4graph = (self.W_K4GraphThreshold[layer](graph_inputs) * attention_mask).view(batch_size, -1, 1,
                                                                                   self.d_k).transpose(1, 2)
        ts_q4graph = self.apply_rotary_position_embeddings(sinusoidal_pos, ts_q4graph)
        ts_k4graph = self.apply_rotary_position_embeddings(sinusoidal_pos, ts_k4graph)

        threshold_matrix4graph = torch.matmul(ts_q4graph, ts_k4graph.transpose(-1, -2)) / np.sqrt(self.d_k)
        threshold4graph = torch.cat(
            [threshold_matrix4graph[:, :, ind, ind:].transpose(-1, -2) for ind in range(seq_len)], dim=1)
        threshold4graph = self.relu(threshold4graph)

        # graph layer attention scoring
        q4graph = (self.W_Q4Graph[layer](graph_inputs) * attention_mask).view(batch_size, -1, self.header_num,
                                                                        self.d_k).transpose(1, 2)
        k4graph = (self.W_K4Graph[layer](graph_inputs) * attention_mask).view(batch_size, -1, self.header_num,
                                                                        self.d_k).transpose(1, 2)
        q4graph = self.apply_rotary_position_embeddings(sinusoidal_pos, q4graph)
        k4graph = self.apply_rotary_position_embeddings(sinusoidal_pos, k4graph)                                                                        
        scoring_matrix4graph = torch.matmul(q4graph, k4graph.transpose(-1, -2)) / np.sqrt(self.d_k)
        scoring4graph = torch.cat([scoring_matrix4graph[:, :self.whole_tag_size, ind, ind:].transpose(-1, -2) for ind in range(seq_len)], dim=1)

        # get adj_matrix for RGAT
        scoring_tri4graph = torch.triu(scoring_matrix4graph, diagonal=0) - torch.triu(scoring_matrix4graph, diagonal=1)
        scoring_matrix4graph = torch.triu(scoring_matrix4graph, diagonal=1)
        att_matrix = scoring_matrix4graph + scoring_matrix4graph.transpose(-1, -2) + scoring_tri4graph
        att_mask_choose = (att_matrix != 0)
        att_matrix = nn.Softmax(dim=-1)(att_matrix)
        adj_matrix = att_matrix * att_mask_choose

        graph_outputs = self.rgat[layer](adj_matrix, graph_inputs)

        return graph_outputs, scoring4graph, threshold4graph

    def forward(self, seq_hiddens, attention_mask):
        '''
        seq_hiddens: (batch_size, seq_len, hidden_size)
        return:
            shaking_hiddenss: (batch_size, (1 + seq_len) * seq_len / 2, hidden_size) (32, 5+4+3+2+1, 5)
        '''
        graph_inputs = seq_hiddens
        residual = torch.zeros_like(seq_hiddens)
        residual.copy_(seq_hiddens)
        seq_len = seq_hiddens.size()[-2]
        batch_size = seq_hiddens.size()[0]
        shaking_hiddens_list = []
        sinusoidal_pos = self.embed_positions(seq_hiddens.shape[:-1])[
                         None, None, :, :
                         ]
        total_attention_scoring4graph, total_threshold4graph = [], []
        for layer in range(self.graph_stack_num):
            graph_inputs, scoring4graph, threshold4graph = self.graph_layer_forward(graph_inputs, attention_mask, sinusoidal_pos, layer)
            total_attention_scoring4graph.append(scoring4graph)
            total_threshold4graph.append(threshold4graph)

        inputs = torch.cat([residual, graph_inputs], dim=-1)
        inputs = self.dropout(inputs)

        # prediction layer thresholding
        ts_q4pred = (self.W_Q4Threshold(inputs) * attention_mask).view(batch_size, -1, 1, self.d_k).transpose(1, 2)
        ts_k4pred = (self.W_K4Threshold(inputs) * attention_mask).view(batch_size, -1, 1, self.d_k).transpose(1, 2)

        ts_q4pred = self.apply_rotary_position_embeddings(sinusoidal_pos, ts_q4pred)
        ts_k4pred = self.apply_rotary_position_embeddings(sinusoidal_pos, ts_k4pred)

        threshold_matrix4pred = torch.matmul(ts_q4pred, ts_k4pred.transpose(-1, -2)) / np.sqrt(self.d_k)
        threshold4pred = torch.cat(
            [threshold_matrix4pred[:, :, ind, ind:].transpose(-1, -2) for ind in range(seq_len)], dim=1)

        threshold4pred = self.relu(threshold4pred)

        # prediction layer attention scoring
        q4pred = (self.W_Q(inputs) * attention_mask).view(batch_size, -1, self.essential_tag_size, self.d_k).transpose(1, 2)
        k4pred = (self.W_K(inputs) * attention_mask).view(batch_size, -1, self.essential_tag_size, self.d_k).transpose(1, 2)

        q4pred = self.apply_rotary_position_embeddings(sinusoidal_pos, q4pred)
        k4pred = self.apply_rotary_position_embeddings(sinusoidal_pos, k4pred)

        scoring_matrix4pred = torch.matmul(q4pred, k4pred.transpose(-1, -2)) / np.sqrt(self.d_k)

        attention_scoring4pred = torch.cat([scoring_matrix4pred[:, :, ind, ind:].transpose(-1, -2) for ind in range(seq_len)],
                                               dim=1)

        return attention_scoring4pred, threshold4pred, total_attention_scoring4graph, total_threshold4graph
