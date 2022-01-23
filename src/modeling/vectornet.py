from typing import Dict, List, Tuple, NamedTuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import copy
from modeling.decoder import Decoder_predict, DecoderResCat
from modeling.lib import MLP, GlobalGraph, LayerNorm, SubGraph, CrossAttention, GlobalGraphRes
import utils
from modeling.TF_utils import (Decoder, DecoderLayer, Encoder, EncoderDecoder,
                        EncoderLayer, GeneratorWithParallelHeads626,
                        LinearEmbedding, MultiHeadAttention,
                        PointerwiseFeedforward, PositionalEncoding, EncoderLayer_NEW,
                        SublayerConnection, Generator_full)

class NewSubGraph(nn.Module):

    def __init__(self, hidden_size, depth=None):
        super(NewSubGraph, self).__init__()
        if depth is None:
            depth = args.sub_graph_depth
        self.layers = nn.ModuleList([MLP(hidden_size, hidden_size // 2) for _ in range(depth)])
        #if 'point_level-4' in args.other_params:
        self.layer_0 = MLP(hidden_size)
        self.layers = nn.ModuleList([GlobalGraph(hidden_size, num_attention_heads=2) for _ in range(depth)])
        self.layers_2 = nn.ModuleList([LayerNorm(hidden_size) for _ in range(depth)])
        self.layers_3 = nn.ModuleList([LayerNorm(hidden_size) for _ in range(depth)])
        self.layers_4 = nn.ModuleList([GlobalGraph(hidden_size) for _ in range(depth)])
        #    if 'point_level-4-3' in args.other_params:
        self.layer_0_again = MLP(hidden_size)

    def forward(self, input_list: list):
        batch_size = len(input_list)
        device = input_list[0].device
        hidden_states, lengths = utils.merge_tensors(input_list, device)
        #print("hidden_states.shape:", hidden_states.shape)
        hidden_size = hidden_states.shape[2]
        max_vector_num = hidden_states.shape[1]

        if 'point_level-4' in args.other_params:
            attention_mask = torch.zeros([batch_size, max_vector_num, max_vector_num], device=device)
            hidden_states = self.layer_0(hidden_states)

        if 'point_level-4-3' in args.other_params:
            hidden_states = self.layer_0_again(hidden_states)
        for i in range(batch_size):
            assert lengths[i] > 0
            attention_mask[i, :lengths[i], :lengths[i]].fill_(1)

        for layer_index, layer in enumerate(self.layers):
            temp = hidden_states
            # hidden_states = layer(hidden_states, attention_mask)
            # hidden_states = self.layers_2[layer_index](hidden_states)
            # hidden_states = F.relu(hidden_states) + temp
            hidden_states = layer(hidden_states, attention_mask)
            hidden_states = F.relu(hidden_states)
            hidden_states = hidden_states + temp
            hidden_states = self.layers_2[layer_index](hidden_states)

        return torch.max(hidden_states, dim=1)[0], torch.cat(utils.de_merge_tensors(hidden_states, lengths))


class VectorNet(nn.Module):
    r"""
    VectorNet

    It has two main components, sub graph and global graph.

    Sub graph encodes a polyline as a single vector.
    """

    def __init__(self, args_: utils.Args):
        super(VectorNet, self).__init__()
        global args
        args = args_
        hidden_size = args.hidden_size

        self.sub_graph = SubGraph(args, hidden_size)
        #if 'point_level' in args.other_params:
            # TODO!  tnt needs stage_one?
            # assert 'stage_one' in args.other_params
        self.point_level_sub_graph = NewSubGraph(hidden_size)
        self.point_level_cross_attention = CrossAttention(hidden_size)

        self.global_graph = GlobalGraph(hidden_size)
        #if 'enhance_global_graph' in args.other_params:
        self.global_graph = GlobalGraphRes(hidden_size)
        #if 'laneGCN' in args.other_params:
        self.laneGCN_A2L = CrossAttention(hidden_size)
        self.laneGCN_L2L = GlobalGraphRes(hidden_size)
        self.laneGCN_L2A = CrossAttention(hidden_size)
        self.agents_A = CrossAttention(hidden_size)
        self.agents_A0 = CrossAttention(hidden_size)
        #if 'stage_one' in args.other_params:
        self.stage_one_sub_graph_map = SubGraph(args, hidden_size)

        self.decoder = Decoder_predict(args, self)

        #if 'complete_traj' in args.other_params:
        self.decoder.complete_traj_cross_attention = CrossAttention(hidden_size)
        self.decoder.complete_traj_decoder = DecoderResCat(hidden_size, hidden_size * 3, out_features=self.decoder.future_frame_num * 2)

        N = 2
        N_lane = 2
        N_social = 2
        d_model = 128
        d_ff = 256
        #pos_dim = 64
        pos_dim = 128
        dist_dim = 128
        h = 2
        dropout = 0
        dropout_atten = 0
        lane_inp_size = 128
        hist_inp_size = 128
        c = copy.deepcopy
        num_queries = 50
        dec_out_size = 2
        self.num_queries = num_queries
        attn = MultiHeadAttention(h, d_model, dropout=dropout_atten)
        ff = PointerwiseFeedforward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)

        self.query_embed = nn.Embedding(num_queries, d_model)
        self.hist_tf = EncoderDecoder(
            Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
            Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
            nn.Sequential(LinearEmbedding(hist_inp_size, d_model), c(position))
        )

        self.lane_enc = Encoder(EncoderLayer(
            d_model, c(attn), c(ff), dropout), N_lane)
        self.lane_dec = Decoder(DecoderLayer(
            d_model, c(attn), c(attn), c(ff), dropout), N_lane)
        self.lane_emb = LinearEmbedding(lane_inp_size, d_model)

        self.agent_emb = LinearEmbedding(lane_inp_size, d_model)



        self.social_enc = Encoder(EncoderLayer(
            d_model, c(attn), c(ff), dropout), N_social)

        self.pos_emb = nn.Sequential(
            nn.Linear(2, pos_dim, bias=True),
            nn.LayerNorm(pos_dim),
            nn.ReLU(),
            nn.Linear(pos_dim, pos_dim, bias=True))
        self.dist_emb = nn.Sequential(
            nn.Linear(num_queries*d_model, dist_dim, bias=True),
            nn.LayerNorm(dist_dim),
            nn.ReLU(),
            nn.Linear(dist_dim, dist_dim, bias=True))

        self.fusion1 = nn.Sequential(
            nn.Linear(d_model+pos_dim, d_model, bias=True),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model, bias=True))
        self.fusion2 = nn.Sequential(
            nn.Linear(dist_dim+pos_dim, d_model, bias=True),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model, bias=True))


        self.out_pos_emb = nn.Sequential(
            nn.Linear(2, pos_dim, bias=True),
            nn.LayerNorm(pos_dim),
            nn.ReLU(),
            nn.Linear(pos_dim, pos_dim, bias=True))

        self.prediction_header = GeneratorWithParallelHeads626(d_model*2, dec_out_size, dropout)

        self.generator_header = Generator_full(d_model*3, 60, dropout)

    def forward_encode_sub_graph(self, mapping: List[Dict], matrix: List[np.ndarray], polyline_spans: List[List[slice]],
                                 device, batch_size) -> Tuple[List[Tensor], List[Tensor]]:
        """
        :param matrix: each value in list is vectors of all element (shape [-1, 128])
        :param polyline_spans: vectors of i_th element is matrix[polyline_spans[i]]
        :return: hidden states of all elements and hidden states of lanes
        """
        outputs_coord_batch = []
        outputs_class_batch = []
        outputs_traj_batch = []
        input_list_list = []
        # TODO(cyrushx): This is not used? Is it because input_list_list includes map data as well?
        # Yes, input_list_list includes map data, this will be used in the future release.
        map_input_list_list = []
        lane_states_batch = None
        for i in range(batch_size):
            input_list = []
            map_input_list = []
            map_start_polyline_idx = mapping[i]['map_start_polyline_idx']
            for j, polyline_span in enumerate(polyline_spans[i]):
                tensor = torch.tensor(matrix[i][polyline_span], device=device)
                input_list.append(tensor)
                if j >= map_start_polyline_idx:
                    map_input_list.append(tensor)

            input_list_list.append(input_list)
            map_input_list_list.append(map_input_list)

        #if 'point_level' in args.other_params:
        element_states_batch = []
        point_level_features_list = []
        for i in range(batch_size):
            a, b = self.point_level_sub_graph(input_list_list[i])
            #print("a.shape", a.shape)
            #print("b.shape", b.shape)
            #print(a[0])
            element_states_batch.append(a)
            point_level_features_list.append(b)
            mapping[i]['point_level_features'] = point_level_features_list[i]


        if 'stage_one' in args.other_params:
            assert 'sub_graph_map' not in args.other_params
            lane_states_batch = utils.merge_tensors_not_add_dim(map_input_list_list,
                                                                module=self.stage_one_sub_graph_map,
                                                                sub_batch_size=16, device=device)
            #print("lane_states_batch.shape,", lane_states_batch.shape)
        if 'laneGCN' in args.other_params:
            inputs_before_laneGCN, inputs_lengths_before_laneGCN = utils.merge_tensors(element_states_batch, device=device)
            for i in range(batch_size):
                map_start_polyline_idx = mapping[i]['map_start_polyline_idx']
                agents = element_states_batch[i][:map_start_polyline_idx]
                #print("agents:", agents[0])
                lanes = element_states_batch[i][map_start_polyline_idx:]
                #print("lanes:", lanes[0])
                #print("agents.shape: ", agents.shape, " lanes.shape: ", lanes.shape)

                social_num = agents.shape[0]
                neighbor_num = lanes.shape[0]
                self.query_batches = self.query_embed.weight.view(1, 1, *self.query_embed.weight.shape).repeat(1, social_num, 1, 1)
                #print("self.query_batches.shape", self.query_batches.shape, "agents.shape:", agents.shape,"agents.unsqueeze(0).unsqueeze(2)", agents.unsqueeze(0).unsqueeze(2).shape)
                agents_input = agents.unsqueeze(0).unsqueeze(2)
                #print("agents_input.shape:", agents_input.shape)
                hist_out = self.hist_tf(agents_input, self.query_batches, None, None)
                #print("hist_out.shape: ", hist_out.shape)
                lane_mask = torch.ones(1, 1, neighbor_num).cuda()
                #print("lanes.unsqueeze(0)", lanes.unsqueeze(0).shape, "lane_mask.shape", lane_mask.shape)
                lane_mem = self.lane_enc(self.lane_emb(lanes.unsqueeze(0)), lane_mask)
                #print("lane_mem.shape: ", lane_mem.shape)
                lane_mem = lane_mem.unsqueeze(1).repeat(1, social_num, 1, 1)
                lane_mask = lane_mask.unsqueeze(1).repeat(1, social_num, 1, 1)
                #print("lane_mem.shape: ", lane_mem.shape)
                #print("lane_mask.shape: ", lane_mask.shape)
                # Lane decoder

                lane_out = self.lane_dec(hist_out, lane_mem, lane_mask, None)
                #print("lane_out.shape: ", lane_out.shape)

                dist = lane_out.view(1, social_num, -1)
                dist = self.dist_emb(dist)
                #print("dist.shape", dist.shape)
                # Social layer
                #print(agents.unsqueeze(0).shape)
                social_inp = self.fusion2(torch.cat([agents.unsqueeze(0), dist], -1))
                social_mask = torch.ones((1, 1, social_num)).cuda()
                social_mem = self.social_enc(social_inp, social_mask)
                social_out = social_mem.unsqueeze(
                    dim=2).repeat(1, 1, self.num_queries, 1)
                #print("social_out.shape", social_out.shape)
                out = torch.cat([social_out, lane_out], -1)
                #print("out.shape", out.shape) #out.shape torch.Size([1, 44, 50, 256])
                # Prediction head
                outputs_coord, outputs_class = self.prediction_header(out)
                #lanes = lanes_out_tbd

                #print("outputs_coord.shape", outputs_coord.shape) #outputs_coord.shape torch.Size([1, 44, 50, 2])
                #print("outputs_class.shape", outputs_class.shape) #outputs_class.shape torch.Size([1, 44, 50])

                outputs_coord_feature = self.out_pos_emb(outputs_coord)
                out = torch.cat([out, outputs_coord_feature], -1)
                outputs_traj = self.generator_header(out)
                #print("outputs_traj", outputs_traj.shape)
                #element_states_batch[i] = torch.cat([agents, lanes])
                outputs_coord_batch.append(outputs_coord.squeeze(0))
                outputs_class_batch.append(outputs_class.squeeze(0))
                outputs_traj_batch.append(outputs_traj.squeeze(0))



        return outputs_coord_batch, outputs_class_batch, outputs_traj_batch

    # @profile
    def forward(self, mapping: List[Dict], device):
        import time
        global starttime
        starttime = time.time()
        #print("MAPPING")
        #print(mapping)
        matrix = utils.get_from_mapping(mapping, 'matrix')
        #print("matrix", matrix[0].shape)
        # vectors of i_th element is matrix[polyline_spans[i]]
        polyline_spans = utils.get_from_mapping(mapping, 'polyline_spans')
        batch_size = len(matrix)
        #print("batch_size: ", batch_size)
        if args.argoverse:
            utils.batch_init(mapping)

        outputs_coord_batch, outputs_class_batch, outputs_traj_batch = self.forward_encode_sub_graph(mapping, matrix, polyline_spans, device, batch_size)
        outputs_coord, coord_length = utils.merge_tensors_coor(outputs_coord_batch,hidden_size0=50, hidden_size1=2, device=device)
        outputs_class, class_length = utils.merge_tensors_class(outputs_class_batch,hidden_size0=50, device=device)
        outputs_traj, traj_length = utils.merge_tensors_traj(outputs_traj_batch, hidden_size0=50, hidden_size1=30, hidden_size2=2, device=device)
        #print("coord_length", coord_length)
        #print("class_length", class_length)

        #print("outputs_coord.shape", outputs_coord.shape)
        #print("outputs_class.shape", outputs_class.shape)
        #print("outputs_traj.shape", outputs_traj.shape)


        output = self.decoder(mapping, batch_size, outputs_coord, outputs_class, outputs_traj, coord_length, device=device)
        #print("output: ", output)

        return output
