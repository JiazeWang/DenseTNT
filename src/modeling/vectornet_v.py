from typing import Dict, List, Tuple, NamedTuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import copy
#from modeling.decoder_centerness import Decoder_predict
from modeling.decoder_centerness_dis_6 import Decoder_predict
from modeling.lib import MLP, GlobalGraph, LayerNorm, SubGraph, CrossAttention, GlobalGraphRes
import utils
from modeling.TF_utils import (Decoder, DecoderLayer, Encoder, EncoderDecoder,
                        EncoderLayer, GeneratorWithParallelHeads_centerness,
                        LinearEmbedding, MultiHeadAttention,
                        PointerwiseFeedforward, PositionalEncoding, EncoderLayer_NEW,
                        SublayerConnection, Generator_traj, Generator_centerness, GeneratorWithParallelHeads626_softmax)

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
        self.point_level_sub_graph_lane = NewSubGraph(hidden_size)
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
        num_queries = 6
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

        self.prediction_header = GeneratorWithParallelHeads626_softmax(d_model*2, dec_out_size, dropout)

        self.generator_header = Generator_traj(d_model*3, 60, dropout)

        self.generator_centerness = Generator_centerness(d_model*3, 1, dropout)
    def preprocess_traj(self, traj, device):
        '''
            Generate the trajectory mask for all agents (including target agent)

            Args:
                traj: [batch, max_agent_num, obs_len, 4]

            Returns:
                social mask: [batch, 1, max_agent_num]

        '''
        # social mask
        batch_size = len(traj)
        social_valid_len = []
        for i in range(0, batch_size):
            #print("traj[i].shape[0]:", traj[i].shape[0])
            social_valid_len.append(traj[i].shape[0])
        #print("batch_size,", batch_size, "social_valid_len:", social_valid_len)
        social_valid_len_max = max(social_valid_len)
        social_mask = torch.zeros(
            (batch_size, 1, social_valid_len_max)).to(device)
        traj_batch = torch.zeros(
            (batch_size, social_valid_len_max, 128)).to(device)
        for i in range(batch_size):
            social_mask[i, 0, :social_valid_len[i]] = 1

        for i in range(batch_size):
            neighbor_num = social_valid_len[i]
            traj_batch[i, 0:neighbor_num] = traj[i]


        return traj_batch, social_mask


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
            """
            assert 'sub_graph_map' not in args.other_params
            lane_states_batch = utils.merge_tensors_not_add_dim(map_input_list_list,
                                                                module=self.stage_one_sub_graph_map,
                                                                sub_batch_size=16, device=device)
            """
            lane_states_batch = []
            for i in range(batch_size):
                a, b = self.point_level_sub_graph_lane(map_input_list_list[i])
                lane_states_batch.append(a)
            #print("lane_states_batch.shape,", lane_states_batch.shape)
        agents_list = []
        lanes_list = []
        if 'laneGCN' in args.other_params:
            inputs_before_laneGCN, inputs_lengths_before_laneGCN = utils.merge_tensors(element_states_batch, device=device)
            for i in range(batch_size):
                map_start_polyline_idx = mapping[i]['map_start_polyline_idx']
                agents = element_states_batch[i][:map_start_polyline_idx]
                #print("agents:", agents.shape)
                #lanes = element_states_batch[i][map_start_polyline_idx:]
                lanes = lane_states_batch[i]
                #print("lanes:", lanes.shape)
                #print("agents.shape: ", agents.shape, " lanes.shape: ", lanes.shape, "lane_states_batch:", lane_states_batch[i].shape)
                #print(torch.max(lanes-lane_states_batch[i]), torch.min(lanes-lane_states_batch[i]))
                agents_list.append(agents)
                lanes_list.append(lanes)

            agent_batch, agent_mask = self.preprocess_traj(agents_list, device)
            #print("agent_batch.shape:", agent_batch.shape, "agent_mask.shape:", agent_mask.shape)

            lane_batch, lane_mask = self.preprocess_traj(lanes_list, device)
            print("lane_batch:", lane_batch.shape)
            print("lane_mask:", lane_mask.shape)
            #print("*agent_batch.shape[:2]: ", *agent_batch.shape[:2])
            batch_size = agent_batch.shape[0]
            social_num = agent_batch.shape[1]
            neighbor_num = lane_batch.shape[1]
            self.query_batches = self.query_embed.weight.view(1, 1, *self.query_embed.weight.shape).repeat(*agent_batch.shape[:2], 1, 1)
            agent_batch_input = agent_batch.unsqueeze(2)
            print("agent_batch_input.shape", agent_batch_input.shape)
            hist_out = self.hist_tf(agent_batch_input, self.query_batches, None, None)
            print("hist_out.shape", hist_out.shape)
            lane_mem = self.lane_enc(self.lane_emb(lane_batch), lane_mask)
            lane_mem = lane_mem.unsqueeze(1).repeat(1, social_num, 1, 1)
            lane_mask = lane_mask.unsqueeze(1).repeat(1, social_num, 1, 1)
            lane_out = self.lane_dec(hist_out, lane_mem, lane_mask, None)
            print("lane_out.shape:", lane_out.shape)
            dist = lane_out.view(batch_size, social_num, -1)
            dist = self.dist_emb(dist)
            social_inp = self.fusion2(torch.cat([agent_batch, dist], -1))
            social_mem = self.social_enc(social_inp, agent_mask)
            social_out = social_mem.unsqueeze(
                dim=2).repeat(1, 1, self.num_queries, 1)
            out = torch.cat([social_out, lane_out], -1)
            outputs_coord, outputs_class = self.prediction_header(out)
            outputs_coord_feature = self.out_pos_emb(outputs_coord)
            out = torch.cat([out, outputs_coord_feature], -1)
            outputs_traj = self.generator_header(out)
            print("outputs_traj:", outputs_traj.shape)
            #print("outputs_coord:", outputs_coord.shape)
            outputs_traj[:,:,:,-1,:] = outputs_coord
            outputs_centerness = self.generator_centerness(out).squeeze(-1)
        return outputs_coord, outputs_class, outputs_traj, outputs_centerness

    # @profile
    def forward(self, mapping: List[Dict], device):
        import time
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

        outputs_coord_batch, outputs_class_batch, outputs_traj_batch, outputs_centerness_batch = self.forward_encode_sub_graph(mapping, matrix, polyline_spans, device, batch_size)
        outputs_coord, coord_length = utils.merge_tensors_coor(outputs_coord_batch,hidden_size0=6, hidden_size1=2, device=device)
        outputs_class, class_length = utils.merge_tensors_class(outputs_class_batch,hidden_size0=6, device=device)
        outputs_traj, traj_length = utils.merge_tensors_traj(outputs_traj_batch, hidden_size0=6, hidden_size1=30, hidden_size2=2, device=device)
        outputs_centerness, centerness_length = utils.merge_tensors_class(outputs_centerness_batch,hidden_size0=6, device=device)
        middletime = time.time()
        output = self.decoder(mapping, batch_size, outputs_coord, outputs_class, outputs_traj, outputs_centerness, coord_length, device=device)
        endtime = time.time()
        return output
