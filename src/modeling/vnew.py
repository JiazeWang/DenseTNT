from typing import Dict, List, Tuple, NamedTuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from modeling.decoder import Decoder, DecoderResCat
from modeling.lib import MLP, GlobalGraph, LayerNorm, SubGraph, CrossAttention, GlobalGraphRes
import utils


class NewSubGraph(nn.Module):

    def __init__(self, hidden_size, depth=None):
        super(NewSubGraph, self).__init__()
        if depth is None:
            depth = args.sub_graph_depth
        self.layers = nn.ModuleList([MLP(hidden_size, hidden_size // 2) for _ in range(depth)])
        if 'point_level-4' in args.other_params:
            self.layer_0 = MLP(hidden_size)
            self.layers = nn.ModuleList([GlobalGraph(hidden_size, num_attention_heads=2) for _ in range(depth)])
            self.layers_2 = nn.ModuleList([LayerNorm(hidden_size) for _ in range(depth)])
            self.layers_3 = nn.ModuleList([LayerNorm(hidden_size) for _ in range(depth)])
            self.layers_4 = nn.ModuleList([GlobalGraph(hidden_size) for _ in range(depth)])
            if 'point_level-4-3' in args.other_params:
                self.layer_0_again = MLP(hidden_size)

    def forward(self, input_list: list):
        batch_size = len(input_list)
        device = input_list[0].device
        hidden_states, lengths = utils.merge_tensors(input_list, device)
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
        else:
            attention_mask = torch.zeros([batch_size, max_vector_num, hidden_size // 2], device=device)
            for i in range(batch_size):
                assert lengths[i] > 0
                attention_mask[i][lengths[i]:max_vector_num].fill_(-10000.0)

            zeros = torch.zeros([hidden_size // 2], device=device)

            for layer_index, layer in enumerate(self.layers):
                new_hidden_states = torch.zeros([batch_size, max_vector_num, hidden_size], device=device)
                encoded_hidden_states = layer(hidden_states)

                if True:
                    max_hidden, _ = torch.max(encoded_hidden_states + attention_mask, dim=1)
                    max_hidden = torch.max(max_hidden, zeros.unsqueeze(0).expand_as(max_hidden))
                    new_hidden_states = torch.cat(
                        (encoded_hidden_states, max_hidden.unsqueeze(1).expand_as(encoded_hidden_states)), dim=-1)
                # for j in range(max_vector_num):
                #     attention_mask[:, j] += -10000.0
                #     max_hidden, _ = torch.max(encoded_hidden_states + attention_mask, dim=1)
                #     max_hidden = torch.max(max_hidden, zeros)
                #     attention_mask[:, j] += 10000.0
                #     new_hidden_states[:, j] = torch.cat((encoded_hidden_states[:, j], max_hidden), dim=-1)
                hidden_states = new_hidden_states

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
        if 'point_level' in args.other_params:
            # TODO!  tnt needs stage_one?
            # assert 'stage_one' in args.other_params
            self.point_level_sub_graph = NewSubGraph(hidden_size)
            self.point_level_cross_attention = CrossAttention(hidden_size)

        self.global_graph = GlobalGraph(hidden_size)
        if 'enhance_global_graph' in args.other_params:
            self.global_graph = GlobalGraphRes(hidden_size)
        if 'laneGCN' in args.other_params:
            self.laneGCN_A2L = CrossAttention(hidden_size)
            self.laneGCN_L2L = GlobalGraphRes(hidden_size)
            self.laneGCN_L2A = CrossAttention(hidden_size)
            self.agents_A = CrossAttention(hidden_size)
            self.agents_A0 = CrossAttention(hidden_size)
        if 'stage_one' in args.other_params:
            self.stage_one_sub_graph_map = SubGraph(args, hidden_size)

        self.decoder = Decoder(args, self)

        if 'complete_traj' in args.other_params:
            self.decoder.complete_traj_cross_attention = CrossAttention(hidden_size)
            self.decoder.complete_traj_decoder = DecoderResCat(hidden_size, hidden_size * 3, out_features=self.decoder.future_frame_num * 2)

    def forward_encode_sub_graph(self, mapping: List[Dict], matrix: List[np.ndarray], polyline_spans: List[List[slice]],
                                 device, batch_size) -> Tuple[List[Tensor], List[Tensor]]:
        """
        :param matrix: each value in list is vectors of all element (shape [-1, 128])
        :param polyline_spans: vectors of i_th element is matrix[polyline_spans[i]]
        :return: hidden states of all elements and hidden states of lanes
        """
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

        if 'point_level' in args.other_params:
            element_states_batch = []
            point_level_features_list = []
            for i in range(batch_size):
                a, b = self.point_level_sub_graph(input_list_list[i])
                element_states_batch.append(a)
                point_level_features_list.append(b)
                mapping[i]['point_level_features'] = point_level_features_list[i]
        else:
            element_states_batch = utils.merge_tensors_not_add_dim(input_list_list, module=self.sub_graph,
                                                                   sub_batch_size=16, device=device)

        if 'stage_one' in args.other_params:
            assert 'sub_graph_map' not in args.other_params
            lane_states_batch = utils.merge_tensors_not_add_dim(map_input_list_list,
                                                                module=self.stage_one_sub_graph_map,
                                                                sub_batch_size=16, device=device)

        if 'laneGCN' in args.other_params:
            inputs_before_laneGCN, inputs_lengths_before_laneGCN = utils.merge_tensors(element_states_batch, device=device)
            for i in range(batch_size):
                map_start_polyline_idx = mapping[i]['map_start_polyline_idx']
                agents = element_states_batch[i][:map_start_polyline_idx]
                lanes = element_states_batch[i][map_start_polyline_idx:]
                if 'laneGCN-4' in args.other_params:
                    #print("lanes.unsqueeze(0).shape, torch.cat([lanes, agents[0:1]]).shape: ",lanes.unsqueeze(0).shape, torch.cat([lanes, agents[0:1]]).shape)
                    #jiaze change here
                    #lanes = lanes + self.laneGCN_A2L(lanes.unsqueeze(0), torch.cat([lanes, agents[0:1]]).unsqueeze(0)).squeeze(0)
                    lanes = lanes + self.laneGCN_A2L(lanes.unsqueeze(0), torch.cat([lanes, agents[0:1]]).unsqueeze(0)).squeeze(0)
                    agents = agents + self.laneGCN_L2A(agents.unsqueeze(0), lanes.unsqueeze(0)).squeeze(0)
                    agents0 = (agents[0:1].unsqueeze(0) + self.agents_A(agents[0:1].unsqueeze(0), agents[1:].unsqueeze(0))).squeeze(0)
                    agents1 = (agents[1:].unsqueeze(0) + self.agents_A0(agents[1:].unsqueeze(0), agents[0:1].unsqueeze(0))).squeeze(0)
                    agents = torch.cat([agents0, agents1])
                    #lanes = lanes + self.laneGCN_A2L(lanes.unsqueeze(0), torch.cat([lanes, agents[0:1]]).unsqueeze(0)).squeeze(0)

                else:
                    lanes = lanes + self.laneGCN_A2L(lanes.unsqueeze(0), agents.unsqueeze(0)).squeeze(0)
                    lanes = lanes + self.laneGCN_L2L(lanes.unsqueeze(0)).squeeze(0)
                    agents = agents + self.laneGCN_L2A(agents.unsqueeze(0), lanes.unsqueeze(0)).squeeze(0)
                #print("agents.shape", agents.shape)
                #print("lanes.shape", lanes.shape)
                element_states_batch[i] = torch.cat([agents, lanes])

        return element_states_batch, lane_states_batch

    # @profile
    def forward(self, mapping: List[Dict], device):
        import time
        global starttime
        starttime = time.time()
        #print("MAPPING")
        #print(mapping)
        matrix = utils.get_from_mapping(mapping, 'matrix')
        # TODO(cyrushx): Can you explain the structure of polyline spans?
        # vectors of i_th element is matrix[polyline_spans[i]]
        polyline_spans = utils.get_from_mapping(mapping, 'polyline_spans')
        batch_size = len(matrix)
        # for i in range(batch_size):
        # polyline_spans[i] = [slice(polyline_span[0], polyline_span[1]) for polyline_span in polyline_spans[i]]

        if args.argoverse:
            utils.batch_init(mapping)

        element_states_batch, lane_states_batch = self.forward_encode_sub_graph(mapping, matrix, polyline_spans, device, batch_size)
        #print("element_states_batch0.shape: ", element_states_batch[0].shape)
        #print("element_states_batch1.shape: ", element_states_batch[1].shape)
        #print("element_states_batch2    .shape: ", element_states_batch[2].shape)
        #print("lane_states_batch0.shape: ", lane_states_batch[0].shape)
        #print("lane_states_batch1.shape: ", lane_states_batch[1].shape)
        #print("lane_states_batch2.shape: ", lane_states_batch[2].shape)
        #print("lane_states_batch.shape: ", lane_states_batch.shape)
        inputs, inputs_lengths = utils.merge_tensors(element_states_batch, device=device)
        max_poly_num = max(inputs_lengths)
        attention_mask = torch.zeros([batch_size, max_poly_num, max_poly_num], device=device)
        for i, length in enumerate(inputs_lengths):
            attention_mask[i][:length][:length].fill_(1)

        hidden_states = self.global_graph(inputs, attention_mask, mapping)
        utils.logging('time3', round(time.time() - starttime, 2), 'secs')
        #print("inputs.shape: ", inputs.shape)
        #print("inputs_length: ", inputs_lengths)
        #print("hidden_states.shape: ", hidden_states.shape)
        output = self.decoder(mapping, batch_size, lane_states_batch, inputs, inputs_lengths, hidden_states, device)
        #print("len(output):", len(output))
        #print("output[0].shape: ", output[0])
        #print("output[1].shape: ", output[1].shape)
        #print("output[2]: ", output[2])
        return output
