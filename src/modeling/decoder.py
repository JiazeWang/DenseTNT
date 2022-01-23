from typing import Dict, List, Tuple, NamedTuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

import structs
import utils_cython
from modeling.lib import PointSubGraph, GlobalGraphRes, CrossAttention, GlobalGraph, MLP
from modeling.matcher import build_matcher
import random
import utils


class DecoderRes(nn.Module):
    def __init__(self, hidden_size, out_features=60):
        super(DecoderRes, self).__init__()
        self.mlp = MLP(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, out_features)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.mlp(hidden_states)
        hidden_states = self.fc(hidden_states)
        return hidden_states


class DecoderResCat(nn.Module):
    def __init__(self, hidden_size, in_features, out_features=60):
        super(DecoderResCat, self).__init__()
        self.mlp = MLP(in_features, hidden_size)
        self.fc = nn.Linear(hidden_size + in_features, out_features)

    def forward(self, hidden_states):
        hidden_states = torch.cat([hidden_states, self.mlp(hidden_states)], dim=-1)
        hidden_states = self.fc(hidden_states)
        return hidden_states


class Decoder_predict(nn.Module):

    def __init__(self, args_: utils.Args, vectornet):
        super(Decoder_predict, self).__init__()
        global args
        args = args_
        hidden_size = args.hidden_size
        self.future_frame_num = args.future_frame_num
        self.mode_num = args.mode_num
        self.future_frame_num = 30
        self.positive_num = 10
        self.negative_num = 50 - self.positive_num
        self.SetCriterion = SetCriterion()


    def forward(self, mapping, batch_size, outputs_coord, outputs_class, outputs_traj, coord_length, device):
        #print("start")
        labels = utils.get_from_mapping(mapping, 'labels')
        labels_is_valid = utils.get_from_mapping(mapping, 'labels_is_valid')
        loss = torch.zeros(batch_size, device=device)
        #print("len labels:", len(labels))
        #print("labels_is_valid.shape", labels_is_valid[0])
        DE = np.zeros([batch_size, self.future_frame_num])

        for i in range(batch_size):
            gt_points = labels[i].reshape([self.future_frame_num, 2])
            target_point = gt_points[-1]
            #print("target_point", target_point)
            #print("gt_points",gt_points.shape)
            goals_2D = mapping[i]['goals_2D']
            #print("goals_2D", len(goals_2D))
            result_t = []
            dis = utils.get_dis_point_2_points(target_point, goals_2D)
            positive_points = []
            positive_index = (dis<2).nonzero()[0].tolist()
            #print(positive_index)
            goals_2D_index = [i for i in range(len(goals_2D))]
            goals_2D_index = set(goals_2D_index)
            positive_index = set(positive_index)
            negative_index = set(goals_2D_index-positive_index)
            negative_points = [goals_2D[i] for i in negative_index]
            positive_points = [goals_2D[i] for i in positive_index]
            #print(len(goals_2D), len(positive_points), len(negative_points))
            negative_points = random.sample(negative_points, self.negative_num)
            positive_points = utils.get_neighbour_points_positive(target_point, num = self.positive_num)
            positive_points = torch.from_numpy(np.array(positive_points)).cuda()
            negative_points = torch.from_numpy(np.array(negative_points)).cuda()
            total_points = torch.cat([positive_points, negative_points], dim=0)
            total_points_class = torch.cat([torch.ones(self.positive_num), torch.zeros(self.negative_num)])
            #print(positive_points.shape, negative_points.shape, total_points.shape)
            coord_i = outputs_coord[i][0]
            class_i = outputs_class[i][0]
            traj_i = outputs_traj[i][0]
            #print("coord.shape", coord_i.shape, class_i.shape, traj_i.shape)
            positive_points_class = torch.ones(self.positive_num).cuda()
            gt_points = torch.from_numpy(gt_points).cuda()
            loss_i, DE_i = self.SetCriterion(positive_points, positive_points_class, gt_points, coord_i, class_i, traj_i)
            loss[i] = loss_i
            DE[i][-1] = DE_i
        #print(loss.mean())
        return loss.mean(), DE, None



class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.matcher = build_matcher()
        self.losses = 0
        self.traj_loss_w = 1
        self.class_loss_w = 1

    def forward(self, total_points, total_points_class, gt_points, coord_i, class_i, traj_i):
        #print("loss: ", total_points.shape, total_points_class.shape, coord_i.shape, class_i.shape, traj_i.shape)


        indices = self.matcher(total_points, total_points_class, coord_i, class_i)
        predict_indices = indices[0][0]
        target_indices = indices[0][1]

        predict_points = torch.stack([coord_i[i] for i in predict_indices])
        predict_class = torch.stack([class_i[i] for i in predict_indices])
        predict_traj = torch.stack([traj_i[i] for i in predict_indices])

        target_point = torch.stack([total_points[i] for i in target_indices])
        target_class = torch.stack([total_points_class[i] for i in target_indices])
        target_traj = gt_points.unsqueeze(0).repeat(10, 1, 1).squeeze(0)
        #print(target_traj.shape, predict_traj.shape)
        traj_loss = F.smooth_l1_loss(predict_traj.float(), target_traj.float())
        #print("traj_loss", traj_loss)
        #print(predict_class.unsqueeze(1).shape, target_class.shape)
        class_loss = F.binary_cross_entropy(predict_class.float(), target_class.float())
        #print("class_loss", class_loss)
        total_loss = self.traj_loss_w*traj_loss+self.class_loss_w*class_loss
        index = torch.argmax(predict_class).item()
        DE = torch.sqrt((predict_points[index][0] - gt_points[-1][0]) ** 2 + (predict_points[index][0] - gt_points[-1][1]) ** 2)
        #print("DE", DE)
        return total_loss, DE


def argmin(lst):
    return lst.index(min(lst))
