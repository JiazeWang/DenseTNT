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
        self.neighbour_dis = args.neighbour_dis
        #self.neighbour_dis = 1
        self.future_frame_num = args.future_frame_num
        self.mode_num = args.mode_num
        self.future_frame_num = 30
        self.positive_num = 10
        self.negative_num = 50 - self.positive_num
        self.SetCriterion = SetCriterion()
        self.nms_threshold = 2
        self.eval_num = 6

    def forward(self, mapping, batch_size, outputs_coord, outputs_class, outputs_traj, coord_length, device):
        #print("start")
        labels = utils.get_from_mapping(mapping, 'labels')
        labels_is_valid = utils.get_from_mapping(mapping, 'labels_is_valid')
        loss = torch.zeros(batch_size, device=device)
        #print("len labels:", len(labels))
        #print("labels_is_valid.shape", labels_is_valid[0])
        DE = np.zeros([batch_size, self.future_frame_num])
        pred_trajs_batch = np.zeros([batch_size, self.eval_num, self.future_frame_num, 2])
        pred_probs_batch = np.zeros([batch_size, self.eval_num])
        for i in range(batch_size):
            gt_points = labels[i].reshape([self.future_frame_num, 2])
            target_point = gt_points[-1]
            gt_points = torch.from_numpy(gt_points).to(device)
            coord_i = outputs_coord[i][0]
            class_i = outputs_class[i][0]
            traj_i = outputs_traj[i][0]
            loss_i, DE_i = self.SetCriterion(gt_points, target_point, coord_i, class_i, traj_i, device)
            loss[i] = loss_i
            DE[i][-1] = DE_i

            #print("class_i", class_i)
            index = torch.argmax(class_i).item()
            #print(index, coord_i[index], class_i[index], class_i.max())
            if not self.training:
                coord_i = coord_i.cpu().detach().numpy()
                class_i = class_i.cpu().detach().numpy()
                traj_i = traj_i.cpu().detach().numpy()
                pred_goals, pred_probs, pred_traj = utils.goals_NMS(mapping[i], coord_i, class_i, traj_i, self.nms_threshold)
                for each in pred_traj:
                    utils.convert_to_origin_coordinate(each, i)
                pred_trajs_batch[i] = pred_traj
                pred_probs_batch[i] = pred_probs
        if self.training:
            return loss.mean(), DE, None
        else:
            return pred_trajs_batch, pred_probs_batch, None



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
        self.negative_points_num = 40

    def distance_loss(self, gt_point, coord_i):
        batch_size = coord_i.shape[0]
        loss = torch.zeros([batch_size, 1])
        for i in range(0, batch_size):
            distance_error_i = torch.sqrt((gt_point[0] - coord_i[i][0]) ** 2 + (gt_point[1] - coord_i[i][1]) ** 2)
            loss[i] = distance_error_i
        return loss

    def forward(self, gt_points, target_point, coord_i, class_i, traj_i, device):
        #print("loss: ", total_points.shape, total_points_class.shape, coord_i.shape, class_i.shape, traj_i.shape)

        distance_loss = self.distance_loss(target_point, coord_i).to(device)

        indices = torch.argmin(distance_loss)
        predict_class = class_i[indices].reshape([1])
        predict_traj = traj_i[indices]
        predict_points = coord_i[indices]
        points_class = torch.ones(1).to(device)
        class_loss = F.binary_cross_entropy(predict_class.float(), points_class.float())
        traj_loss = F.smooth_l1_loss(predict_traj.float(), gt_points.float())
        target_point = torch.from_numpy(target_point).to(device)
        point_loss = F.mse_loss(predict_points.float(), target_point.float())
        total_loss = self.traj_loss_w*traj_loss+self.class_loss_w*class_loss + points_class

        index = torch.argmax(class_i).item()
        DE = torch.sqrt((coord_i[index][0] - gt_points[-1][0]) ** 2 + (coord_i[index][1] - gt_points[-1][1]) ** 2)
        #print("DE", DE)
        return total_loss, DE


def argmin(lst):
    return lst.index(min(lst))
