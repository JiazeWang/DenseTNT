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
        self.neighbour_dis = 2
        #self.neighbour_dis = 1
        self.future_frame_num = args.future_frame_num
        self.mode_num = args.mode_num
        self.future_frame_num = 30
        self.positive_num = 6
        self.negative_num = 6-self.positive_num
        self.SetCriterion = SetCriterion()
        self.nms_threshold = 2
        self.eval_num = 6

    def forward(self, mapping, batch_size, outputs_coord, outputs_class, outputs_traj, outputs_centerness, coord_length, device):
        #print("start")
        labels = utils.get_from_mapping(mapping, 'labels')
        labels_is_valid = utils.get_from_mapping(mapping, 'labels_is_valid')
        loss = torch.zeros(batch_size, device=device)
        #print("len labels:", len(labels))
        #print("labels_is_valid.shape", labels_is_valid[0])
        DE = np.zeros([batch_size, self.future_frame_num])
        pred_trajs_batch = np.zeros([batch_size, self.eval_num, self.future_frame_num, 2])
        pred_probs_batch = np.zeros([batch_size, self.eval_num])

        if not self.training:
            for i in range(batch_size):
                coord_i = outputs_coord[i][0]
                class_i = outputs_class[i][0]
                traj_i = outputs_traj[i][0]
                centerness_i = outputs_centerness[i][0]
                print("class_i", class_i)
                print("centerness_i", centerness_i)
                class_i = class_i.mul(centerness_i)
                index = torch.argmax(class_i).item()
                with open("result.txt", 'a') as file:
                    file.write(str(index)+"\n")
                coord_i = coord_i.cpu().detach().numpy()
                class_i = class_i.cpu().detach().numpy()
                traj_i = traj_i.cpu().detach().numpy()
                pred_goals, pred_probs, pred_traj = utils.goals_NMS(mapping[i], coord_i, class_i, traj_i, self.nms_threshold)
                for each in pred_traj:
                    utils.convert_to_origin_coordinate(each, i)
                pred_trajs_batch[i] = pred_traj
                pred_probs_batch[i] = pred_probs
            return pred_trajs_batch, pred_probs_batch, None

        else:
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
                positive_points = utils.get_neighbour_points_positive(target_point, num = self.positive_num, neighbour_dis=self.neighbour_dis)
                positive_points = torch.from_numpy(np.array(positive_points)).to(device)
                negative_points = torch.from_numpy(np.array(negative_points)).to(device)
                total_points = torch.cat([positive_points, negative_points], dim=0)
                total_points_class = torch.cat([torch.ones(self.positive_num), torch.zeros(self.negative_num)]).to(device)
                #print(positive_points.shape, negative_points.shape, total_points.shape)
                coord_i = outputs_coord[i][0]
                class_i = outputs_class[i][0]
                traj_i = outputs_traj[i][0]
                #centerness added
                centerness_i = outputs_centerness[i][0]
                #print("coord.shape", coord_i.shape, class_i.shape, traj_i.shape)
                positive_points_class = torch.ones(self.positive_num).to(device)
                negative_points_class = torch.zeros(self.negative_num).to(device)
                gt_points = torch.from_numpy(gt_points).to(device)
                loss_i, DE_i = self.SetCriterion(total_points, total_points_class, negative_points_class, gt_points, coord_i, class_i, traj_i, centerness_i, device)
                loss[i] = loss_i
                DE[i][-1] = DE_i

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
        self.positive_points_num = 6
        self.negative_points_num = 6-6
        self.temper = 0.01

    def distance_loss(self, gt_point, coord_i):
        batch_size = coord_i.shape[0]
        centerness = torch.zeros([batch_size, 1])
        distance_all = torch.zeros([batch_size, 1])
        for i in range(0, batch_size):
            distance_error_i = torch.sqrt((gt_point[0] - coord_i[i][0]) ** 2 + (gt_point[1] - coord_i[i][1]) ** 2)
            if distance_error_i >=2:
                centerness_i = 0
            else:
                centerness_i = 1 - torch.sqrt(distance/2)
            centerness[i] = centerness_i
            distance_all[i]  = distance_error_i
        return centerness, distance_all

    def forward(self, total_points, total_points_class, negative_points_class, gt_points, coord_i, class_i, traj_i, centerness_i, device):
        #print("loss: ", total_points.shape, total_points_class.shape, coord_i.shape, class_i.shape, traj_i.shape)
        #centerness_gt = self.centerness_gt(total_points[0], coord_i).to(device)
        target_point = total_points[0]
        centerness_gt, distance_loss = self.distance_loss(target_point.to(device), coord_i.to(device))
        centerness_gt =centerness_gt.to(device)
        distance_loss = distance_loss.to(device)
        #centerness_gt = centerness_gt.to(device)
        #distance_loss = distance_loss.to(device)
        indices = torch.argmin(distance_loss)
        predict_class = class_i
        predict_traj = traj_i[indices]
        predict_points = coord_i[indices]
        predict_centerness = centerness_i[indices]


        points_class = torch.ones(6).to(device)
        print("predict_class:", predict_class.shape, "points_class:", points_class.shape)
        class_loss = F.binary_cross_entropy(predict_class.float(), points_class.float())
        print("predict_traj:", predict_traj.shape, "gt_points:", gt_points.shape)
        traj_loss = F.smooth_l1_loss(predict_traj.float(), gt_points.float())
        target_centerness = centerness_gt[indices].detach()
        print("predict_centerness:",predict_centerness.shape, "target_centerness:",target_centerness.shape)
        centerness_loss = F.binary_cross_entropy(predict_centerness.float(), target_centerness.squeeze().float())
        print("predict_points:",predict_points.shape,"target_point:",target_point.shape)
        point_loss = F.smooth_l1_loss(predict_points.float(), target_point.float())
        total_loss = self.traj_loss_w*traj_loss+self.class_loss_w*class_loss + points_loss + centerness_loss

        class_i = class_i.mul(centerness_i)
        index = torch.argmax(class_i).item()
        DE = torch.sqrt((coord_i[index][0] - gt_points[-1][0]) ** 2 + (coord_i[index][1] - gt_points[-1][1]) ** 2)
        #print("DE", DE)
        return total_loss, DE


def argmin(lst):
    return lst.index(min(lst))
