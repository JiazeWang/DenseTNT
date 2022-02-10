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
            total_points_class = torch.cat([torch.ones(self.positive_num), torch.zeros(self.negative_num)])
            #print(positive_points.shape, negative_points.shape, total_points.shape)
            coord_i = outputs_coord[i][0]
            class_i = outputs_class[i][0]
            traj_i = outputs_traj[i][0]
            centerness_i = outputs_centerness[i][0]
            #print("coord.shape", coord_i.shape, class_i.shape, traj_i.shape)
            positive_points_class = torch.ones(self.positive_num).to(device)
            negative_points_class = torch.zeros(self.negative_num).to(device)
            gt_points = torch.from_numpy(gt_points).to(device)
            loss_i, DE_i = self.SetCriterion(positive_points, positive_points_class, negative_points_class, gt_points, coord_i, class_i, traj_i, centerness_i, device)
            loss[i] = loss_i
            DE[i][-1] = DE_i

            #print("class_i", class_i)
            index = torch.argmax(class_i).item()
            #print(index, coord_i[index], class_i[index], class_i.max())
            if not self.training:
                #print(coord_i)
                coord_i = coord_i.cpu().detach().numpy()
                class_i = class_i.cpu().detach().numpy()
                traj_i = traj_i.cpu().detach().numpy()
                pred_goals, pred_probs, pred_traj = utils.goals_NMS(mapping[i], coord_i, class_i, traj_i, self.nms_threshold)
                #print("pred_goals.shape", pred_goals.shape)
                #print("pred_probs,shape", pred_probs.shape)
                #print("pred_traj,shape", pred_traj.shape)
                #for each in pred_traj:
                #print(pred_traj.shape)
                #print(pred_goals)
                #print(target_point)
                for each in pred_traj:
                    utils.convert_to_origin_coordinate(each, i)
                pred_trajs_batch[i] = pred_traj
                pred_probs_batch[i] = pred_probs
                #for each in predict_trajs:
                #    utils.to_origin_coordinate(each, i)
                #    pred_trajs_batch.append(predict_trajs)
        #print(loss.mean())
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


    def centerness_gt(self, gt_point, coord_i):

        #print(gt_point.shape, coord_i.shape)

        batch_size = coord_i.shape[0]
        distance_all = torch.zeros([batch_size, 1])
        for i in range(0, batch_size):
            distance = torch.sqrt((gt_point[0] - coord_i[i][0]) ** 2 + (gt_point[1] - coord_i[i][1]) ** 2)
            if distance >=2:
                centerness_i = 0
            else:
                centerness_i = 1 - torch.sqrt(distance/2)
            distance_all[i] = centerness_i
        return distance_all

    def forward(self, total_points, total_points_class, negative_points_class, gt_points, coord_i, class_i, traj_i, centerness_i, device):
        #print("loss: ", total_points.shape, total_points_class.shape, coord_i.shape, class_i.shape, traj_i.shape)

        centerness_gt = self.centerness_gt(total_points[0], coord_i).to(device)
        #print("centerness_gt.shape:", centerness_gt.shape)
        indices = self.matcher(total_points, total_points_class, coord_i, class_i)
        predict_indices = indices[0][0]
        target_indices = indices[0][1]
        #print("predict_indices", predict_indices)
        predict_points = torch.stack([coord_i[i] for i in predict_indices])
        predict_class = torch.stack([class_i[i] for i in predict_indices])
        predict_traj = torch.stack([traj_i[i] for i in predict_indices])
        predict_centerness = torch.stack([centerness_i[i] for i in predict_indices])
        #print("predict_centerness.shape", predict_centerness.shape)
        target_class = torch.stack([total_points_class[i] for i in target_indices])
        target_centerness = torch.stack([centerness_gt[i] for i in target_indices])
        #print("target_centerness.shape", target_centerness.shape)
        #print("predict: ", predict_centerness)
        #print("target: ", target_centerness)
        target_centerness = target_centerness.detach()
        centerness_loss = F.binary_cross_entropy(predict_centerness.float(), target_centerness.squeeze().float())
        #print("centerness_loss:", centerness_loss)

        target_traj = gt_points.unsqueeze(0).repeat(10, 1, 1).squeeze(0)

        #v2 loss added
        target_point = total_points[0].unsqueeze(0).repeat(10, 1)
        point_loss = F.smooth_l1_loss(predict_points.float(), target_point.float())
        #point_loss = F.mse_loss(predict_points.float(), target_point.float())
        #print("point_loss", point_loss)
        #print(target_traj.shape, predict_traj.shape)

        total_shape = coord_i.shape[0]
        total_list = [i for i in range(0, total_shape)]
        negative_indices = []

        for i in total_list:
            if i not in predict_indices:
                negative_indices.append(i)
        negative_predict_class = torch.stack([class_i[i] for i in negative_indices])
        negative_class_loss = F.binary_cross_entropy(negative_predict_class.float(), negative_points_class.float())

        traj_loss = F.smooth_l1_loss(predict_traj.float(), target_traj.float())
        class_loss = F.binary_cross_entropy(predict_class.float(), target_class.float())
        total_loss = self.traj_loss_w*traj_loss+self.class_loss_w*(class_loss + negative_class_loss) + point_loss + centerness_loss
        #print("class_i", class_i.shape, centerness_i.shape)
        class_i = class_i.mul(centerness_i)
        index = torch.argmax(class_i).item()
        DE = torch.sqrt((coord_i[index][0] - gt_points[-1][0]) ** 2 + (coord_i[index][1] - gt_points[-1][1]) ** 2)
        #print("DE", DE)
        return total_loss, DE


def argmin(lst):
    return lst.index(min(lst))
