import torch
from scipy.optimize import linear_sum_assignment
from torch import nn



class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox

    @torch.no_grad()
    def forward(self, total_points, total_points_class, coord_i, class_i):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """

        bs = 1
        num_queries = coord_i.shape[0]

        # We flatten to compute the cost matrices in a batch
        out_prob = class_i  # [batch_size * num_queries, num_classes]
        out_bbox = coord_i  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.ones(6)*6
        tgt_bbox = total_points

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        #cost_class = torch.dist(out_bbox, tgt_bbox, p=1)
        out_prob = out_prob.unsqueeze(-1)
        cost_class = (1-out_prob).repeat(1, 36).unsqueeze(0)
        #print("cost_class.shape", cost_class.shape)
        #print(cost_class)
        # Compute the L1 cost between boxes
        out_bbox = out_bbox.unsqueeze(0).to(torch.float32)
        tgt_bbox = tgt_bbox.unsqueeze(0).to(torch.float32)
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        #print("cost_bbox.shape", cost_bbox.shape)
        # Final cost matrix
        #print("cost_bbox", cost_bbox.max(), cost_bbox.min())
        #print("cost_class", cost_class.max(), cost_class.min())
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class
        #print("C.shape", C.shape)
        C = C.view(bs, num_queries, -1).cpu()
        sizes = 36
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        #print("indices.shape", indices)
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


def build_matcher():
    return HungarianMatcher()
